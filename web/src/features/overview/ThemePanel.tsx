import { useMemo, useState } from 'react';
import { KeyRound, Sparkles } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { topQuotientsByTheme, type QuotientEntry } from '@/core/locational-quotient';
import { numberLocale } from '@/lib/i18n/labels';
import { FIELD } from '@/lib/schema';
import { useAiConfig } from '@/state/ai-config.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { AiSettingsModal } from '@/components/AiSettingsModal';
import { ReadingTip } from '@/components/InfoTip';
import { HybridPanel } from './hybrid/HybridPanel';

export function ThemePanel() {
  const active = useDataset((state) => state.active);
  const clustering = useDataset((state) => state.clustering);
  const hybridRun = useDataset((state) => state.hybridRun);
  const hasThemes = clustering !== null || hybridRun !== null;
  const categorize = useDataset((state) => state.categorizeThemes);
  const isCategorizingThemes = useDataset((state) => state.isCategorizingThemes);
  const busy = isCategorizingThemes;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  // O rótulo vem pronto do core como "QL"; em inglês a sigla é "LQ" (location quotient).
  const qlLabel = (entry: QuotientEntry | undefined) =>
    entry && locale === 'en' ? `${entry.entity} (LQ: ${entry.quotient.toFixed(2)})` : entry?.label;
  const isAiConfigured = useAiConfig((state) => state.isConfigured());
  const [aiModalOpen, setAiModalOpen] = useState(false);

  const themes = useMemo(() => {
    if (!active || !hasThemes) return [];

    const counts = new Map<string, number>();
    for (const doc of active) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme) counts.set(theme, (counts.get(theme) ?? 0) + 1);
    }

    return [...counts.entries()]
      .map(([name, documents]) => ({ name, documents }))
      .sort((left, right) => right.documents - left.documents);
  }, [active, hasThemes]);

  const quotients = useMemo(
    () => (active && hasThemes ? topQuotientsByTheme(active) : null),
    [active, hasThemes],
  );

  if (!active) return null;

  return (
    <>
      <div className="space-y-4">
          {!isAiConfigured && (
            <div className="rounded-xl border border-purple-200 bg-purple-50/70 p-3 text-xs text-purple-900 flex flex-wrap items-center justify-between gap-2 dark:border-purple-900 dark:bg-purple-950/40 dark:text-purple-300">
              <div className="flex items-center gap-2">
                <KeyRound className="size-4 shrink-0 text-purple-600" />
                <span>{t('theme_no_key_warning')}</span>
              </div>
              <Button
                variant="ai"
                size="sm"
                onClick={() => setAiModalOpen(true)}
                className="h-7 text-xs font-bold"
              >
                {t('ai_settings_btn')}
              </Button>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="ai"
              onClick={() => void categorize()}
              disabled={busy}
              className="font-semibold shadow-xs"
            >
              <Sparkles className="size-4" aria-hidden />
              {clustering ? t('theme_btn_recalc') : t('theme_btn_identify')}
            </Button>

            {isAiConfigured && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAiModalOpen(true)}
                className="ml-auto gap-1.5 text-xs"
              >
                <KeyRound className="size-3.5" aria-hidden />
                <span>{t('ai_configured')}</span>
              </Button>
            )}

            {clustering && (
              <span className="inline-flex items-center gap-2 rounded-full border border-purple-200 bg-purple-50 px-3 py-1 text-xs font-medium text-purple-800 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-300">
                <span className="size-1.5 rounded-full bg-purple-500" />
                {clustering.clusterCount} {t('theme_clusters_found')} · Silhouette{' '}
                <strong className="tabular-nums">{clustering.silhouette.toFixed(3)}</strong>
              </span>
            )}
          </div>

          {themes.length > 0 && (
            <div className="overflow-x-auto border duration-300 animate-in fade-in-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('theme_table_theme')}</TableHead>
                    <TableHead>{t('theme_table_docs')}</TableHead>
                    <TableHead>{t('theme_table_top_author')}</TableHead>
                    <TableHead>{t('theme_table_top_country')}</TableHead>
                    <TableHead>{t('theme_table_top_venue')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {themes.map((theme) => (
                    <TableRow key={theme.name}>
                      <TableCell className="font-medium text-foreground">{theme.name}</TableCell>
                      <TableCell>
                        <Badge variant="purple" className="tabular-nums font-semibold">
                          {theme.documents.toLocaleString(numberLocale(locale))}
                        </Badge>
                      </TableCell>
                      <TableCell
                        className="max-w-56 truncate text-xs text-muted-foreground"
                        title={qlLabel(quotients?.authors.get(theme.name))}
                      >
                        {qlLabel(quotients?.authors.get(theme.name)) ?? '—'}
                      </TableCell>
                      <TableCell
                        className="max-w-48 truncate text-xs text-muted-foreground"
                        title={qlLabel(quotients?.countries.get(theme.name))}
                      >
                        {qlLabel(quotients?.countries.get(theme.name)) ?? '—'}
                      </TableCell>
                      <TableCell
                        className="max-w-64 truncate text-xs text-muted-foreground"
                        title={qlLabel(quotients?.venues.get(theme.name))}
                      >
                        {qlLabel(quotients?.venues.get(theme.name)) ?? '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {hasThemes && <ReadingTip>{t('theme_ql_explanation')}</ReadingTip>}

          <HybridPanel />
      </div>

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </>
  );
}
