import { useMemo, useState } from 'react';
import { KeyRound, Sparkles } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { topQuotientsByTheme } from '@/core/locational-quotient';
import { FIELD } from '@/lib/schema';
import { useAiConfig } from '@/state/ai-config.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { AiSettingsModal } from '@/components/AiSettingsModal';

export function ThemePanel() {
  const active = useDataset((state) => state.active);
  const clustering = useDataset((state) => state.clustering);
  const categorize = useDataset((state) => state.categorizeThemes);
  const isCategorizingThemes = useDataset((state) => state.isCategorizingThemes);
  const busy = isCategorizingThemes;
  const t = useLocale((state) => state.t);
  const isAiConfigured = useAiConfig((state) => state.isConfigured());
  const [aiModalOpen, setAiModalOpen] = useState(false);

  const themes = useMemo(() => {
    if (!active || !clustering) return [];

    const counts = new Map<string, number>();
    for (const doc of active) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme) counts.set(theme, (counts.get(theme) ?? 0) + 1);
    }

    return [...counts.entries()]
      .map(([name, documents]) => ({ name, documents }))
      .sort((left, right) => right.documents - left.documents);
  }, [active, clustering]);

  const quotients = useMemo(
    () => (active && clustering ? topQuotientsByTheme(active) : null),
    [active, clustering],
  );

  if (!active) return null;

  return (
    <>
      <Card className="border-t-4 border-t-purple-500 bg-gradient-to-br from-purple-500/[0.03] via-card to-card shadow-xs">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2.5 text-base font-bold text-foreground">
              <div className="flex size-7 items-center justify-center rounded-lg bg-purple-100 text-purple-600 shadow-2xs dark:bg-purple-950 dark:text-purple-400">
                <Sparkles className="size-4" aria-hidden />
              </div>
              {t('theme_title')}
            </CardTitle>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setAiModalOpen(true)}
              className="gap-1.5 rounded-lg text-xs font-semibold"
            >
              <KeyRound className="size-3.5 text-purple-600" />
              <span>{isAiConfigured ? t('ai_configured') : t('ai_settings_btn')}</span>
            </Button>
          </div>
          <CardDescription>
            {t('theme_description')}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
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

            {clustering && (
              <span className="inline-flex items-center gap-2 rounded-full border border-purple-200 bg-purple-50 px-3 py-1 text-xs font-medium text-purple-800 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-300">
                <span className="size-1.5 rounded-full bg-purple-500" />
                {clustering.clusterCount} {t('theme_clusters_found')} · Silhouette{' '}
                <strong className="tabular-nums">{clustering.silhouette.toFixed(3)}</strong>
              </span>
            )}
          </div>

          {themes.length > 0 && (
            <div className="overflow-x-auto rounded-xl border">
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
                          {theme.documents.toLocaleString('pt-BR')}
                        </Badge>
                      </TableCell>
                      <TableCell
                        className="max-w-56 truncate text-xs text-muted-foreground"
                        title={quotients?.authors.get(theme.name)?.label}
                      >
                        {quotients?.authors.get(theme.name)?.label ?? '—'}
                      </TableCell>
                      <TableCell
                        className="max-w-48 truncate text-xs text-muted-foreground"
                        title={quotients?.countries.get(theme.name)?.label}
                      >
                        {quotients?.countries.get(theme.name)?.label ?? '—'}
                      </TableCell>
                      <TableCell
                        className="max-w-64 truncate text-xs text-muted-foreground"
                        title={quotients?.venues.get(theme.name)?.label}
                      >
                        {quotients?.venues.get(theme.name)?.label ?? '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {clustering && (
            <p className="text-xs text-muted-foreground">
              {t('theme_ql_explanation')}
            </p>
          )}
        </CardContent>
      </Card>

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </>
  );
}
