import { useState } from 'react';
import { ArrowRight, BarChart3, CheckCircle2, ListChecks, Search } from 'lucide-react';

import { Collapse } from '@/components/Collapse';
import { SectionTitle } from '@/components/InfoTip';
import { UploadPanel } from '@/components/UploadPanel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { numberLocale } from '@/lib/i18n/labels';
import { useStickyValue } from '@/lib/use-sticky-value';
import { useDataset, type DedupStrategy } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { useReviewNav } from '@/state/review.store';

/**
 * Tela Dados — o primeiro passo da jornada: importar, deduplicar e seguir para um módulo.
 * Os módulos (Motor de Busca, Análise Bibliométrica, Revisão) só abrem com a base pronta.
 */
export default function DataTab() {
  const active = useDataset((state) => state.active);
  const t = useLocale((state) => state.t);

  if (!active) {
    return (
      <div className="space-y-4">
        <UploadPanel />
        <div className="space-y-4 border border-dashed border-border p-10 text-center">
          <SectionTitle
            className="justify-center"
            title={t('empty_start_title')}
            info={
              <>
                <p>{t('empty_start_desc')}</p>
                <p>{t('empty_client_note')}</p>
              </>
            }
          />
          {/* A revisão começa antes da busca: o protocolo não precisa de base. */}
          <p className="text-sm text-muted-foreground">
            {t('data_protocol_first')}{' '}
            <button
              type="button"
              onClick={() => {
                useReviewNav.getState().setStep('protocol');
                useNavigation.getState().setActiveTab('review');
              }}
              className="cursor-pointer font-medium text-highlight underline-offset-4 hover:underline"
            >
              {t('data_protocol_first_btn')} →
            </button>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <UploadPanel />
      <DedupCard />
      <NextSteps />
    </div>
  );
}

/** Com a base pronta, os caminhos: explorar no Motor de Busca ou seguir para uma análise. */
function NextSteps() {
  const t = useLocale((state) => state.t);
  const setActiveTab = useNavigation((state) => state.setActiveTab);
  return (
    <Card>
      <CardHeader className="pb-3">
        <SectionTitle title={t('data_next_title')} info={t('data_next_desc')} />
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2.5">
        <Button variant="gradient" onClick={() => setActiveTab('search')} className="cursor-pointer">
          <Search className="size-4" aria-hidden />
          {t('tab_search')}
          <ArrowRight className="size-4" aria-hidden />
        </Button>
        <Button variant="outline" onClick={() => setActiveTab('bibliometrics')} className="cursor-pointer">
          <BarChart3 className="size-4" aria-hidden />
          {t('tab_bibliometrics')}
        </Button>
        <Button variant="outline" onClick={() => setActiveTab('review')} className="cursor-pointer">
          <ListChecks className="size-4" aria-hidden />
          {t('tab_review')}
        </Button>
      </CardContent>
    </Card>
  );
}

function DedupCard() {
  const duplicates = useDataset((state) => state.duplicates);
  const shownDuplicates = useStickyValue(duplicates.length > 0 ? duplicates : null).value ?? [];
  const dedupStrategy = useDataset((state) => state.dedupStrategy);
  const applyDedup = useDataset((state) => state.applyDedup);
  const isDeduplicating = useDataset((state) => state.isDeduplicating);
  const isIngesting = useDataset((state) => state.isIngesting);
  const isDemo = useDataset((state) => state.isDemo);
  const busy = isDeduplicating || isIngesting;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const isEn = locale === 'en';
  const nf = numberLocale(locale);

  const [selectedStrategy, setSelectedStrategy] = useState<DedupStrategy>(dedupStrategy);
  // Estratégia executada por um clique neste card: se não removeu nada, o card avisa.
  const [ranStrategy, setRanStrategy] = useState<DedupStrategy | null>(null);
  const run = async (strategy: DedupStrategy): Promise<void> => {
    await applyDedup(strategy);
    setRanStrategy(useDataset.getState().error ? null : strategy);
  };
  const nothingFound =
    !busy && ranStrategy !== null && ranStrategy !== 'none' && ranStrategy === dedupStrategy && duplicates.length === 0;

  // A estratégia aplicada mudou (outra base, outro projeto): o seletor acompanha. Ajuste
  // durante o render, e não num efeito — o efeito renderizava duas vezes a cada troca.
  const [appliedStrategy, setAppliedStrategy] = useState(dedupStrategy);
  if (appliedStrategy !== dedupStrategy) {
    setAppliedStrategy(dedupStrategy);
    setSelectedStrategy(dedupStrategy);
  }

  const dedupLabels: Record<DedupStrategy, string> = {
    none: t('dedup_none'),
    doi: t('dedup_doi'),
    similarity: t('dedup_similarity'),
    both: t('dedup_both'),
  };

  return (
  <Card data-tour="dedup">
    <CardHeader className="pb-3">
      <SectionTitle title={t('dedup_title')} info={t('dedup_description')} />
    </CardHeader>
    <CardContent className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="w-72 sm:w-80" title={isDemo ? t('demo_readonly_hint') : undefined}>
          <Select
            value={selectedStrategy}
            onValueChange={(val) => {
              const strategy = val as DedupStrategy;
              setSelectedStrategy(strategy);
              // "Base completa" não tem o que executar: escolhê-la já desfaz a deduplicação.
              if (strategy === 'none' && dedupStrategy !== 'none') void applyDedup('none');
            }}
            disabled={busy || isDemo}
          >
            <SelectTrigger className="h-9" aria-label={t('dedup_strategy_aria')}>
              <SelectValue placeholder={t('dedup_strategy_label')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">{t('dedup_none')}</SelectItem>
              <SelectItem value="doi">{t('dedup_doi')}</SelectItem>
              <SelectItem value="similarity">{t('dedup_similarity')}</SelectItem>
              <SelectItem value="both">{t('dedup_both')}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {selectedStrategy !== 'none' && (
          <Button
            size="sm"
            disabled={busy}
            onClick={() => void run(selectedStrategy)}
            className="cursor-pointer"
          >
            {t('dedup_execute_btn')}
          </Button>
        )}

        {dedupStrategy !== 'none' && <Badge variant="blue">{dedupLabels[dedupStrategy]}</Badge>}

        {duplicates.length > 0 && (
          <Badge variant="warning">
            {duplicates.length.toLocaleString(nf)} {t('dedup_removed')}
          </Badge>
        )}
      </div>

      {/* Sempre montado: uma região viva que nasce junto com o texto não é anunciada. */}
      <p
        role="status"
        className={
          nothingFound
            ? 'flex items-center gap-2 rounded-md border border-include/40 bg-include/5 px-3 py-2 text-sm text-foreground animate-in fade-in-0 duration-300'
            : 'sr-only'
        }
      >
        {nothingFound && (
          <>
            <CheckCircle2 className="size-4 shrink-0 text-include" aria-hidden />
            {t('dedup_nothing_found').replace('{criterion}', t(`dedup_criterion_${ranStrategy as Exclude<DedupStrategy, 'none'>}`))}
          </>
        )}
      </p>

      {/* O relatório abre e fecha animado; durante o fechamento mostra a última lista. */}
      <Collapse open={duplicates.length > 0} delayOpen={false}>
          <div className="space-y-3 pt-2">
            <SectionTitle
              title={isEn ? 'Removed documents report' : 'Relatório de documentos excluídos'}
              info={
                isEn
                  ? 'Each row shows the removed document and the one kept in its place.'
                  : 'Cada linha indica o documento removido e qual foi mantido em seu lugar.'
              }
            />
            <div className="max-h-96 overflow-auto border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{isEn ? 'Removed document' : 'Documento removido'}</TableHead>
                    <TableHead>{isEn ? 'Kept in its place' : 'Mantido no lugar'}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {shownDuplicates.slice(0, 200).map((doc, index) => (
                    <TableRow key={`${String(doc['TITLE'])}-${index}`}>
                      <TableCell className="max-w-96 truncate" title={String(doc['TITLE'])}>
                        {String(doc['TITLE'])}
                      </TableCell>
                      <TableCell
                        className="max-w-96 truncate"
                        title={doc['DOCUMENTO DE REFERÊNCIA (MANTIDO)']}
                      >
                        {doc['DOCUMENTO DE REFERÊNCIA (MANTIDO)']}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {shownDuplicates.length > 200 && (
              <p className="eyebrow">
                {isEn ? 'Showing the first 200 of' : 'Exibindo as 200 primeiras de'}{' '}
                {shownDuplicates.length.toLocaleString(nf)}.
              </p>
            )}
          </div>
      </Collapse>
    </CardContent>
  </Card>
  );
}
