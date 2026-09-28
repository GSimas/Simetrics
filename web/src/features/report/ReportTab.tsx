import { useEffect, useRef, useState, type ReactNode } from 'react';
import { CheckSquare, Download, FileText, FileType, Loader2, Square } from 'lucide-react';

import { SectionTitle } from '@/components/InfoTip';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { EmptyState } from '@/features/EmptyState';
import { getAnalyticsWorker } from '@/workers/client';
import { useIdleRender } from '@/lib/use-idle-render';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import { hybridMethodsText } from '@/core/hybrid/report';
import { hybridReportRows, hybridReportTitle } from '@/core/hybrid/report-rows';
import { cn } from '@/lib/utils';

import { captureReportCharts } from './capture';
import { reportNamedThemesChart, reportThemesChart } from './chart-renderer';
import {
  ADVANCED_CHARTS,
  DEFAULT_SELECTION,
  REPORT_GROUPS,
  REPORT_ITEMS,
  selectionOf,
  type ReportChartId,
  type ReportGroup,
  type ReportItemId,
  type ReportSelection,
} from './report-catalog';
import {
  BoxplotFigure,
  CollabChordFigure,
  ConceptFigure,
  GeneticsFigure,
  HistoriographFigure,
  LotkaFigure,
  NetworkChordFigure,
  NetworkFigure,
  ProductionFigure,
  RankingFigure,
  SafeFigure,
  SankeyFigure,
  ThematicMapFigure,
  ThemesFigure,
  WordCloudFigure,
  WorldMapFigure,
} from './ReportCharts';

const TOP_N_OPTIONS = [10, 15, 25, 50] as const;

type ExportFormat = 'pdf' | 'docx';

export default function ReportTab() {
  const active = useDataset((state) => state.active);
  const overview = useDataset((state) => state.overview);
  const tables = useDataset((state) => state.tables);
  const sna = useDataset((state) => state.sna);
  const network = useDataset((state) => state.network);
  const clustering = useDataset((state) => state.clustering);
  const hybridRun = useDataset((state) => state.hybridRun);
  const computeOverview = useDataset((state) => state.computeOverview);
  const computeTables = useDataset((state) => state.computeTables);
  const computeSna = useDataset((state) => state.computeSna);
  const computeNetwork = useDataset((state) => state.computeNetwork);
  const { locale, t } = useLocale();
  const isEn = locale === 'en';

  const [selection, setSelection] = useState<ReportSelection>(DEFAULT_SELECTION);
  const [topN, setTopN] = useState<number>(15);
  // `undefined`: ainda calculando; `null`: falhou.
  const [collaboration, setCollaboration] = useState<CollaborationNetwork | null | undefined>(undefined);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const paperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active) return;
    if (!overview) void computeOverview();
    if (!tables) void computeTables();
    if (!sna) void computeSna();
    if (!network) void computeNetwork('Palavras-chave', 40, 'Grau Absoluto');
  }, [active, overview, tables, sna, network, computeOverview, computeTables, computeSna, computeNetwork]);

  // Efeito próprio: antes ele dependia de overview/tables/sna/network e pedia a mesma
  // rede de colaboração ao worker a cada um que chegava (até cinco vezes por visita).
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    getAnalyticsWorker()
      .collaboration(active, 30)
      .then((res) => {
        if (!cancelled) setCollaboration(res);
      })
      .catch(() => {
        if (!cancelled) setCollaboration(null);
      });
    return () => {
      cancelled = true;
    };
  }, [active]);

  // O gráfico de temas não tem equivalente nas abas: segue desenhado em canvas.
  const themesChartImg = useIdleRender(
    !selection.themesChart
      ? null
      : hybridRun
        ? () => reportNamedThemesChart(hybridReportRows(hybridRun), active?.length ?? 0, locale)
        : !clustering || clustering.clusters.length === 0
          ? null
          : () => reportThemesChart(clustering.clusters, active?.length ?? 0, locale),
    [selection.themesChart, clustering, hybridRun, active, locale],
  );

  if (!active) {
    return <EmptyState title={isEn ? 'Scientific Report' : 'Relatório Científico'} />;
  }

  const toggle = (id: ReportItemId) => setSelection((prev) => ({ ...prev, [id]: !prev[id] }));

  // Um frame para o modal mostrar "Gerando…" antes do trabalho pesado ocupar a página.
  const nextPaint = (): Promise<void> =>
    new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

  const handleDownload = async (format: ExportFormat) => {
    const paper = paperRef.current;
    if (!paper) return;
    setExporting(format);
    setExportMessage(null);
    try {
      await nextPaint();
      const chartIds = REPORT_ITEMS.filter((item) => item.kind === 'chart' && selection[item.id]).map(
        (item) => item.id as ReportChartId,
      );
      const { images, missing } = await captureReportCharts(paper, chartIds);
      const data = { dataset: active, overview, tables, sna, clustering, hybridRun, selection, images, topN, locale };
      if (format === 'pdf') {
        const { generatePdfReport } = await import('./pdf-generator');
        generatePdfReport(data);
      } else {
        const { generateDocxReport } = await import('./docx-generator');
        await generateDocxReport(data);
      }
      if (missing.length > 0) {
        const names = missing.map((id) => REPORT_ITEMS.find((item) => item.id === id)?.label[locale] ?? id).join(', ');
        setExportMessage(
          isEn
            ? `Downloaded. These charts had not finished loading and were left out: ${names}.`
            : `Baixado. Estes gráficos ainda não tinham terminado de carregar e ficaram de fora: ${names}.`,
        );
      } else {
        setDownloadOpen(false);
      }
    } catch (error) {
      console.error(error);
      setExportMessage(isEn ? 'Could not generate the file.' : 'Não foi possível gerar o arquivo.');
    } finally {
      setExporting(null);
    }
  };

  const counts: Partial<Record<ReportItemId, string | undefined>> = {
    summary: `${active.length} docs`,
    authors: tables ? `${tables.authors.length}` : undefined,
    countries: tables ? `${tables.countries.length}` : undefined,
    venues: tables ? `${tables.venues.length}` : undefined,
    keywords: tables ? `${tables.keywords.length}` : undefined,
    themes: hybridRun ? `${hybridRun.finalTaxonomy.length}` : clustering ? `${clustering.clusters.length}` : undefined,
    networkTopology: sna ? `${sna.global.nodeCount}` : undefined,
  };
  const selectedCount = REPORT_ITEMS.filter((item) => selection[item.id]).length;
  const groups = Object.keys(REPORT_GROUPS) as ReportGroup[];
  const firstAdvanced = REPORT_ITEMS.find((item) => ADVANCED_CHARTS.includes(item.id as ReportChartId) && selection[item.id])?.id;

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
      {/* Seleção: coluna própria, fixa enquanto a prévia rola ao lado. */}
      <Card data-tour="report-builder" className="lg:sticky lg:top-36 lg:max-h-[calc(100vh-10rem)] lg:overflow-y-auto">
        <CardHeader className="space-y-4">
          <div className="space-y-1">
            <SectionTitle title={t('report_title')} info={t('report_info')} />
            <p className="eyebrow">
              {selectedCount} / {REPORT_ITEMS.length} {t('report_selected')}
            </p>
          </div>
          <Button onClick={() => { setExportMessage(null); setDownloadOpen(true); }} className="w-full cursor-pointer gap-2">
            <Download className="size-4" aria-hidden />
            {isEn ? 'Download' : 'Baixar'}
          </Button>
        </CardHeader>

        <CardContent className="space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-2 border-y border-border/70 py-2">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => setSelection(selectionOf(() => true))} className="h-8 cursor-pointer gap-1.5 px-2 text-xs font-semibold">
                <CheckSquare className="size-3.5 text-primary" aria-hidden />
                {isEn ? 'All' : 'Tudo'}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setSelection(selectionOf(() => false))} className="h-8 cursor-pointer gap-1.5 px-2 text-xs font-semibold text-muted-foreground">
                <Square className="size-3.5" aria-hidden />
                {isEn ? 'None' : 'Nenhum'}
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="top-n-select" className="text-xs font-semibold text-muted-foreground">
                {isEn ? 'Table rows' : 'Linhas por tabela'}
              </Label>
              <Select value={String(topN)} onValueChange={(val) => setTopN(Number(val))}>
                <SelectTrigger id="top-n-select" className="h-8 w-24 text-xs font-bold">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TOP_N_OPTIONS.map((opt) => (
                    <SelectItem key={opt} value={String(opt)}>
                      Top {opt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {groups.map((group) => (
            <fieldset key={group} className="space-y-1">
              <legend className="eyebrow mb-1.5 text-muted-foreground">{REPORT_GROUPS[group][locale]}</legend>
              {REPORT_ITEMS.filter((item) => item.group === group).map((item) => (
                <label
                  key={item.id}
                  className="flex cursor-pointer items-center gap-2.5 rounded-md px-1.5 py-1.5 text-sm transition-colors hover:bg-muted/60"
                >
                  <Checkbox checked={selection[item.id]} onCheckedChange={() => toggle(item.id)} />
                  <span className={cn('min-w-0 flex-1', !selection[item.id] && 'text-muted-foreground')}>{item.label[locale]}</span>
                  {counts[item.id] && (
                    <span className="shrink-0 tabular-nums text-[11px] text-muted-foreground">{counts[item.id]}</span>
                  )}
                </label>
              ))}
            </fieldset>
          ))}
        </CardContent>
      </Card>

      {/* Prévia ao vivo: a folha que vai para o PDF e o DOCX, sempre em papel claro. */}
      <div data-tour="report-preview" className="min-w-0 space-y-3">
        <SectionTitle
          title={isEn ? 'Report preview' : 'Pré-visualização do relatório'}
          info={
            isEn
              ? 'Updates live as you select sections and charts. The downloaded file follows this layout.'
              : 'Atualiza ao vivo conforme a seleção. O arquivo baixado segue esta diagramação.'
          }
        />

        <div ref={paperRef} className="report-paper space-y-8 border border-border bg-background p-6 text-foreground shadow-lg sm:p-10">
          <ReportHeader overview={overview} total={active.length} isEn={isEn} />

          {REPORT_ITEMS.filter((item) => selection[item.id]).map((item) => (
            <div key={item.id} className="space-y-8">
              {item.id === firstAdvanced && (
                <SectionHeading>{isEn ? '9. Advanced Visual Analyses' : '9. Análises Visuais Avançadas'}</SectionHeading>
              )}
              <SafeFigure>{renderBlock(item.id)}</SafeFigure>
            </div>
          ))}

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 text-xs text-muted-foreground">
            <p>
              {isEn
                ? 'Simetrics · Bibliometric Intelligence Platform · Developed by'
                : 'Simetrics · Plataforma de Inteligência Bibliométrica · Desenvolvido por'}{' '}
              <span className="font-bold text-foreground">Gustavo Simas</span>
            </p>
            <p className="text-[11px] italic">
              {isEn ? 'Document rendered client-side.' : 'Documento processado localmente no navegador.'}
            </p>
          </div>
        </div>
      </div>

      <Dialog open={downloadOpen} onOpenChange={(open) => !exporting && setDownloadOpen(open)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{isEn ? 'Download report' : 'Baixar relatório'}</DialogTitle>
            <DialogDescription>
              {isEn
                ? `${selectedCount} selected items, laid out as in the preview.`
                : `${selectedCount} itens selecionados, diagramados como na pré-visualização.`}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 sm:grid-cols-2">
            {(
              [
                ['pdf', FileText, 'PDF', isEn ? 'To read and print' : 'Para ler e imprimir'],
                ['docx', FileType, 'Word (.docx)', isEn ? 'To edit' : 'Para editar'],
              ] as const
            ).map(([format, Icon, title, hint]) => (
              <button
                key={format}
                type="button"
                disabled={exporting !== null}
                aria-busy={exporting === format}
                onClick={() => void handleDownload(format)}
                className="flex cursor-pointer flex-col items-start gap-2 border border-border p-4 text-left transition-colors hover:border-highlight focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-wait disabled:opacity-60"
              >
                {exporting === format ? (
                  <Loader2 className="size-6 animate-spin text-highlight" aria-hidden />
                ) : (
                  <Icon className="size-6 text-highlight" aria-hidden />
                )}
                <span className="text-sm font-semibold">{title}</span>
                <span className="text-xs text-muted-foreground">
                  {exporting === format ? (isEn ? 'Generating…' : 'Gerando…') : hint}
                </span>
              </button>
            ))}
          </div>
          {exportMessage && (
            <p role="alert" className="text-sm text-muted-foreground">
              {exportMessage}
            </p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );

  function renderBlock(id: ReportItemId): ReactNode {
    const nf = (value: number) => value.toLocaleString(isEn ? 'en-US' : 'pt-BR');
    switch (id) {
      case 'summary':
        return overview && (
          <div className="space-y-2 border border-border bg-card p-4">
            <h2 className="eyebrow text-highlight">{isEn ? 'Executive Summary & Dataset Scope' : 'Resumo Executivo & Escopo da Base'}</h2>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {isEn
                ? `This report compiles bibliometric metrics, collaboration graphs, and research themes from a corpus of ${nf(active!.length)} papers published between ${overview.summary.timespan || 'N/A'}. A total of ${nf(overview.summary.authorsCount)} authors and ${nf(overview.summary.countriesCount)} countries participated in the production.`
                : `Este relatório consolida indicadores cientométricos, redes de colaboração e tópicos de pesquisa a partir de uma base com ${nf(active!.length)} documentos indexados no período ${overview.summary.timespan || 'N/A'}. A produção envolveu ${nf(overview.summary.authorsCount)} autores e ${nf(overview.summary.countriesCount)} países.`}
            </p>
          </div>
        );
      case 'kpis': {
        if (!overview) return null;
        const totalCitations = active!.reduce((acc, d) => acc + (toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0), 0);
        return (
          <div className="space-y-3">
            <SectionHeading>{isEn ? '1. Core Scientometric Indicators' : '1. Indicadores Cientométricos Globais'}</SectionHeading>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {(
                [
                  [isEn ? 'Documents' : 'Documentos', nf(overview.summary.totalDocs)],
                  [isEn ? 'Authors' : 'Autores', nf(overview.summary.authorsCount)],
                  [isEn ? 'Citations' : 'Citações', nf(totalCitations)],
                  [isEn ? 'Annual growth' : 'Crescimento anual', `${overview.summary.bibliometrix.growthRate.toFixed(2)}%`],
                ] as const
              ).map(([label, value]) => (
                <div key={label} className="border border-border bg-card p-3">
                  <p className="eyebrow text-muted-foreground">{label}</p>
                  <p className="mt-1 text-xl font-bold tabular-nums">{value}</p>
                </div>
              ))}
            </div>
          </div>
        );
      }
      case 'authors':
        return tables && tables.authors.length > 0 && (
          <EntityTable
            title={isEn ? `2. Top ${topN} Authors by Production & Impact` : `2. Principais Autores (Top ${topN})`}
            head={[isEn ? 'Author' : 'Autor', 'Docs', isEn ? 'Citations' : 'Citações', 'h', 'g', 'i10', 'm']}
            rows={tables.authors.slice(0, topN).map((a) => [a.entity, a.docCount, a.citations, a.h, a.g, a.i10, a.m.toFixed(2)])}
          />
        );
      case 'countries':
        return tables && tables.countries.length > 0 && (
          <EntityTable
            title={isEn ? `3. Geographic Distribution (Top ${topN} Countries)` : `3. Distribuição Geográfica (Top ${topN})`}
            head={[isEn ? 'Country' : 'País', 'Docs', isEn ? 'Citations' : 'Citações', 'h', isEn ? 'Mean' : 'Média']}
            rows={tables.countries.slice(0, topN).map((c) => [c.entity, c.docCount, c.citations, c.h, c.meanCitations.toFixed(1)])}
          />
        );
      case 'venues':
        return tables && tables.venues.length > 0 && (
          <EntityTable
            title={isEn ? `4. Top Publishing Venues (Top ${topN})` : `4. Principais Veículos de Publicação (Top ${topN})`}
            head={['Venue', 'Docs', isEn ? 'Citations' : 'Citações', 'h', isEn ? 'Mean Cit.' : 'Média Cit.']}
            rows={tables.venues.slice(0, topN).map((v) => [v.entity, v.docCount, v.citations, v.h, v.meanCitations.toFixed(1)])}
          />
        );
      case 'keywords':
        return tables && tables.keywords.length > 0 && (
          <EntityTable
            title={isEn ? `5. Top Keywords & Lexicometrics (Top ${topN})` : `5. Palavras-Chave & Lexicometria (Top ${topN})`}
            head={[isEn ? 'Keyword' : 'Palavra-chave', 'Docs', isEn ? 'Citations' : 'Citações', 'h']}
            rows={tables.keywords.slice(0, topN).map((k) => [k.entity, k.docCount, k.citations, k.h])}
          />
        );
      case 'themes':
        // Agrupamento e classificação híbrida, quando os dois existem — como nos geradores.
        return (
          <>
            {clustering && clustering.clusters.length > 0 && (
              <div className="space-y-3">
                <SectionHeading>
                  {isEn
                    ? `6. AI Thematic Clusters (Silhouette: ${clustering.silhouette.toFixed(3)})`
                    : `6. Agrupamento Temático por IA (Silhouette: ${clustering.silhouette.toFixed(3)})`}
                </SectionHeading>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {clustering.clusters.map((c) => (
                    <ThemeCard key={c.clusterId} name={`${isEn ? 'Theme' : 'Tema'} ${c.clusterId + 1}`} share={(c.size / active!.length) * 100}>
                      <strong>{c.size}</strong> {isEn ? 'documents' : 'artigos'} · {c.topTerms.slice(0, 5).join(', ')}
                    </ThemeCard>
                  ))}
                </div>
              </div>
            )}
            {hybridRun && (
            <div className="space-y-3">
              <SectionHeading>6. {hybridReportTitle(hybridRun, locale)}</SectionHeading>
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {hybridReportRows(hybridRun).map((row) => (
                  <ThemeCard key={row.name} name={row.name} share={row.share}>
                    <strong>{row.documents}</strong> {isEn ? 'documents' : 'artigos'} · {isEn ? 'mean confidence' : 'confiança média'}{' '}
                    {row.meanConfidence.toFixed(2)}
                  </ThemeCard>
                ))}
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">{hybridMethodsText(hybridRun, locale)}</p>
            </div>
            )}
          </>
        );
      case 'networkTopology':
        return sna && (
          <div className="space-y-3">
            <SectionHeading>{isEn ? '7. Deep Knowledge Ecology & Network Topology' : '7. Topologia da Rede & Ecologia Profunda'}</SectionHeading>
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
              {(
                [
                  [isEn ? 'Density' : 'Densidade', sna.global.density.toFixed(4)],
                  [isEn ? 'Clustering' : 'Clustering médio', sna.global.clustering.toFixed(4)],
                  [isEn ? 'Shannon entropy' : 'Entropia de Shannon', sna.global.entropy.toFixed(3)],
                  [
                    isEn ? 'Global efficiency' : 'Eficiência global',
                    typeof sna.global.efficiency === 'number'
                      ? sna.global.efficiency.toFixed(4)
                      : isEn
                        ? String(sna.global.efficiency).replace('Grafo Denso', 'dense graph')
                        : String(sna.global.efficiency),
                  ],
                  [isEn ? 'Mean degree' : 'Grau médio', sna.global.meanDegree.toFixed(2)],
                  [isEn ? 'Power law exponent' : 'Lei de potência', sna.global.powerLawExponent.toFixed(2)],
                ] as const
              ).map(([label, value]) => (
                <div key={label} className="border border-border bg-card p-2.5">
                  <p className="text-[11px] font-semibold text-muted-foreground">{label}</p>
                  <p className="text-sm font-bold tabular-nums">{value}</p>
                </div>
              ))}
            </div>
          </div>
        );
      case 'topDocuments': {
        const columns = collectColumns(active!);
        const titleCol = pickColumn(columns, FIELD_CANDIDATES.title);
        const authCol = pickColumn(columns, FIELD_CANDIDATES.authors);
        const docs = [...active!]
          .sort((a, b) => (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0))
          .slice(0, topN);
        return (
          <EntityTable
            title={isEn ? `8. Highly Cited Seminal Documents (Top ${topN})` : `8. Documentos Mais Citados da Base (Top ${topN})`}
            head={[isEn ? 'Title' : 'Título', isEn ? 'Authors' : 'Autores', isEn ? 'Year' : 'Ano', isEn ? 'Citations' : 'Citações']}
            rows={docs.map((d) => [
              titleCol ? String(d[titleCol] ?? '') : '—',
              authCol ? String(d[authCol] ?? '') : '—',
              toNumeric(d[FIELD.YEAR_CLEAN]) ?? '—',
              toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0,
            ])}
            truncate
          />
        );
      }
      case 'production':
        return <ProductionFigure docsPerYear={overview?.docsPerYear} />;
      case 'rankingAuthors':
        return <RankingFigure id={id} rows={tables?.authors} dataset={active!} />;
      case 'rankingCountries':
        return <RankingFigure id={id} rows={tables?.countries} dataset={active!} />;
      case 'rankingVenues':
        return <RankingFigure id={id} rows={tables?.venues} dataset={active!} />;
      case 'rankingDocuments':
        return <RankingFigure id={id} rows={undefined} dataset={active!} />;
      case 'worldMap':
        return <WorldMapFigure collaboration={collaboration} />;
      case 'collabChord':
        return <CollabChordFigure collaboration={collaboration} />;
      case 'wordCloud':
        return <WordCloudFigure keywords={tables?.keywords} />;
      case 'themesChart':
        return <ThemesFigure image={themesChartImg} alt={isEn ? 'AI thematic distribution' : 'Distribuição temática por IA'} />;
      case 'network':
        return <NetworkFigure network={network} />;
      case 'networkChord':
        return <NetworkChordFigure network={network} />;
      case 'sankey':
        return <SankeyFigure dataset={active!} />;
      case 'boxplot':
        return <BoxplotFigure dataset={active!} />;
      case 'genetics':
        return <GeneticsFigure dataset={active!} />;
      case 'concept2d':
        return <ConceptFigure dataset={active!} dimensions="2d" />;
      case 'concept3d':
        return <ConceptFigure dataset={active!} dimensions="3d" />;
      case 'thematicMap':
        return <ThematicMapFigure dataset={active!} />;
      case 'historiograph':
        return <HistoriographFigure dataset={active!} />;
      case 'lotka':
        return <LotkaFigure lotka={overview ? overview.lotka : undefined} />;
    }
  }
}

function ReportHeader({ overview, total, isEn }: { overview: { summary: { timespan: string } } | null; total: number; isEn: boolean }) {
  return (
    <div className="border-b border-border pb-6">
      <p className="eyebrow text-highlight">{isEn ? 'Simetrics · Scientometric Report' : 'Simetrics · Relatório Cientométrico'}</p>
      <h1 className="mt-3 text-2xl font-bold tracking-tight">
        {isEn ? 'Scientometric Intelligence ' : 'Relatório Cientométrico & '}
        <em className="accent-serif text-highlight">{isEn ? 'Report' : 'Bibliométrico'}</em>
      </h1>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
        <span>
          <strong>{isEn ? 'Corpus scope' : 'Escopo'}:</strong> {total.toLocaleString(isEn ? 'en-US' : 'pt-BR')}{' '}
          {isEn ? 'documents' : 'documentos'} · {overview?.summary.timespan || 'N/A'}
        </span>
        <span>
          <strong>{isEn ? 'Generated on' : 'Emissão'}:</strong>{' '}
          {new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', { day: '2-digit', month: 'long', year: 'numeric' })}
        </span>
      </div>
    </div>
  );
}

function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="border-b border-border pb-1.5 text-sm font-bold">{children}</h2>;
}

function EntityTable({
  title,
  head,
  rows,
  truncate = false,
}: {
  title: string;
  head: readonly string[];
  rows: readonly (readonly (string | number)[])[];
  truncate?: boolean;
}) {
  return (
    <div className="space-y-3">
      <SectionHeading>{title}</SectionHeading>
      <div className="overflow-hidden border border-border">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/60 text-[11px]">
              <TableHead className="w-10">#</TableHead>
              {head.map((label, index) => (
                <TableHead key={label} className={index === 0 ? undefined : 'text-right'}>
                  {label}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody className="text-xs">
            {rows.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                <TableCell className="font-semibold text-muted-foreground">{rowIndex + 1}</TableCell>
                {row.map((cell, index) => (
                  <TableCell
                    key={index}
                    className={cn(
                      index === 0 ? 'font-semibold' : 'text-right tabular-nums',
                      truncate && index === 0 && 'max-w-72 truncate',
                      truncate && index === 1 && 'max-w-40 truncate text-left text-muted-foreground',
                    )}
                  >
                    {cell}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function ThemeCard({ name, share, children }: { name: string; share: number; children: ReactNode }) {
  return (
    <div className="space-y-1.5 border border-border bg-card p-3.5">
      <div className="flex items-center justify-between gap-2">
        <p className="truncate text-xs font-bold">{name}</p>
        <span className="eyebrow text-highlight">{share.toFixed(1)}%</span>
      </div>
      <p className="text-[11px] text-muted-foreground">{children}</p>
    </div>
  );
}
