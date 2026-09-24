import { useEffect, useState } from 'react';
import {
  BarChart3,
  BookOpen,
  CheckSquare,
  Download,
  FileCheck,
  FileSpreadsheet,
  FileText,
  Globe2,
  Image as ImageIcon,
  Layers,
  Network,
  PieChart,
  Quote,
  Sparkles,
  Square,
  Users,
} from 'lucide-react';

import { SectionTitle } from '@/components/InfoTip';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
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
// Os geradores (jsPDF, docx) pesam ~1,2 MB: entram só no clique de exportar.
import type { ReportSectionsSelection } from './pdf-generator';
import { ReportChartImage } from './ReportChartImage';
import {
  REPORT_CHART_SIZE,
  reportAuthorsChart,
  reportCountriesChart,
  reportNetworkChart,
  reportProductionChart,
  reportThemesChart,
  reportWordCloudChart,
  reportWorldMapChart,
} from './chart-renderer';

const DEFAULT_SELECTION: ReportSectionsSelection = {
  summary: true,
  kpis: true,
  chartProduction: true,
  authors: true,
  chartAuthors: true,
  countries: true,
  chartCountries: true,
  chartWorldMap: true,
  venues: true,
  keywords: true,
  chartKeywords: true,
  themes: true,
  chartThemes: true,
  networkTopology: true,
  chartNetwork: true,
  topDocuments: true,
};

const TOP_N_OPTIONS = [10, 15, 25, 50] as const;

export default function ReportTab() {
  const active = useDataset((state) => state.active);
  const overview = useDataset((state) => state.overview);
  const tables = useDataset((state) => state.tables);
  const sna = useDataset((state) => state.sna);
  const network = useDataset((state) => state.network);
  const clustering = useDataset((state) => state.clustering);
  const computeOverview = useDataset((state) => state.computeOverview);
  const computeTables = useDataset((state) => state.computeTables);
  const computeSna = useDataset((state) => state.computeSna);
  const computeNetwork = useDataset((state) => state.computeNetwork);
  const { locale, t } = useLocale();
  const isEn = locale === 'en';

  const [selection, setSelection] = useState<ReportSectionsSelection>(DEFAULT_SELECTION);
  const [topN, setTopN] = useState<number>(15);
  const [collaboration, setCollaboration] = useState<CollaborationNetwork | null>(null);
  const [isExportingPdf, setIsExportingPdf] = useState(false);
  const [isExportingDocx, setIsExportingDocx] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

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

  // Gera as imagens dos gráficos sob demanda — as mesmas que a exportação reaproveita.
  const productionChartImg = useIdleRender(
    !overview || overview.docsPerYear.length === 0 ? null : () => reportProductionChart(overview.docsPerYear, locale),
    [overview, locale],
  );

  const authorsChartImg = useIdleRender(
    !tables || tables.authors.length === 0 ? null : () => reportAuthorsChart(tables.authors, locale),
    [tables, locale],
  );

  const countriesChartImg = useIdleRender(
    !tables || tables.countries.length === 0 ? null : () => reportCountriesChart(tables.countries, locale),
    [tables, locale],
  );

  const worldMapChartImg = useIdleRender(
    !collaboration || collaboration.nodes.length === 0 ? null : () => reportWorldMapChart(collaboration, locale),
    [collaboration, locale],
  );

  const networkChartImg = useIdleRender(
    !network || network.nodes.length === 0 ? null : () => reportNetworkChart(network.nodes, network.edges, locale),
    [network, locale],
  );

  const themesChartImg = useIdleRender(
    !clustering || clustering.clusters.length === 0
      ? null
      : () => reportThemesChart(clustering.clusters, active?.length ?? 0, locale),
    [clustering, active, locale],
  );

  const wordCloudChartImg = useIdleRender(
    !tables || tables.keywords.length === 0 ? null : () => reportWordCloudChart(tables.keywords, locale),
    [tables, locale],
  );

  if (!active) {
    return <EmptyState title={isEn ? 'Scientific Report' : 'Relatório Científico'} />;
  }

  const toggleSection = (key: keyof ReportSectionsSelection) => {
    setSelection((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const selectAll = () => {
    setSelection({
      summary: true,
      kpis: true,
      chartProduction: true,
      authors: true,
      chartAuthors: true,
      countries: true,
      chartCountries: true,
      chartWorldMap: true,
      venues: true,
      keywords: true,
      chartKeywords: true,
      themes: true,
      chartThemes: true,
      networkTopology: true,
      chartNetwork: true,
      topDocuments: true,
    });
  };

  const deselectAll = () => {
    setSelection({
      summary: false,
      kpis: false,
      chartProduction: false,
      authors: false,
      chartAuthors: false,
      countries: false,
      chartCountries: false,
      chartWorldMap: false,
      venues: false,
      keywords: false,
      chartKeywords: false,
      themes: false,
      chartThemes: false,
      networkTopology: false,
      chartNetwork: false,
      topDocuments: false,
    });
  };

  // Um frame para o botão mostrar "Gerando…" antes do trabalho pesado ocupar a página.
  const nextPaint = (): Promise<void> =>
    new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

  const handleExportPdf = async () => {
    setIsExportingPdf(true);
    setExportError(null);
    try {
      await nextPaint();
      const { generatePdfReport } = await import('./pdf-generator');
      generatePdfReport({
        dataset: active,
        overview,
        tables,
        sna,
        network,
        collaboration,
        clustering,
        selection,
        topN,
        locale,
      });
    } catch (error) {
      console.error(error);
      setExportError(isEn ? 'Could not generate the PDF.' : 'Não foi possível gerar o PDF.');
    } finally {
      setIsExportingPdf(false);
    }
  };

  const handleExportDocx = async () => {
    setIsExportingDocx(true);
    setExportError(null);
    try {
      await nextPaint();
      const { generateDocxReport } = await import('./docx-generator');
      await generateDocxReport({
        dataset: active,
        overview,
        tables,
        sna,
        network,
        collaboration,
        clustering,
        selection,
        topN,
        locale,
      });
    } catch (error) {
      console.error(error);
      setExportError(isEn ? 'Could not generate the DOCX file.' : 'Não foi possível gerar o DOCX.');
    } finally {
      setIsExportingDocx(false);
    }
  };

  const totalCitations = active.reduce((acc, d) => acc + (toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0), 0);

  const sectionsList: {
    key: keyof ReportSectionsSelection;
    label: string;
    labelEn: string;
    desc: string;
    descEn: string;
    icon: typeof FileText;
    isChart?: boolean;
    count?: string | undefined;
  }[] = [
    {
      key: 'summary',
      label: 'Resumo Executivo & Escopo',
      labelEn: 'Executive Summary & Scope',
      desc: 'Panorama sintético, período temporal e contagens gerais da base.',
      descEn: 'High-level synthesis, timespan, and global dataset volume.',
      icon: FileText,
      count: `${active.length} docs`,
    },
    {
      key: 'kpis',
      label: 'Indicadores Cientométricos Globais',
      labelEn: 'Core Scientometric KPIs',
      desc: 'Documentos, Citações, Taxa de Crescimento Anual e Colaboração Internacional.',
      descEn: 'Articles, Citations, Annual Growth Rate, and International Collaboration Rate.',
      icon: Quote,
      count: overview ? `${overview.summary.totalDocs} docs` : undefined,
    },
    {
      key: 'chartProduction',
      label: '📈 Gráfico: Produção Anual (Linha do Tempo)',
      labelEn: '📈 Chart: Annual Production Timeline',
      desc: 'Visualização da evolução histórica da publicação de artigos por ano.',
      descEn: 'Historical evolution chart of published papers per year.',
      icon: BarChart3,
      isChart: true,
    },
    {
      key: 'authors',
      label: 'Ranking de Autores & Produtividade',
      labelEn: 'Authors Ranking & Impact',
      desc: 'Tabela de autores com contagem de artigos, citações, índices h, g, i10 e m.',
      descEn: 'Author metrics table including papers, citations, and h/g/i10/m indices.',
      icon: Users,
      count: tables ? `${tables.authors.length} autores` : undefined,
    },
    {
      key: 'chartAuthors',
      label: '📊 Gráfico: Top 10 Autores Mais Produtivos',
      labelEn: '📊 Chart: Top 10 Most Prolific Authors',
      desc: 'Gráfico horizontal comparativo do volume de artigos e impacto dos autores.',
      descEn: 'Horizontal bar chart comparing author publication volume and impact.',
      icon: BarChart3,
      isChart: true,
    },
    {
      key: 'countries',
      label: 'Geografia & Colaboração Internacional',
      labelEn: 'Geographic Distribution',
      desc: 'Produção por países e documentos mais citados de cada nação.',
      descEn: 'Country-level scientific output and most cited articles.',
      icon: Globe2,
      count: tables ? `${tables.countries.length} países` : undefined,
    },
    {
      key: 'chartCountries',
      label: '🌍 Gráfico: Top 10 Países com Maior Produção',
      labelEn: '🌍 Chart: Top 10 Leading Countries',
      desc: 'Gráfico de barras da distribuição geográfica da pesquisa.',
      descEn: 'Bar chart of geographic distribution across nations.',
      icon: Globe2,
      isChart: true,
    },
    {
      key: 'chartWorldMap',
      label: '🌐 Gráfico: Mapa-Múndi de Colaboração Global',
      labelEn: '🌐 Chart: World Collaboration Map',
      desc: 'Mapa-múndi cartográfico com conexões e arcos de coautoria entre países.',
      descEn: 'World map showing cross-border co-authorship arcs and output hubs.',
      icon: Globe2,
      isChart: true,
    },
    {
      key: 'venues',
      label: 'Veículos de Publicação (Periódicos/Venues)',
      labelEn: 'Publishing Venues / Journals',
      desc: 'Principais periódicos, anais e veículos que publicam sobre o tema.',
      descEn: 'Top journals, conferences, and publishing outlets.',
      icon: BookOpen,
      count: tables ? `${tables.venues.length} venues` : undefined,
    },
    {
      key: 'keywords',
      label: 'Palavras-Chave & Lexicometria',
      labelEn: 'Keywords & Lexicometrics',
      desc: 'Frequência de palavras-chave, citações agregadas e densidade vocabular.',
      descEn: 'Keyword frequency, aggregate citations, and vocabulary density.',
      icon: FileSpreadsheet,
      count: tables ? `${tables.keywords.length} termos` : undefined,
    },
    {
      key: 'chartKeywords',
      label: '☁️ Gráfico: Nuvem de Palavras-Chave',
      labelEn: '☁️ Chart: Lexicometric Word Cloud',
      desc: 'Diagrama visual de nuvem com termos e densidades mais expressivas.',
      descEn: 'Visual keyword cloud showing prominent scientific concepts.',
      icon: ImageIcon,
      isChart: true,
    },
    {
      key: 'themes',
      label: 'Estrutura Temática por IA (Clusters)',
      labelEn: 'AI Thematic Clusters',
      desc: 'Clusters semânticos descobertos, score de silhueta e termos característicos.',
      descEn: 'Semantic research themes, silhouette score, and representative terms.',
      icon: Sparkles,
      count: clustering ? `${clustering.clusters.length} temas` : undefined,
    },
    {
      key: 'chartThemes',
      label: '🎯 Gráfico: Distribuição de Temas por IA',
      labelEn: '🎯 Chart: AI Thematic Distribution (Donut)',
      desc: 'Gráfico de pizza/donut demonstrando a proporção de cada vertente de pesquisa.',
      descEn: 'Donut chart illustrating the relative share of each research theme.',
      icon: PieChart,
      isChart: true,
    },
    {
      key: 'networkTopology',
      label: 'Topologia da Rede (Ecologia Profunda)',
      labelEn: 'Deep Knowledge Ecology Topology',
      desc: '11 métricas globais de rede (Densidade, Clustering, Entropia, Eficiência, PageRank, etc.).',
      descEn: '11 global SNA metrics (Density, Clustering, Shannon Entropy, Efficiency, PageRank).',
      icon: Network,
      count: sna ? `${sna.global.nodeCount} nós` : undefined,
    },
    {
      key: 'chartNetwork',
      label: '🕸️ Gráfico: Rede de Coocorrência (Louvain)',
      labelEn: '🕸️ Chart: Co-occurrence Network (Louvain)',
      desc: 'Grafo de conexões conceituais, nós centrais e agrupamento por comunidades.',
      descEn: 'Network graph of conceptual co-occurrences and Louvain community hubs.',
      icon: Network,
      isChart: true,
    },
    {
      key: 'topDocuments',
      label: 'Documentos Fundamentais (Mais Citados)',
      labelEn: 'Highly Cited Seminal Documents',
      desc: 'Tabela dos artigos mais influentes com autores, ano, citações e periódico.',
      descEn: 'Most influential publications with authors, year, citations, and journal.',
      icon: Layers,
      count: `${active.length} total`,
    },
  ];

  const selectedCount = sectionsList.filter(({ key }) => selection[key]).length;

  const columns = collectColumns(active);
  const titleCol = pickColumn(columns, FIELD_CANDIDATES.title);
  const authCol = pickColumn(columns, FIELD_CANDIDATES.authors);

  const sortedTopDocs = [...active]
    .sort((a, b) => (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0))
    .slice(0, topN);

  return (
    <div className="space-y-6">
      {/* 1. Painel de Controle de Exportação */}
      <Card data-tour="report-builder">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1">
              <SectionTitle title={t('report_title')} info={t('report_info')} />
              <p className="eyebrow">
                {selectedCount} / {sectionsList.length} {t('report_selected')} · Top {topN}
              </p>
            </div>

            {/* Botões de Ação de Download */}
            <div className="flex flex-wrap items-center gap-2.5">
              <Button
                variant="default"
                onClick={() => void handleExportPdf()}
                disabled={isExportingPdf}
                aria-busy={isExportingPdf}
                className="gap-2 cursor-pointer"
              >
                <Download className="size-4" />
                {isExportingPdf ? (isEn ? 'Building PDF...' : 'Gerando PDF...') : isEn ? 'Export PDF' : 'Baixar PDF'}
              </Button>

              <Button
                variant="outline"
                onClick={() => void handleExportDocx()}
                disabled={isExportingDocx}
                aria-busy={isExportingDocx}
                className="gap-2 cursor-pointer"
              >
                <FileCheck className="size-4" />
                {isExportingDocx ? (isEn ? 'Building DOCX...' : 'Gerando DOCX...') : isEn ? 'Export DOCX (Word)' : 'Baixar DOCX (Word)'}
              </Button>
              {exportError && (
                <p role="alert" className="basis-full text-sm text-destructive">
                  {exportError}
                </p>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-5">
          {/* Barra de Seleção Rápida e Opções */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-y border-border/70 py-3">
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={selectAll} className="h-8 gap-1.5 text-xs font-semibold cursor-pointer">
                <CheckSquare className="size-3.5 text-primary" />
                {isEn ? 'Select All' : 'Selecionar Tudo'}
              </Button>
              <Button variant="ghost" size="sm" onClick={deselectAll} className="h-8 gap-1.5 text-xs font-semibold text-muted-foreground cursor-pointer">
                <Square className="size-3.5" />
                {isEn ? 'Deselect All' : 'Desmarcar Tudo'}
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Label htmlFor="top-n-select" className="text-xs font-semibold text-muted-foreground">
                {isEn ? 'Items per table:' : 'Itens por tabela:'}
              </Label>
              <Select value={String(topN)} onValueChange={(val) => setTopN(Number(val))}>
                <SelectTrigger id="top-n-select" className="h-8 w-28 text-xs font-bold">
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

          {/* Grid de Seções e Gráficos com Checkboxes Interativos */}
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
            {sectionsList.map(({ key, label, labelEn, desc, descEn, count, isChart }) => {
              const isChecked = selection[key];
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => toggleSection(key)}
                  className={`flex items-start gap-3 rounded-xl border p-3 text-left transition-all duration-150 cursor-pointer ${
                    isChecked
                      ? isChart
                        ? 'border-indigo-500/80 bg-indigo-500/[0.08] shadow-2xs'
                        : 'border-blue-500/80 bg-blue-500/[0.06] shadow-2xs'
                      : 'border-border/60 bg-card opacity-65 hover:opacity-100 hover:border-border'
                  }`}
                >
                  <div
                    className={`mt-0.5 grid size-5 shrink-0 place-items-center rounded border transition-colors ${
                      isChecked
                        ? isChart
                          ? 'border-indigo-600 bg-indigo-600 text-white'
                          : 'border-blue-600 bg-blue-600 text-white'
                        : 'border-muted-foreground/40 bg-background'
                    }`}
                  >
                    {isChecked && <CheckSquare className="size-3.5" />}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <p className="text-xs font-bold text-foreground truncate">
                        {isEn ? labelEn : label}
                      </p>
                      {count && (
                        <span className="rounded bg-muted px-1.5 py-0.2 text-[10px] font-semibold text-muted-foreground shrink-0">
                          {count}
                        </span>
                      )}
                      {isChart && (
                        <Badge variant="purple" className="text-[9px] px-1 py-0 shrink-0">
                          {isEn ? 'Chart' : 'Gráfico'}
                        </Badge>
                      )}
                    </div>
                    <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground line-clamp-2">
                      {isEn ? descEn : desc}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* 2. Pré-Visualização Ao Vivo do Documento (A4 Executive Styling com Gráficos) */}
      <div data-tour="report-preview" className="space-y-3">
        <SectionTitle
          title={isEn ? 'Report preview' : 'Pré-visualização do relatório'}
          info={
            isEn
              ? 'Laid out according to the sections and charts selected above.'
              : 'Diagramado conforme as seções e gráficos selecionados acima.'
          }
        />

        <div className="mx-auto max-w-4xl rounded-2xl border border-border/90 bg-card p-6 sm:p-10 shadow-lg space-y-8 text-foreground transition-all">
          {/* Header do Relatório */}
          <div className="border-b border-border/80 pb-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="grid size-11 place-items-center border border-border text-highlight">
                  <FileText className="size-6" />
                </div>
                <div>
                  <h1 className="text-2xl font-black tracking-tight text-foreground">
                    SIMETRICS
                  </h1>
                  <p className="text-xs font-semibold text-blue-600 dark:text-blue-400">
                    {isEn ? 'Scientometric & Bibliometric Intelligence Report' : 'Relatório Cientométrico & Bibliométrico'}
                  </p>
                </div>
              </div>
              <Badge variant="blue" className="text-xs font-semibold">
                {isEn ? 'Official Synthesis' : 'Síntese Oficial'}
              </Badge>
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground border-t border-border/40 pt-3">
              <span>
                <strong>{isEn ? 'Corpus Scope' : 'Escopo'}:</strong> {active.length.toLocaleString(isEn ? 'en-US' : 'pt-BR')}{' '}
                {isEn ? 'documents' : 'artigos'} · {overview?.summary.timespan || 'N/A'}
              </span>
              <span>
                <strong>{isEn ? 'Generated on' : 'Emissão'}:</strong>{' '}
                {new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', {
                  day: '2-digit',
                  month: 'long',
                  year: 'numeric',
                })}
              </span>
            </div>
          </div>

          {/* 1. Resumo Executivo */}
          {selection.summary && overview && (
            <div className="rounded-xl border border-blue-200 bg-blue-500/[0.04] p-4.5 dark:border-blue-900/60 space-y-2">
              <h2 className="text-sm font-bold text-blue-700 dark:text-blue-400">
                {isEn ? 'Executive Summary & Dataset Scope' : 'Resumo Executivo & Escopo da Base'}
              </h2>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {isEn
                  ? `This report compiles bibliometric metrics, collaboration graphs, and research themes from a corpus of ${active.length.toLocaleString('en-US')} papers published between ${overview.summary.timespan || 'N/A'}. A total of ${overview.summary.authorsCount.toLocaleString('en-US')} authors and ${overview.summary.countriesCount.toLocaleString('en-US')} countries participated in the production.`
                  : `Este relatório consolida indicadores cientométricos, redes de colaboração e tópicos de pesquisa a partir de uma base com ${active.length.toLocaleString('pt-BR')} documentos indexados no período ${overview.summary.timespan || 'N/A'}. A produção envolveu ${overview.summary.authorsCount.toLocaleString('pt-BR')} autores e ${overview.summary.countriesCount.toLocaleString('pt-BR')} países.`}
              </p>
            </div>
          )}

          {/* 2. Indicadores Cientométricos Globais */}
          {selection.kpis && overview && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? '1. Core Scientometric Indicators' : '1. Indicadores Cientométricos Globais'}
              </h2>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg border border-border/70 p-3 bg-muted/20">
                  <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                    {isEn ? 'Total Documents' : 'Documentos'}
                  </p>
                  <p className="text-xl font-bold tabular-nums text-foreground mt-0.5">
                    {overview.summary.totalDocs.toLocaleString(isEn ? 'en-US' : 'pt-BR')}
                  </p>
                </div>
                <div className="rounded-lg border border-border/70 p-3 bg-muted/20">
                  <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                    {isEn ? 'Total Authors' : 'Autores'}
                  </p>
                  <p className="text-xl font-bold tabular-nums text-foreground mt-0.5">
                    {overview.summary.authorsCount.toLocaleString(isEn ? 'en-US' : 'pt-BR')}
                  </p>
                </div>
                <div className="rounded-lg border border-border/70 p-3 bg-muted/20">
                  <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                    {isEn ? 'Total Citations' : 'Citações'}
                  </p>
                  <p className="text-xl font-bold tabular-nums text-foreground mt-0.5">
                    {totalCitations.toLocaleString(isEn ? 'en-US' : 'pt-BR')}
                  </p>
                </div>
                <div className="rounded-lg border border-border/70 p-3 bg-muted/20">
                  <p className="text-[10px] font-semibold uppercase text-muted-foreground">
                    {isEn ? 'Annual Growth' : 'Crescimento Anual'}
                  </p>
                  <p className="text-xl font-bold tabular-nums text-foreground mt-0.5">
                    {overview.summary.bibliometrix.growthRate.toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Gráfico 1: Produção Anual */}
          {selection.chartProduction && (
            <ReportChartImage image={productionChartImg} {...REPORT_CHART_SIZE.production} alt="Gráfico de Evolução da Produção Científica" />
          )}

          {/* 3. Top Autores */}
          {selection.authors && tables && tables.authors.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? `2. Top ${topN} Authors by Production & Impact` : `2. Principais Autores (Top ${topN})`}
              </h2>
              <div className="rounded-xl border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px]">
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>{isEn ? 'Author' : 'Autor'}</TableHead>
                      <TableHead className="text-right">Docs</TableHead>
                      <TableHead className="text-right">{isEn ? 'Citations' : 'Citações'}</TableHead>
                      <TableHead className="text-right">h</TableHead>
                      <TableHead className="text-right">g</TableHead>
                      <TableHead className="text-right">i10</TableHead>
                      <TableHead className="text-right">m</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody className="text-xs">
                    {tables.authors.slice(0, topN).map((a, idx) => (
                      <TableRow key={a.entity}>
                        <TableCell className="font-semibold text-muted-foreground">{idx + 1}</TableCell>
                        <TableCell className="font-bold text-foreground">{a.entity}</TableCell>
                        <TableCell className="text-right tabular-nums">{a.docCount}</TableCell>
                        <TableCell className="text-right tabular-nums font-semibold">{a.citations}</TableCell>
                        <TableCell className="text-right tabular-nums">{a.h}</TableCell>
                        <TableCell className="text-right tabular-nums">{a.g}</TableCell>
                        <TableCell className="text-right tabular-nums">{a.i10}</TableCell>
                        <TableCell className="text-right tabular-nums">{a.m.toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* Gráfico 2: Top Autores */}
          {selection.chartAuthors && (
            <ReportChartImage image={authorsChartImg} {...REPORT_CHART_SIZE.bars} alt="Gráfico dos Top Autores" />
          )}

          {/* 4. Top Países */}
          {selection.countries && tables && tables.countries.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? `3. Geographic Distribution (Top ${topN} Countries)` : `3. Distribuição Geográfica (Top ${topN})`}
              </h2>
              <div className="rounded-xl border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px]">
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>{isEn ? 'Country' : 'País'}</TableHead>
                      <TableHead className="text-right">Docs</TableHead>
                      <TableHead className="text-right">{isEn ? 'Citations' : 'Citações'}</TableHead>
                      <TableHead className="text-right">h</TableHead>
                      <TableHead className="text-right">{isEn ? 'Mean' : 'Média'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody className="text-xs">
                    {tables.countries.slice(0, topN).map((c, idx) => (
                      <TableRow key={c.entity}>
                        <TableCell className="font-semibold text-muted-foreground">{idx + 1}</TableCell>
                        <TableCell className="font-bold text-foreground">{c.entity}</TableCell>
                        <TableCell className="text-right tabular-nums">{c.docCount}</TableCell>
                        <TableCell className="text-right tabular-nums font-semibold">{c.citations}</TableCell>
                        <TableCell className="text-right tabular-nums">{c.h}</TableCell>
                        <TableCell className="text-right tabular-nums">{c.meanCitations.toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* Gráfico 3: Top Países */}
          {selection.chartCountries && (
            <ReportChartImage image={countriesChartImg} {...REPORT_CHART_SIZE.bars} alt="Gráfico dos Top Países" />
          )}

          {/* Gráfico 4: Mapa-Múndi de Colaboração Internacional */}
          {selection.chartWorldMap && (
            <ReportChartImage image={worldMapChartImg} {...REPORT_CHART_SIZE.worldMap} alt="Mapa Global de Colaboração Internacional" />
          )}

          {/* 5. Top Venues */}
          {selection.venues && tables && tables.venues.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? `4. Top Publishing Venues (Top ${topN})` : `4. Principais Veículos de Publicação (Top ${topN})`}
              </h2>
              <div className="rounded-xl border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px]">
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>Venue / Journal</TableHead>
                      <TableHead className="text-right">Docs</TableHead>
                      <TableHead className="text-right">{isEn ? 'Citations' : 'Citações'}</TableHead>
                      <TableHead className="text-right">h</TableHead>
                      <TableHead className="text-right">{isEn ? 'Mean Cit.' : 'Média Cit.'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody className="text-xs">
                    {tables.venues.slice(0, topN).map((v, idx) => (
                      <TableRow key={v.entity}>
                        <TableCell className="font-semibold text-muted-foreground">{idx + 1}</TableCell>
                        <TableCell className="font-bold text-foreground">{v.entity}</TableCell>
                        <TableCell className="text-right tabular-nums">{v.docCount}</TableCell>
                        <TableCell className="text-right tabular-nums font-semibold">{v.citations}</TableCell>
                        <TableCell className="text-right tabular-nums">{v.h}</TableCell>
                        <TableCell className="text-right tabular-nums">{v.meanCitations.toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* 6. Palavras-Chave */}
          {selection.keywords && tables && tables.keywords.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? `5. Top Keywords & Lexicometrics (Top ${topN})` : `5. Palavras-Chave & Lexicometria (Top ${topN})`}
              </h2>
              <div className="rounded-xl border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px]">
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>{isEn ? 'Keyword' : 'Palavra-chave'}</TableHead>
                      <TableHead className="text-right">Docs</TableHead>
                      <TableHead className="text-right">{isEn ? 'Citations' : 'Citações'}</TableHead>
                      <TableHead className="text-right">h</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody className="text-xs">
                    {tables.keywords.slice(0, topN).map((k, idx) => (
                      <TableRow key={k.entity}>
                        <TableCell className="font-semibold text-muted-foreground">{idx + 1}</TableCell>
                        <TableCell className="font-bold text-foreground">{k.entity}</TableCell>
                        <TableCell className="text-right tabular-nums">{k.docCount}</TableCell>
                        <TableCell className="text-right tabular-nums font-semibold">{k.citations}</TableCell>
                        <TableCell className="text-right tabular-nums">{k.h}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* Gráfico 5: Nuvem de Palavras-Chave */}
          {selection.chartKeywords && (
            <ReportChartImage image={wordCloudChartImg} {...REPORT_CHART_SIZE.wordCloud} alt="Nuvem de Palavras-Chave" />
          )}

          {/* 7. Mapeamento Temático por IA */}
          {selection.themes && clustering && clustering.clusters.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn
                  ? `6. AI Thematic Clusters (Silhouette: ${clustering.silhouette.toFixed(3)})`
                  : `6. Agrupamento Temático por IA (Silhouette: ${clustering.silhouette.toFixed(3)})`}
              </h2>
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {clustering.clusters.map((c) => {
                  const share = active.length > 0 ? (c.size / active.length) * 100 : 0;
                  return (
                    <div key={c.clusterId} className="rounded-xl border border-border/80 bg-card p-3.5 shadow-2xs space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs font-bold text-foreground truncate">Tema {c.clusterId + 1}</p>
                        <Badge variant="purple" className="text-[10px]">
                          {share.toFixed(1)}%
                        </Badge>
                      </div>
                      <p className="text-[11px] text-muted-foreground">
                        <strong>{c.size}</strong> {isEn ? 'documents' : 'artigos'}
                      </p>
                      <p className="text-[10px] text-muted-foreground/80 italic truncate">
                        {c.topTerms.slice(0, 5).join(', ')}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Gráfico 6: Distribuição de Temas por IA */}
          {selection.chartThemes && (
            <ReportChartImage image={themesChartImg} {...REPORT_CHART_SIZE.themes} alt="Distribuição Temática por IA" />
          )}

          {/* 8. Topologia da Rede */}
          {selection.networkTopology && sna && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? '7. Deep Knowledge Ecology & Network Topology' : '7. Topologia da Rede & Ecologia Profunda'}
              </h2>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Density' : 'Densidade'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">{sna.global.density.toFixed(4)}</p>
                </div>
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Clustering' : 'Clustering Médio'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">{sna.global.clustering.toFixed(4)}</p>
                </div>
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Shannon Entropy' : 'Entropia de Shannon'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">{sna.global.entropy.toFixed(3)}</p>
                </div>
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Global Efficiency' : 'Eficiência Global'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">
                    {typeof sna.global.efficiency === 'number' ? sna.global.efficiency.toFixed(4) : String(sna.global.efficiency)}
                  </p>
                </div>
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Mean Degree' : 'Grau Médio'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">{sna.global.meanDegree.toFixed(2)}</p>
                </div>
                <div className="rounded-lg border border-border/70 p-2.5 bg-muted/20">
                  <p className="text-[10px] font-semibold text-muted-foreground">{isEn ? 'Power Law Exponent' : 'Lei de Potência'}</p>
                  <p className="text-sm font-bold tabular-nums text-foreground">{sna.global.powerLawExponent.toFixed(2)}</p>
                </div>
              </div>
            </div>
          )}

          {/* Gráfico 7: Rede de Coocorrência (Louvain) */}
          {selection.chartNetwork && (
            <ReportChartImage image={networkChartImg} {...REPORT_CHART_SIZE.network} alt="Rede de Coocorrência e Comunidades" />
          )}

          {/* 9. Top Documentos Mais Citados */}
          {selection.topDocuments && sortedTopDocs.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-bold text-foreground border-b border-border/60 pb-1.5">
                {isEn ? `8. Highly Cited Seminal Documents (Top ${topN})` : `8. Documentos Mais Citados da Base (Top ${topN})`}
              </h2>
              <div className="rounded-xl border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px]">
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>{isEn ? 'Title' : 'Título'}</TableHead>
                      <TableHead>{isEn ? 'Authors' : 'Autores'}</TableHead>
                      <TableHead className="text-right">{isEn ? 'Year' : 'Ano'}</TableHead>
                      <TableHead className="text-right">{isEn ? 'Citations' : 'Citações'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody className="text-xs">
                    {sortedTopDocs.map((d, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-semibold text-muted-foreground">{idx + 1}</TableCell>
                        <TableCell className="font-bold text-foreground max-w-72 truncate">
                          {titleCol ? String(d[titleCol] ?? '') : '—'}
                        </TableCell>
                        <TableCell className="text-muted-foreground max-w-40 truncate">
                          {authCol ? String(d[authCol] ?? '') : '—'}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          {toNumeric(d[FIELD.YEAR_CLEAN]) ?? '—'}
                        </TableCell>
                        <TableCell className="text-right tabular-nums font-bold text-amber-600">
                          {toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* Footer do Relatório */}
          <div className="border-t border-border/80 pt-4 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
            <p>
              Simetrics · Plataforma de Inteligência Bibliométrica · Desenvolvido por{' '}
              <a
                href="https://gustavosimas.com"
                target="_blank"
                rel="noopener noreferrer"
                className="font-bold text-primary hover:underline"
              >
                Gustavo Simas
              </a>
            </p>
            <p className="text-[11px] italic">
              {isEn ? 'Document rendered client-side.' : 'Documento processado localmente no navegador.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
