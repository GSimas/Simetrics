import { useEffect, useMemo, useState } from 'react';
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

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import { generatePdfReport, type ReportSectionsSelection } from './pdf-generator';
import { generateDocxReport } from './docx-generator';
import {
  renderHorizontalBarChart,
  renderNetworkGraphCanvas,
  renderProductionTimelineCanvas,
  renderThemesPieChart,
  renderWordCloudCanvas,
  renderWorldCollaborationMapCanvas,
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
  const { locale } = useLocale();
  const isEn = locale === 'en';

  const [selection, setSelection] = useState<ReportSectionsSelection>(DEFAULT_SELECTION);
  const [topN, setTopN] = useState<number>(15);
  const [collaboration, setCollaboration] = useState<CollaborationNetwork | null>(null);
  const [isExportingPdf, setIsExportingPdf] = useState(false);
  const [isExportingDocx, setIsExportingDocx] = useState(false);

  useEffect(() => {
    if (active) {
      if (!overview) void computeOverview();
      if (!tables) void computeTables();
      if (!sna) void computeSna();
      if (!network) void computeNetwork('Palavras-chave', 40, 'Grau Absoluto');

      getAnalyticsWorker()
        .collaboration(active, 30)
        .then((res) => setCollaboration(res))
        .catch(() => setCollaboration(null));
    }
  }, [active, overview, tables, sna, network, computeOverview, computeTables, computeSna, computeNetwork]);

  // Gera as imagens dos gráficos sob demanda com suporte a internacionalização
  const productionChartImg = useMemo(() => {
    if (!overview || overview.docsPerYear.length === 0) return null;
    try {
      return renderProductionTimelineCanvas(overview.docsPerYear, { width: 1000, height: 400, locale });
    } catch {
      return null;
    }
  }, [overview, locale]);

  const authorsChartImg = useMemo(() => {
    if (!tables || tables.authors.length === 0) return null;
    try {
      const items = tables.authors.slice(0, 10).map((a) => ({
        label: a.entity,
        value: a.docCount,
        sub: `${a.citations} ${isEn ? 'cit.' : 'cit.'} | h=${a.h}`,
      }));
      const title = isEn ? 'Top 10 Most Prolific Authors (Published Papers)' : 'Top 10 Autores Mais Produtivos';
      return renderHorizontalBarChart(title, items, { width: 1000, height: 420, locale });
    } catch {
      return null;
    }
  }, [tables, locale, isEn]);

  const countriesChartImg = useMemo(() => {
    if (!tables || tables.countries.length === 0) return null;
    try {
      const items = tables.countries.slice(0, 10).map((c) => ({
        label: c.entity,
        value: c.docCount,
        sub: `${c.citations} ${isEn ? 'cit.' : 'cit.'}`,
      }));
      const title = isEn ? 'Top 10 Leading Countries by Scientific Output' : 'Top 10 Países com Maior Produção Científica';
      return renderHorizontalBarChart(title, items, { width: 1000, height: 420, locale });
    } catch {
      return null;
    }
  }, [tables, locale, isEn]);

  const worldMapChartImg = useMemo(() => {
    if (!collaboration || collaboration.nodes.length === 0) return null;
    try {
      return renderWorldCollaborationMapCanvas(collaboration, { width: 1000, height: 500, locale });
    } catch {
      return null;
    }
  }, [collaboration, locale]);

  const networkChartImg = useMemo(() => {
    if (!network || network.nodes.length === 0) return null;
    try {
      return renderNetworkGraphCanvas(network.nodes, network.edges, { width: 1000, height: 520, locale });
    } catch {
      return null;
    }
  }, [network, locale]);

  const themesChartImg = useMemo(() => {
    if (!clustering || clustering.clusters.length === 0) return null;
    try {
      const items = clustering.clusters.map((c) => ({
        clusterId: c.clusterId,
        name: `Tema ${c.clusterId + 1}`,
        docCount: c.size,
        share: active ? (c.size / active.length) * 100 : 0,
      }));
      return renderThemesPieChart(items, { width: 1000, height: 420, locale });
    } catch {
      return null;
    }
  }, [clustering, active, locale]);

  const wordCloudChartImg = useMemo(() => {
    if (!tables || tables.keywords.length === 0) return null;
    try {
      return renderWordCloudCanvas(tables.keywords, { width: 1000, height: 420, locale });
    } catch {
      return null;
    }
  }, [tables, locale]);

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

  const handleExportPdf = () => {
    try {
      setIsExportingPdf(true);
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
    } finally {
      setIsExportingPdf(false);
    }
  };

  const handleExportDocx = async () => {
    try {
      setIsExportingDocx(true);
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

  const columns = collectColumns(active);
  const titleCol = pickColumn(columns, FIELD_CANDIDATES.title);
  const authCol = pickColumn(columns, FIELD_CANDIDATES.authors);

  const sortedTopDocs = [...active]
    .sort((a, b) => (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0))
    .slice(0, topN);

  return (
    <div className="space-y-6">
      {/* 1. Painel de Controle de Exportação */}
      <Card className="border-t-4 border-t-blue-600 shadow-sm">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <CardTitle className="text-xl font-bold text-foreground">
                {isEn ? 'Custom Scientometric Report Generator' : 'Gerador de Relatório Cientométrico Personalizado'}
              </CardTitle>
              <CardDescription className="mt-1">
                {isEn
                  ? 'Select the specific tables, KPIs, and charts to include in your executive report. Export in high-resolution PDF or Microsoft Word DOCX.'
                  : 'Escolha as tabelas, indicadores e gráficos a incluir no relatório. Exporte diretamente em PDF diagramado em alta resolução ou Microsoft Word DOCX.'}
              </CardDescription>
            </div>

            {/* Botões de Ação de Download */}
            <div className="flex flex-wrap items-center gap-2.5">
              <Button
                variant="default"
                onClick={handleExportPdf}
                disabled={isExportingPdf}
                className="gap-2 bg-gradient-to-r from-red-600 to-rose-600 font-bold text-white shadow-sm hover:from-red-700 hover:to-rose-700 cursor-pointer"
              >
                <Download className="size-4" />
                {isExportingPdf ? (isEn ? 'Building PDF...' : 'Gerando PDF...') : isEn ? 'Export PDF' : 'Baixar PDF'}
              </Button>

              <Button
                variant="outline"
                onClick={handleExportDocx}
                disabled={isExportingDocx}
                className="gap-2 border-blue-300 font-bold text-blue-700 hover:bg-blue-50 dark:border-blue-800 dark:text-blue-400 dark:hover:bg-blue-950/40 cursor-pointer"
              >
                <FileCheck className="size-4 text-blue-600" />
                {isExportingDocx ? (isEn ? 'Building DOCX...' : 'Gerando DOCX...') : isEn ? 'Export DOCX (Word)' : 'Baixar DOCX (Word)'}
              </Button>
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
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
            {isEn ? 'Live Document Preview (with Embedded Charts & Tables)' : 'Pré-visualização do Relatório (com Gráficos e Tabelas)'}
          </p>
          <span className="text-xs text-muted-foreground">
            {isEn ? 'Formatted according to selected sections' : 'Diagramado conforme as seções e gráficos selecionados acima'}
          </span>
        </div>

        <div className="mx-auto max-w-4xl rounded-2xl border border-border/90 bg-card p-6 sm:p-10 shadow-lg space-y-8 text-foreground transition-all">
          {/* Header do Relatório */}
          <div className="border-b border-border/80 pb-6">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="grid size-11 place-items-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-sm">
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
                    {(overview.summary.bibliometrix.growthRate * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Gráfico 1: Produção Anual */}
          {selection.chartProduction && productionChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={productionChartImg}
                alt="Gráfico de Evolução da Produção Científica"
                className="w-full h-auto object-contain"
              />
            </div>
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
          {selection.chartAuthors && authorsChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={authorsChartImg}
                alt="Gráfico dos Top Autores"
                className="w-full h-auto object-contain"
              />
            </div>
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
          {selection.chartCountries && countriesChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={countriesChartImg}
                alt="Gráfico dos Top Países"
                className="w-full h-auto object-contain"
              />
            </div>
          )}

          {/* Gráfico 4: Mapa-Múndi de Colaboração Internacional */}
          {selection.chartWorldMap && worldMapChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={worldMapChartImg}
                alt="Mapa Global de Colaboração Internacional"
                className="w-full h-auto object-contain"
              />
            </div>
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
          {selection.chartKeywords && wordCloudChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={wordCloudChartImg}
                alt="Nuvem de Palavras-Chave"
                className="w-full h-auto object-contain"
              />
            </div>
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
          {selection.chartThemes && themesChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={themesChartImg}
                alt="Distribuição Temática por IA"
                className="w-full h-auto object-contain"
              />
            </div>
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
          {selection.chartNetwork && networkChartImg && (
            <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
              <img
                src={networkChartImg}
                alt="Rede de Coocorrência e Comunidades"
                className="w-full h-auto object-contain"
              />
            </div>
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
