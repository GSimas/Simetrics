import { useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Bot,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Globe2,
  HelpCircle,
  KeyRound,
  Network,
  Rocket,
  Search,
  Sparkles,
  Zap,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useLocale } from '@/state/locale.store';

interface TutorialStep {
  title: string;
  badge: string;
  badgeVariant: 'blue' | 'purple' | 'success' | 'warning' | 'indigo' | 'cyan';
  subtitle: string;
  description: string;
  highlights: { icon: typeof Zap; label: string; text: string }[];
  previewType: 'overview' | 'upload' | 'kpis' | 'networks' | 'byok' | 'search-ai';
}

const TUTORIAL_STEPS_PT: TutorialStep[] = [
  {
    title: 'Bem-vindo ao Simetrics',
    badge: 'Visão Geral',
    badgeVariant: 'blue',
    subtitle: 'Plataforma de Inteligência Bibliométrica e Mapeamento Científico',
    description:
      'O Simetrics transforma bases de dados brutas de exportações acadêmicas em visões estratégicas, redes de colaboração e clusters temáticos orientados por Inteligência Artificial.',
    highlights: [
      {
        icon: Zap,
        label: '100% no seu navegador',
        text: 'Seus dados nunca saem do seu computador. Todo o processamento é feito via Web Workers ultrarrápidos.',
      },
      {
        icon: Network,
        label: 'Ecologia do Conhecimento',
        text: 'Mapeie a estrutura intelectual através de grafos, métricas SNA, PCA e similaridade.',
      },
      {
        icon: Sparkles,
        label: 'Inteligência Artificial',
        text: 'Categorização temática automática e assistente conversacional sobre sua base de artigos.',
      },
    ],
    previewType: 'overview',
  },
  {
    title: '1. Importação & Deduplicação',
    badge: 'Entrada de Dados',
    badgeVariant: 'success',
    subtitle: 'Compatível com as principais bases acadêmicas do mundo',
    description:
      'Envie um ou múltiplos arquivos simultaneamente. O Simetrics identifica a origem e permite harmonizar bases heterogêneas sem conflito de metadados.',
    highlights: [
      {
        icon: FileText,
        label: 'Formatos suportados',
        text: 'RIS (SciELO, WoS, Scopus, Cochrane), CSV, Excel (XLSX) e TXT/NBIB (PubMed).',
      },
      {
        icon: Rocket,
        label: 'Base de demonstração instantânea',
        text: 'Clique em "Carregar exemplo" para explorar com quase 1.000 artigos reais imediatamente.',
      },
      {
        icon: Database,
        label: 'Deduplicação avançada',
        text: 'Filtre registros duplicados por DOI estrito ou por similaridade de título (Jaccard).',
      },
    ],
    previewType: 'upload',
  },
  {
    title: '2. Indicadores & Análises Visuais',
    badge: 'Métricas Cientométricas',
    badgeVariant: 'warning',
    subtitle: 'Estatística descritiva completa e modelos de impacto',
    description:
      'Acesse indicadores consolidados e tabelas analíticas para deep-dive em autores, países, fontes e termos com cálculo de índices clássicos.',
    highlights: [
      {
        icon: BarChart3,
        label: 'Índices h, g, i10 e m',
        text: 'Estatísticas completas de citação (média, mediana, desvio padrão) por entidade.',
      },
      {
        icon: FileSpreadsheet,
        label: 'Tabelas e Exportação CSV',
        text: 'Filtre e ordene qualquer tabela na tela e exporte os dados tratados com um clique.',
      },
      {
        icon: Globe2,
        label: 'Visualizações Especiais',
        text: 'Distribuição (Boxplot), Evolução temática (Sankey), Genética dos termos e Historiograph.',
      },
    ],
    previewType: 'kpis',
  },
  {
    title: '3. Redes de Grafos & Colaboração',
    badge: 'Grafos & Comunidades',
    badgeVariant: 'purple',
    subtitle: 'Descubra comunidades científicas e especializações temáticas',
    description:
      'Explore as conexões entre pesquisadores e descubra tópicos emergentes através de algoritmos de aprendizado não-supervisionado.',
    highlights: [
      {
        icon: Network,
        label: 'Redes Interativas (Sigma.js)',
        text: 'Grafos de coautoria, termos e países com detecção de comunidades via Louvain.',
      },
      {
        icon: Globe2,
        label: 'Colaboração Internacional',
        text: 'Mapa-múndi coroplético, grafo chordal e matriz de parcerias entre nações.',
      },
      {
        icon: Sparkles,
        label: 'Mapeamento Conceitual PCA',
        text: 'Projeção 2D e 3D da topologia do conhecimento com cálculo de Quociente Locacional.',
      },
    ],
    previewType: 'networks',
  },
  {
    title: '4. IA em Modo BYOK (Traga sua Chave)',
    badge: 'Bring Your Own Key',
    badgeVariant: 'indigo',
    subtitle: 'Use sua chave de API própria com total privacidade e liberdade',
    description:
      'Configure sua chave de API preferida (Google Gemini, OpenAI ChatGPT, Anthropic Claude, OpenRouter ou endpoint local compatível). Sua chave nunca sai do seu navegador.',
    highlights: [
      {
        icon: KeyRound,
        label: 'Múltiplos Provedores',
        text: 'Suporte nativo a Gemini 2.5, GPT-4o, Claude 3.5, OpenRouter e modelos locais via Ollama.',
      },
      {
        icon: Sparkles,
        label: 'Clusterização Temática',
        text: 'A IA lê os artigos representativos e dá nomes aos agrupamentos de pesquisa.',
      },
      {
        icon: Bot,
        label: 'Assistente Científico Flutuante',
        text: 'Acesse o assistente via widget flutuante no canto inferior direito a partir de qualquer aba.',
      },
    ],
    previewType: 'byok',
  },
  {
    title: '5. Motor de Busca & Dossiês',
    badge: 'Dossiê & Exploração',
    badgeVariant: 'cyan',
    subtitle: 'Investigação profunda de autores, periódicos e termos',
    description:
      'Consulte dossiês detalhados de qualquer autor, instituição ou termo e descubra perfis com DNA acadêmico similar.',
    highlights: [
      {
        icon: Search,
        label: 'Dossiê Acadêmico',
        text: 'Gere instantaneamente o perfil com métricas, nuvem de palavras e documentos mais citados.',
      },
      {
        icon: Zap,
        label: 'Entidades Semelhantes',
        text: 'Similaridade de Jaccard calculada sobre coautores, venues e vocabulário.',
      },
      {
        icon: CheckCircle2,
        label: 'Busca Multifacetada',
        text: 'Filtre instantaneamente por autores, países, venues e termos com busca em tempo real.',
      },
    ],
    previewType: 'search-ai',
  },
  {
    title: '6. Relatório Executivo Personalizado',
    badge: 'Exportação & Síntese',
    badgeVariant: 'blue',
    subtitle: 'Exporte relatórios científicos completos em PDF e DOCX',
    description:
      'Crie relatórios executivos sob medida escolhendo exatamente quais seções e tabelas incluir. Exporte diretamente sem passar pela janela de impressão.',
    highlights: [
      {
        icon: FileText,
        label: 'Seleção Modular',
        text: 'Escolha seções como resumo executivo, KPIs globais, rankings, clusters de IA e topologia da rede.',
      },
      {
        icon: Download,
        label: 'Formatos PDF & Word (DOCX)',
        text: 'Geração direta e instantânea no navegador em PDF diagramado em alta resolução ou DOCX editável.',
      },
      {
        icon: CheckCircle2,
        label: 'Pronto para Começar!',
        text: 'Reabra este tutorial a qualquer momento pelo botão no cabeçalho. Boa pesquisa!',
      },
    ],
    previewType: 'overview',
  },
];

const TUTORIAL_STEPS_EN: TutorialStep[] = [
  {
    title: 'Welcome to Simetrics',
    badge: 'Overview',
    badgeVariant: 'blue',
    subtitle: 'Bibliometric Intelligence & Scientific Mapping Platform',
    description:
      'Simetrics transforms raw export files from academic databases into strategic insights, collaboration networks, and AI-powered research clusters.',
    highlights: [
      {
        icon: Zap,
        label: '100% in your browser',
        text: 'Your research data never leaves your computer. All compute runs locally via high-performance Web Workers.',
      },
      {
        icon: Network,
        label: 'Knowledge Ecology',
        text: 'Map intellectual structures using graphs, SNA metrics, PCA, and similarity algorithms.',
      },
      {
        icon: Sparkles,
        label: 'Generative AI (BYOK)',
        text: 'Automated semantic theme labeling and conversational assistant grounded in your papers.',
      },
    ],
    previewType: 'overview',
  },
  {
    title: '1. Ingestion & Deduplication',
    badge: 'Data Input',
    badgeVariant: 'success',
    subtitle: 'Compatible with major academic bibliographic databases',
    description:
      'Upload one or multiple files at once. Simetrics identifies the source schema and integrates heterogeneous files without metadata conflicts.',
    highlights: [
      {
        icon: FileText,
        label: 'Supported Formats',
        text: 'RIS (SciELO, WoS, Scopus, Cochrane), CSV, Excel (XLSX), and TXT/NBIB (PubMed).',
      },
      {
        icon: Rocket,
        label: 'Instant Demo Dataset',
        text: 'Click "Load demo dataset" to explore almost 1,000 real papers immediately.',
      },
      {
        icon: Database,
        label: 'Smart Deduplication',
        text: 'Filter duplicate records by strict DOI match or Jaccard title similarity.',
      },
    ],
    previewType: 'upload',
  },
  {
    title: '2. Indicators & Visual Analyses',
    badge: 'Scientometrics',
    badgeVariant: 'warning',
    subtitle: 'Comprehensive descriptive statistics and impact metrics',
    description:
      'Access consolidated KPIs and deep-dive tables for authors, countries, venues, and keywords with classical scientometric indices.',
    highlights: [
      {
        icon: BarChart3,
        label: 'h, g, i10 & m Indices',
        text: 'Full citation statistics (mean, median, standard deviation) per academic entity.',
      },
      {
        icon: FileSpreadsheet,
        label: 'Tables & CSV Export',
        text: 'Sort and filter any table on screen and export treated data in CSV with one click.',
      },
      {
        icon: Globe2,
        label: 'Special Visualizations',
        text: 'Distribution (Boxplot), Thematic evolution (Sankey), Keyword genetics, and Historiograph.',
      },
    ],
    previewType: 'kpis',
  },
  {
    title: '3. Knowledge Networks & Graphs',
    badge: 'Graphs & Communities',
    badgeVariant: 'purple',
    subtitle: 'Uncover scientific communities and research clusters',
    description:
      'Explore co-authorship and keyword co-occurrence through unsupervised community detection and topological projections.',
    highlights: [
      {
        icon: Network,
        label: 'Interactive Graphs (Sigma.js)',
        text: 'Co-authorship and keyword graphs with Louvain community detection algorithm.',
      },
      {
        icon: Globe2,
        label: 'International Collaboration',
        text: 'Choropleth world map and chordal partner graph connecting countries.',
      },
      {
        icon: Sparkles,
        label: '2D/3D Concept PCA',
        text: 'Dimensionality reduction mapping knowledge schools and Locational Quotients (LQ).',
      },
    ],
    previewType: 'networks',
  },
  {
    title: '4. AI in BYOK Mode (Bring Your Own Key)',
    badge: 'Bring Your Own Key',
    badgeVariant: 'indigo',
    subtitle: 'Use your favorite AI provider with maximum privacy and freedom',
    description:
      'Bring your own API key (Google Gemini, OpenAI ChatGPT, Anthropic Claude, OpenRouter, or local models). Your key is stored strictly in your browser.',
    highlights: [
      {
        icon: KeyRound,
        label: 'Multiple Providers',
        text: 'Native support for Gemini 2.5, GPT-4o, Claude 3.5, OpenRouter, and Ollama/LM Studio.',
      },
      {
        icon: Sparkles,
        label: 'Theme Labeling',
        text: 'AI reads representative abstracts to synthesize high-level cluster names.',
      },
      {
        icon: Bot,
        label: 'Floating Scientific Assistant',
        text: 'Access the conversational assistant anytime via the floating widget in the bottom-right corner across all tabs.',
      },
    ],
    previewType: 'byok',
  },
  {
    title: '5. Search Engine & Academic Dossiers',
    badge: 'Dossier & Exploration',
    badgeVariant: 'cyan',
    subtitle: 'Deep investigation of authors, venues, and keywords',
    description:
      'Look up comprehensive profiles for any author or journal and discover peers with matching academic DNA.',
    highlights: [
      {
        icon: Search,
        label: 'Academic Dossier',
        text: 'Generate output stats, word clouds, and most cited papers for any entity.',
      },
      {
        icon: Zap,
        label: 'Similar Profiles',
        text: 'Jaccard similarity computed across co-authors, venues, and vocabulary.',
      },
      {
        icon: CheckCircle2,
        label: 'Multifaceted Search',
        text: 'Instantly filter across authors, countries, venues, and keywords with real-time feedback.',
      },
    ],
    previewType: 'search-ai',
  },
  {
    title: '6. Custom Executive Report',
    badge: 'Synthesis & Export',
    badgeVariant: 'blue',
    subtitle: 'Export comprehensive scientometric dossiers in PDF and DOCX',
    description:
      'Generate tailor-made research reports by choosing exactly which sections to include. Download directly without opening browser print dialogs.',
    highlights: [
      {
        icon: FileText,
        label: 'Modular Selection',
        text: 'Choose from executive summaries, global KPIs, rankings, AI clusters, and network topology.',
      },
      {
        icon: Download,
        label: 'Direct PDF & DOCX (Word)',
        text: 'Client-side vector PDF generation and editable Microsoft Word DOCX output.',
      },
      {
        icon: CheckCircle2,
        label: 'Ready to Explore!',
        text: 'You can reopen this guide anytime from the header button. Happy researching!',
      },
    ],
    previewType: 'overview',
  },
];

interface TutorialModalProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function TutorialModal({ open: controlledOpen, onOpenChange }: TutorialModalProps) {
  const locale = useLocale((state) => state.locale);
  const steps = locale === 'en' ? TUTORIAL_STEPS_EN : TUTORIAL_STEPS_PT;

  const [internalOpen, setInternalOpen] = useState(() => {
    if (typeof window === 'undefined') return false;
    return !localStorage.getItem('simetrics-tutorial-seen');
  });
  const [stepIndex, setStepIndex] = useState(0);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : internalOpen;

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      localStorage.setItem('simetrics-tutorial-seen', 'true');
    }
    if (isControlled && onOpenChange) {
      onOpenChange(open);
    } else {
      setInternalOpen(open);
    }
  };

  const currentStep: TutorialStep = steps[stepIndex] ?? steps[0]!;
  const isFirst = stepIndex === 0;
  const isLast = stepIndex === steps.length - 1;

  const nextStep = () => {
    if (isLast) {
      handleOpenChange(false);
    } else {
      setStepIndex((prev) => Math.min(prev + 1, steps.length - 1));
    }
  };

  const prevStep = () => {
    setStepIndex((prev) => Math.max(prev - 1, 0));
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-2xl sm:max-w-3xl overflow-hidden p-0 gap-0 border-border/80 bg-card shadow-2xl rounded-2xl">
        {/* Barra superior de progresso com gradiente */}
        <div className="h-1.5 w-full bg-muted overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 transition-all duration-300 ease-out"
            style={{ width: `${((stepIndex + 1) / steps.length) * 100}%` }}
          />
        </div>

        <div className="p-6 sm:p-7 space-y-5 max-h-[82vh] overflow-y-auto">
          {/* Header do Passo */}
          <DialogHeader className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Badge variant={currentStep.badgeVariant} className="text-xs font-semibold uppercase tracking-wider">
                {currentStep.badge} · {locale === 'en' ? 'Step' : 'Etapa'} {stepIndex + 1} / {steps.length}
              </Badge>
              <span className="text-xs text-muted-foreground font-medium">
                Simetrics {locale === 'en' ? 'Quickstart' : 'Guia Rápido'}
              </span>
            </div>
            <DialogTitle className="text-xl sm:text-2xl font-bold tracking-tight text-foreground">
              {currentStep.title}
            </DialogTitle>
            <DialogDescription className="text-sm font-medium text-foreground/80">
              {currentStep.subtitle}
            </DialogDescription>
          </DialogHeader>

          {/* Miniatura / Visual Ilustrativo */}
          <TutorialStepPreview type={currentStep.previewType} locale={locale} />

          {/* Descrição e Highlights */}
          <div className="space-y-3.5">
            <p className="text-xs sm:text-sm leading-relaxed text-muted-foreground">
              {currentStep.description}
            </p>

            <div className="grid gap-2.5 sm:grid-cols-3 pt-1">
              {currentStep.highlights.map(({ icon: Icon, label, text }) => (
                <div
                  key={label}
                  className="rounded-xl border border-border/80 bg-gradient-to-br from-muted/50 to-card p-3 shadow-2xs space-y-1.5 transition-all hover:border-primary/40"
                >
                  <div className="flex items-center gap-2 text-foreground font-semibold text-xs">
                    <div className="grid size-6 place-items-center rounded-md bg-primary/10 text-primary">
                      <Icon className="size-3.5" />
                    </div>
                    <span>{label}</span>
                  </div>
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    {text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Rodapé de Navegação */}
        <div className="border-t border-border/80 bg-muted/30 px-6 py-4 flex flex-wrap items-center justify-between gap-3">
          {/* Indicadores de bolinha */}
          <div className="flex items-center gap-1.5">
            {steps.map((_, index) => (
              <button
                key={index}
                type="button"
                onClick={() => setStepIndex(index)}
                className={`size-2.5 rounded-full transition-all ${
                  index === stepIndex
                    ? 'w-6 bg-primary'
                    : 'bg-muted-foreground/30 hover:bg-muted-foreground/60'
                }`}
                title={`Etapa ${index + 1}`}
                aria-label={`Etapa ${index + 1}`}
              />
            ))}
          </div>

          <div className="flex items-center gap-2">
            {!isFirst && (
              <Button variant="outline" size="sm" onClick={prevStep} className="gap-1.5 text-xs font-medium">
                <ArrowLeft className="size-3.5" />
                {locale === 'en' ? 'Previous' : 'Anterior'}
              </Button>
            )}

            <Button
              variant={isLast ? 'success' : 'gradient'}
              size="sm"
              onClick={nextStep}
              className="gap-1.5 text-xs font-semibold shadow-xs"
            >
              {isLast ? (
                <>
                  <CheckCircle2 className="size-3.5" />
                  {locale === 'en' ? 'Start Exploring' : 'Começar a Explorar'}
                </>
              ) : (
                <>
                  {locale === 'en' ? 'Next' : 'Próximo'}
                  <ArrowRight className="size-3.5" />
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** Componente visual representativo de cada etapa do Simetrics */
function TutorialStepPreview({
  type,
  locale,
}: {
  type: TutorialStep['previewType'];
  locale: 'pt' | 'en';
}) {
  const isEn = locale === 'en';

  if (type === 'overview') {
    return (
      <div className="rounded-xl border border-blue-200/80 bg-gradient-to-br from-blue-500/10 via-card to-indigo-500/10 p-4 shadow-2xs dark:border-blue-900/50">
        <div className="flex flex-wrap items-center justify-around gap-2 text-center">
          <div className="flex flex-col items-center p-2 rounded-lg bg-card/80 border border-border/60 shadow-2xs">
            <span className="text-xl">📂</span>
            <span className="mt-1 text-[11px] font-bold text-foreground">
              {isEn ? 'RIS / CSV Datasets' : 'Bases RIS / CSV'}
            </span>
            <span className="text-[10px] text-muted-foreground">Scopus, WoS, SciELO</span>
          </div>
          <span className="text-muted-foreground font-bold text-sm">➔</span>
          <div className="flex flex-col items-center p-2 rounded-lg bg-card/80 border border-blue-200 shadow-2xs">
            <span className="text-xl">⚡</span>
            <span className="mt-1 text-[11px] font-bold text-primary">Web Workers</span>
            <span className="text-[10px] text-muted-foreground">
              {isEn ? 'Local Compute' : 'Processamento Local'}
            </span>
          </div>
          <span className="text-muted-foreground font-bold text-sm">➔</span>
          <div className="flex flex-col items-center p-2 rounded-lg bg-card/80 border border-border/60 shadow-2xs">
            <span className="text-xl">📊</span>
            <span className="mt-1 text-[11px] font-bold text-foreground">
              {isEn ? 'Graphs & BYOK AI' : 'Grafos & IA (BYOK)'}
            </span>
            <span className="text-[10px] text-muted-foreground">
              {isEn ? 'Visual Scientometrics' : 'Epistemologia Visual'}
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'upload') {
    return (
      <div className="rounded-xl border border-emerald-200/80 bg-gradient-to-br from-emerald-500/10 via-card to-teal-500/10 p-4 shadow-2xs dark:border-emerald-900/50">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-bold text-foreground">
              {isEn ? 'Unified Upload & Parser' : 'Painel de Upload Unificado'}
            </span>
          </div>
          <div className="flex gap-1.5">
            <Badge variant="success" className="text-[10px]">
              {isEn ? 'Demo Ready (973 docs)' : 'Exemplo Pronto (973 docs)'}
            </Badge>
            <Badge variant="blue" className="text-[10px]">
              {isEn ? 'DOI Deduplication' : 'Deduplicação DOI'}
            </Badge>
          </div>
        </div>
        <div className="mt-3 flex items-center justify-center rounded-lg border-2 border-dashed border-emerald-300 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-950/20 py-3 text-center">
          <p className="text-xs text-emerald-800 dark:text-emerald-300 font-medium">
            {isEn
              ? 'Drag & drop RIS, CSV or Excel files or click "Load Demo Dataset"'
              : 'Arraste arquivos RIS, CSV ou Excel ou use "Carregar Exemplo"'}
          </p>
        </div>
      </div>
    );
  }

  if (type === 'kpis') {
    return (
      <div className="rounded-xl border border-amber-200/80 bg-gradient-to-br from-amber-500/10 via-card to-orange-500/10 p-3 shadow-2xs dark:border-amber-900/50">
        <div className="grid grid-cols-4 gap-2">
          <div className="rounded-lg border border-blue-200 bg-card p-2 text-center shadow-2xs">
            <p className="text-[10px] text-muted-foreground font-semibold">{isEn ? 'Docs' : 'Docs'}</p>
            <p className="text-sm font-bold text-blue-600">973</p>
          </div>
          <div className="rounded-lg border border-purple-200 bg-card p-2 text-center shadow-2xs">
            <p className="text-[10px] text-muted-foreground font-semibold">{isEn ? 'Authors' : 'Autores'}</p>
            <p className="text-sm font-bold text-purple-600">1.629</p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-card p-2 text-center shadow-2xs">
            <p className="text-[10px] text-muted-foreground font-semibold">{isEn ? 'Growth' : 'Crescimento'}</p>
            <p className="text-sm font-bold text-emerald-600">+14.2%</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-card p-2 text-center shadow-2xs">
            <p className="text-[10px] text-muted-foreground font-semibold">{isEn ? 'h-index' : 'Índice h'}</p>
            <p className="text-sm font-bold text-amber-600">38</p>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'networks') {
    return (
      <div className="rounded-xl border border-purple-200/80 bg-gradient-to-br from-purple-500/10 via-card to-indigo-500/10 p-3.5 shadow-2xs dark:border-purple-900/50">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-foreground">
            {isEn ? 'Heterogeneous Graph & Communities' : 'Grafo Heterogêneo & Clusters'}
          </span>
          <span className="text-[11px] text-purple-700 dark:text-purple-300 font-medium">Louvain + PCA</span>
        </div>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          <Badge variant="purple" className="text-[10px]">
            {isEn ? '🟣 Co-authorship Networks' : '🟣 Redes de Coautoria'}
          </Badge>
          <Badge variant="indigo" className="text-[10px]">
            {isEn ? '🌐 International Collab' : '🌐 Parcerias Internacionais'}
          </Badge>
          <Badge variant="cyan" className="text-[10px]">
            {isEn ? '🧬 2D/3D Concept PCA' : '🧬 Mapa Conceitual PCA'}
          </Badge>
        </div>
      </div>
    );
  }

  if (type === 'byok') {
    return (
      <div className="rounded-xl border border-indigo-200/80 bg-gradient-to-br from-indigo-500/10 via-card to-purple-500/10 p-3.5 shadow-2xs dark:border-indigo-900/50">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-foreground">
            {isEn ? 'Bring Your Own Key (BYOK)' : 'Traga sua Chave de IA (BYOK)'}
          </span>
          <Badge variant="indigo" className="text-[10px]">
            {isEn ? 'Direct Browser Connection' : 'Conexão Direta'}
          </Badge>
        </div>
        <div className="mt-2.5 flex flex-wrap gap-1.5 text-xs">
          <Badge variant="blue" className="text-[10px]">Google Gemini</Badge>
          <Badge variant="success" className="text-[10px]">OpenAI ChatGPT</Badge>
          <Badge variant="purple" className="text-[10px]">Anthropic Claude</Badge>
          <Badge variant="warning" className="text-[10px]">OpenRouter / Ollama</Badge>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-cyan-200/80 bg-gradient-to-br from-cyan-500/10 via-card to-blue-500/10 p-3.5 shadow-2xs dark:border-cyan-900/50">
      <div className="flex items-center justify-between text-xs">
        <span className="font-bold text-foreground">
          {isEn ? 'Search Dossiers & AI Assistant' : 'Busca de Dossiês & Assistente IA'}
        </span>
        <Badge variant="cyan" className="text-[10px]">BM25 + Streaming RAG</Badge>
      </div>
      <div className="mt-2.5 rounded-lg border border-border/80 bg-card p-2 text-xs text-muted-foreground flex items-center gap-2">
        <Bot className="size-4 text-emerald-600 shrink-0" />
        <span>
          {isEn
            ? '"What are the foundational papers and leading authors on this topic?"'
            : '"Quais são os documentos e autores mais influentes desta base?"'}
        </span>
      </div>
    </div>
  );
}

export function TutorialTriggerButton({ onClick }: { onClick: () => void }) {
  const t = useLocale((state) => state.t);

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className="h-9 gap-1.5 rounded-xl border-border/80 bg-card/80 text-xs font-semibold text-foreground shadow-2xs hover:border-primary/40 hover:bg-muted"
    >
      <HelpCircle className="size-3.5 text-primary" aria-hidden />
      <span>{t('tutorial_btn')}</span>
    </Button>
  );
}
