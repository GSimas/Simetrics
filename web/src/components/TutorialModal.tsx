import { useState, type ReactNode } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Award,
  BarChart3,
  BookOpen,
  Bot,
  CheckCircle2,
  Compass,
  Copy,
  Cpu,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  FileUp,
  FolderOpen,
  Globe2,
  KeyRound,
  Network,
  Orbit,
  Plug,
  Rocket,
  Search,
  Share2,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Upload,
  Users,
  Zap,
  type LucideIcon,
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
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useTour } from '@/state/tour.store';

interface TutorialStep {
  title: string;
  badge: string;
  subtitle: string;
  description: string;
  highlights: { icon: typeof Zap; label: string; text: string }[];
  previewType: 'overview' | 'upload' | 'kpis' | 'networks' | 'byok' | 'search-ai';
}

const TUTORIAL_STEPS_PT: TutorialStep[] = [
  {
    title: 'Bem-vindo ao Simetrics',
    badge: 'Visão Geral',
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
        label: 'Simi - Assistente Científica Flutuante',
        text: 'Acesse a Simi via widget flutuante no canto inferior direito a partir de qualquer aba.',
      },
    ],
    previewType: 'byok',
  },
  {
    title: '5. Motor de Busca & Dossiês',
    badge: 'Dossiê & Exploração',
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
        label: 'Simi - Floating Scientific Assistant',
        text: 'Access Simi anytime via the floating widget in the bottom-right corner across all tabs.',
      },
    ],
    previewType: 'byok',
  },
  {
    title: '5. Search Engine & Academic Dossiers',
    badge: 'Dossier & Exploration',
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
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Sempre controlado — a tela inicial (LandingScreen) é quem decide a primeira
 * experiência do usuário agora, este modal só abre por ação explícita (botão do header
 * ou "Ver tutorial" na landing).
 */
export function TutorialModal({ open: isOpen, onOpenChange: handleOpenChange }: TutorialModalProps) {
  const locale = useLocale((state) => state.locale);
  const steps = locale === 'en' ? TUTORIAL_STEPS_EN : TUTORIAL_STEPS_PT;
  const [stepIndex, setStepIndex] = useState(0);
  const startTour = useTour((state) => state.start);

  // O tour percorre a tela de verdade: o modal sai da frente antes de ele começar.
  const launchTour = (): void => {
    handleOpenChange(false);
    startTour();
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
        {/* Barra superior de progresso */}
        <div className="h-1.5 w-full bg-muted overflow-hidden">
          <div
            className="h-full bg-highlight transition-all duration-300 ease-out"
            style={{ width: `${((stepIndex + 1) / steps.length) * 100}%` }}
          />
        </div>

        {/* Chave por etapa: cada passo entra com fade, em vez de trocar o texto de uma vez. */}
        <div
          key={stepIndex}
          className="p-6 sm:p-7 space-y-5 max-h-[82vh] overflow-y-auto duration-300 animate-in fade-in-0 slide-in-from-right-2"
        >
          {/* Header do Passo */}
          <DialogHeader className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Badge variant="outline" className="border-highlight/50 text-highlight">
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
                  className="border border-border bg-card p-3 space-y-1.5 transition-colors hover:border-highlight/60"
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

            {(isFirst || isLast) && (
              <div className="flex flex-col gap-3 border border-highlight/40 bg-highlight/5 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-3">
                  <span className="grid size-8 shrink-0 place-items-center border border-highlight/50 text-highlight">
                    <Compass className="size-4" aria-hidden />
                  </span>
                  <div className="space-y-0.5">
                    <p className="text-sm font-semibold text-foreground">
                      {locale === 'en' ? 'Prefer to see it in practice?' : 'Prefere ver na prática?'}
                    </p>
                    <p className="text-xs leading-snug text-muted-foreground">
                      {locale === 'en'
                        ? 'The guided tour walks through every block, chart and table on the real screen.'
                        : 'O tour guiado percorre cada bloco, gráfico e tabela na tela de verdade.'}
                    </p>
                  </div>
                </div>
                <Button size="sm" onClick={launchTour} className="shrink-0 gap-1.5">
                  <Compass className="size-3.5" aria-hidden />
                  {locale === 'en' ? 'Start guided tour' : 'Iniciar tour guiado'}
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Rodapé de Navegação */}
        <div className="border-t border-border/80 bg-muted/30 px-6 py-4 flex flex-wrap items-center justify-between gap-3">
          {/* Indicadores de bolinha */}
          <div className="flex flex-wrap items-center gap-3">
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
          {!isFirst && !isLast && (
            <button
              type="button"
              onClick={launchTour}
              className="eyebrow flex cursor-pointer items-center gap-1.5 transition-colors hover:text-highlight"
            >
              <Compass className="size-3.5" aria-hidden />
              {locale === 'en' ? 'Guided tour' : 'Tour guiado'}
            </button>
          )}
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

/** Nó do diagrama de fluxo da etapa de boas-vindas. */
function FlowNode({
  Icon,
  title,
  caption,
  accent,
}: {
  Icon: LucideIcon;
  title: string;
  caption: string;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        'flex min-w-0 flex-1 flex-col items-center gap-1.5 border p-3 text-center',
        accent ? 'border-highlight/60' : 'border-border',
      )}
    >
      <span
        className={cn(
          'grid size-8 place-items-center border',
          accent ? 'border-highlight/50 text-highlight' : 'border-border text-foreground',
        )}
      >
        <Icon className="size-4" aria-hidden />
      </span>
      <span className={cn('text-[11px] font-semibold', accent ? 'text-highlight' : 'text-foreground')}>
        {title}
      </span>
      <span className="eyebrow text-[9.5px]">{caption}</span>
    </div>
  );
}

/** Etiqueta com ícone — no lugar dos emojis das prévias. */
function Tag({ Icon, children }: { Icon: LucideIcon; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 border border-border px-2 py-1 font-mono text-[10.5px] uppercase tracking-[0.08em] text-muted-foreground">
      <Icon className="size-3.5 text-highlight" aria-hidden />
      {children}
    </span>
  );
}

function PreviewFrame({
  Icon,
  title,
  aside,
  children,
}: {
  Icon: LucideIcon;
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="border border-border border-l-2 border-l-highlight bg-card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-xs font-semibold text-foreground">
          <Icon className="size-4 text-highlight" aria-hidden />
          {title}
        </span>
        {aside}
      </div>
      {children}
    </div>
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
      <div className="border border-border border-l-2 border-l-highlight bg-card p-4">
        <div className="flex items-stretch gap-2">
          <FlowNode
            Icon={FolderOpen}
            title={isEn ? 'RIS / CSV datasets' : 'Bases RIS / CSV'}
            caption="Scopus · WoS · SciELO"
          />
          <ArrowRight className="size-4 shrink-0 self-center text-muted-foreground" aria-hidden />
          <FlowNode Icon={Cpu} title="Web Workers" caption={isEn ? 'Local compute' : 'Processamento local'} accent />
          <ArrowRight className="size-4 shrink-0 self-center text-muted-foreground" aria-hidden />
          <FlowNode
            Icon={Share2}
            title={isEn ? 'Graphs & BYOK AI' : 'Grafos & IA (BYOK)'}
            caption={isEn ? 'Visual scientometrics' : 'Epistemologia visual'}
          />
        </div>
      </div>
    );
  }

  if (type === 'upload') {
    return (
      <PreviewFrame
        Icon={Upload}
        title={isEn ? 'Unified upload & parser' : 'Painel de upload unificado'}
        aside={
          <span className="flex flex-wrap gap-1.5">
            <Tag Icon={Rocket}>{isEn ? 'Demo · 973 docs' : 'Exemplo · 973 docs'}</Tag>
            <Tag Icon={Copy}>{isEn ? 'DOI dedup' : 'Deduplicação DOI'}</Tag>
          </span>
        }
      >
        <div className="flex items-center justify-center gap-2 border border-dashed border-highlight/40 px-3 py-3 text-center text-xs text-muted-foreground">
          <FileUp className="size-4 shrink-0 text-highlight" aria-hidden />
          {isEn
            ? 'Drop RIS, CSV or Excel files, or click "Load demo"'
            : 'Arraste arquivos RIS, CSV ou Excel ou use "Carregar exemplo"'}
        </div>
      </PreviewFrame>
    );
  }

  if (type === 'kpis') {
    const kpis = [
      { Icon: BookOpen, label: 'Docs', value: '973' },
      { Icon: Users, label: isEn ? 'Authors' : 'Autores', value: '1.630' },
      { Icon: TrendingUp, label: isEn ? 'Growth' : 'Crescimento', value: '6,82%' },
      { Icon: Award, label: isEn ? 'h-index' : 'Índice h', value: '38' },
    ];
    return (
      <div className="grid grid-cols-2 gap-2 border border-border border-l-2 border-l-highlight bg-card p-3 sm:grid-cols-4">
        {kpis.map(({ Icon, label, value }) => (
          <div key={label} className="border border-border p-2.5">
            <div className="flex items-center justify-between gap-1">
              <span className="eyebrow text-[9.5px]">{label}</span>
              <Icon className="size-3.5 text-highlight" aria-hidden />
            </div>
            <p className="mt-1 text-lg font-semibold tabular-nums text-foreground">{value}</p>
          </div>
        ))}
      </div>
    );
  }

  if (type === 'networks') {
    return (
      <PreviewFrame
        Icon={Network}
        title={isEn ? 'Heterogeneous graph & communities' : 'Grafo heterogêneo & comunidades'}
        aside={<span className="eyebrow text-highlight">Louvain + PCA</span>}
      >
        <div className="flex flex-wrap gap-1.5">
          <Tag Icon={Users}>{isEn ? 'Co-authorship' : 'Coautoria'}</Tag>
          <Tag Icon={Globe2}>{isEn ? 'International collab' : 'Parcerias internacionais'}</Tag>
          <Tag Icon={Orbit}>{isEn ? '2D/3D concept PCA' : 'Mapa conceitual PCA'}</Tag>
        </div>
      </PreviewFrame>
    );
  }

  if (type === 'byok') {
    return (
      <PreviewFrame
        Icon={KeyRound}
        title={isEn ? 'Bring your own key (BYOK)' : 'Traga sua chave de IA (BYOK)'}
        aside={<Tag Icon={ShieldCheck}>{isEn ? 'Direct connection' : 'Conexão direta'}</Tag>}
      >
        <div className="flex flex-wrap gap-1.5">
          {['Google Gemini', 'OpenAI ChatGPT', 'Anthropic Claude', 'OpenRouter / Ollama'].map((provider) => (
            <Tag key={provider} Icon={Plug}>
              {provider}
            </Tag>
          ))}
        </div>
      </PreviewFrame>
    );
  }

  return (
    <PreviewFrame
      Icon={Search}
      title={isEn ? 'Search dossiers & AI assistant' : 'Dossiês de busca & assistente IA'}
      aside={<span className="eyebrow text-highlight">BM25 + streaming RAG</span>}
    >
      <div className="flex items-center gap-2 border border-border p-2.5 text-xs text-muted-foreground">
        <Bot className="size-4 shrink-0 text-highlight" aria-hidden />
        <span>
          {isEn
            ? '"What are the foundational papers and leading authors on this topic?"'
            : '"Quais são os documentos e autores mais influentes desta base?"'}
        </span>
      </div>
    </PreviewFrame>
  );
}
