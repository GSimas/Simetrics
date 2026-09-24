import { Suspense, useEffect, useState } from 'react';
import { BarChart3, FileText, FolderOpen, Network, Search } from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { GithubButton, SettingsButton } from '@/components/HeaderActions';
import { KpiTicker } from '@/components/KpiTicker';
import { TutorialTriggerButton } from '@/components/TutorialTriggerButton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BuyMeCoffeeButton } from '@/components/BuyMeCoffeeButton';
import { LandingScreen } from '@/features/landing/LandingScreen';
import { type TranslationKey } from '@/lib/i18n/translations';
import { lazyWithPreload, whenIdle } from '@/lib/lazy';
import { useHashRoute } from '@/lib/use-hash-route';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { useProjectStore } from '@/state/project.store';
import { useTour } from '@/state/tour.store';

// Cada aba é um chunk próprio: a landing (o primeiro paint) não baixa nem executa o
// código do workspace. Ao entrar no workspace, as abas são pré-carregadas no ócio, então a
// troca de aba não passa pelo fallback do Suspense.
const OverviewTab = lazyWithPreload(() => import('@/features/overview/OverviewTab'));
const NetworksTab = lazyWithPreload(() => import('@/features/networks/NetworksTab'));
const SearchTab = lazyWithPreload(() => import('@/features/search/SearchTab'));
const ReportTab = lazyWithPreload(() => import('@/features/report/ReportTab'));
// Modal do tutorial e tour: só quando alguém os abre.
const TutorialModal = lazyWithPreload(() =>
  import('@/components/TutorialModal').then((module) => ({ default: module.TutorialModal })),
);
// Assistente (cliente de IA, ferramentas, markdown): só no workspace, baixado no ócio.
const ChatWidget = lazyWithPreload(() =>
  import('@/features/chat/ChatWidget').then((module) => ({ default: module.ChatWidget })),
);
const GuidedTour = lazyWithPreload(() =>
  import('@/features/tour/GuidedTour').then((module) => ({ default: module.GuidedTour })),
);

const TABS = [
  {
    value: 'overview',
    labelKey: 'tab_overview' as TranslationKey,
    Icon: BarChart3,
    iconColor: 'text-muted-foreground group-data-[state=active]:text-highlight',
    Panel: OverviewTab,
  },
  {
    value: 'networks',
    labelKey: 'tab_networks' as TranslationKey,
    Icon: Network,
    iconColor: 'text-muted-foreground group-data-[state=active]:text-highlight',
    Panel: NetworksTab,
  },
  {
    value: 'search',
    labelKey: 'tab_search' as TranslationKey,
    Icon: Search,
    iconColor: 'text-muted-foreground group-data-[state=active]:text-highlight',
    Panel: SearchTab,
  },
  {
    value: 'report',
    labelKey: 'tab_report' as TranslationKey,
    Icon: FileText,
    iconColor: 'text-muted-foreground group-data-[state=active]:text-highlight',
    Panel: ReportTab,
  },
] as const;

export default function App() {
  const documentCount = useDataset((state) => state.active?.length ?? 0);
  const t = useLocale((state) => state.t);
  const activeTab = useNavigation((state) => state.activeTab);
  const setActiveTab = useNavigation((state) => state.setActiveTab);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  // Montado desde a primeira abertura: fechar precisa do componente para a animação.
  const [tutorialMounted, setTutorialMounted] = useState(false);
  const [route, navigate] = useHashRoute();
  const tourActive = useTour((state) => state.active);

  const openTutorial = (): void => {
    setTutorialMounted(true);
    setTutorialOpen(true);
  };

  // O tour percorre o workspace: iniciado na landing, entra nele primeiro.
  useEffect(() => {
    if (tourActive && route.view === 'landing') navigate('workspace');
  }, [tourActive, route.view, navigate]);

  // No workspace, as abas são baixadas no ócio — a primeira troca de aba já as encontra.
  useEffect(() => {
    if (route.view !== 'workspace') return;
    return whenIdle(() => {
      void OverviewTab.preload();
      void NetworksTab.preload();
      void SearchTab.preload();
      void ReportTab.preload();
      void ChatWidget.preload();
    });
  }, [route.view]);

  // Com uma base carregada, os gráficos virão: baixa seus bundles em segundo plano para
  // que o primeiro gráfico aberto não pisque em "Carregando gráfico…".
  useEffect(() => {
    if (documentCount === 0) return;
    return whenIdle(() => {
      void import('@/components/charts/SigmaGraph');
      void import('@/components/charts/RadialGraph');
      void import('@/components/charts/WordCloud');
    });
  }, [documentCount]);

  // Recuperação de `#/workspace/<id>` — recarregar a página, ou navegar via
  // back/forward do navegador entre dois projetos, cai aqui: só re-hidrata quando o
  // projeto pedido pela rota é diferente do que já está carregado (evita reabrir em
  // laço a cada render). Um id inexistente/corrompido volta para a landing.
  useEffect(() => {
    if (route.view !== 'workspace' || !route.projectId) return;
    if (useProjectStore.getState().activeProjectId === route.projectId) return;

    void (async () => {
      await useProjectStore.getState().open(route.projectId!);
      if (useProjectStore.getState().error) navigate('landing');
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route.view, route.projectId]);

  if (route.view === 'landing') {
    return (
      <>
        <LandingScreen navigate={navigate} onOpenTutorial={openTutorial} />
        {/* Montado aqui também: sem isso, abrir o tutorial a partir da landing não
            renderizaria nada, já que este early-return substitui a árvore inteira do
            workspace (onde o modal normalmente vive, mais abaixo). */}
        {tutorialMounted && (
          <Suspense fallback={null}>
            <TutorialModal open={tutorialOpen} onOpenChange={setTutorialOpen} />
          </Suspense>
        )}
      </>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col justify-between relative">
      <div>
        <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-md">
          <div className="container flex flex-wrap items-center justify-between gap-4 py-3.5">
            <button
              type="button"
              onClick={() => navigate('landing')}
              className="flex items-center gap-4 text-left cursor-pointer transition-opacity duration-200 hover:opacity-80 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={t('landing_projects_title')}
              title={t('landing_projects_title')}
            >
              <span className="brand-mark h-10 sm:h-11 text-foreground" aria-hidden />
              <h1 className="sr-only">{t('app_title')}</h1>
              <span className="hidden 2xl:flex flex-col gap-1 border-l border-border pl-4">
                <span className="eyebrow max-w-[22rem] leading-snug">{t('app_subtitle')}</span>
              </span>
            </button>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                data-tour="projects"
                onClick={() => navigate('landing')}
                // No celular o texto some; o nome acessível não pode sumir junto.
                aria-label={t('nav_projects_btn')}
                className="header-chip cursor-pointer"
              >
                <FolderOpen className="size-3.5" aria-hidden />
                <span className="hidden sm:inline">{t('nav_projects_btn')}</span>
              </button>

              {documentCount > 0 && (
                <div className="eyebrow flex h-9 items-center gap-2 px-2 text-foreground">
                  <span className="size-1.5 rounded-full bg-highlight shadow-[0_0_10px_var(--highlight)]" />
                  <span className="tabular-nums">{documentCount.toLocaleString('pt-BR')}</span>
                  <span className="hidden text-muted-foreground xl:inline">{t('active_docs')}</span>
                </div>
              )}

              <SaveStatus />

              <TutorialTriggerButton onClick={openTutorial} />
              <SettingsButton />
              <GithubButton />
            </div>
          </div>
          <div data-tour="ticker">
            <KpiTicker />
          </div>
        </header>

        <main className="container py-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList data-tour="tabs" className="h-auto w-full flex-wrap justify-start gap-x-2">
              {TABS.map(({ value, labelKey, Icon, iconColor }, index) => (
                <TabsTrigger
                  key={value}
                  value={value}
                  className="group gap-2 px-3 py-3 text-xs sm:text-sm font-medium"
                >
                  <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground group-data-[state=active]:text-highlight">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <Icon className={cn('size-4 shrink-0 transition-colors', iconColor)} aria-hidden />
                  {t(labelKey)}
                </TabsTrigger>
              ))}
            </TabsList>

            {TABS.map(({ value, labelKey, Panel }) => (
              <TabsContent key={value} value={value} className="mt-5">
                {/* Uma aba que quebre (ou cujo chunk não baixe) não leva as outras junto. */}
                <ErrorBoundary variant="page" label={t(labelKey)}>
                  <Suspense fallback={<TabFallback />}>
                    <Panel />
                  </Suspense>
                </ErrorBoundary>
              </TabsContent>
            ))}
          </Tabs>
        </main>
      </div>

      {/* Widget Flutuante da Simi - Assistente Científica (FAB - Canto Inferior Direito).
          Se quebrar, some sozinho em vez de derrubar o workspace. */}
      <ErrorBoundary label="Simi" className="hidden">
        <Suspense fallback={null}>
          <ChatWidget />
        </Suspense>
      </ErrorBoundary>

      {/* Botão Flutuante de Café Luminoso (Pague-me um café - Canto Inferior Esquerdo) */}
      <BuyMeCoffeeButton />

      {/* Rodapé com crédito de desenvolvimento centralizado */}
      <footer className="mt-20 border-t border-border bg-background py-8">
        <div className="container flex flex-col items-center justify-between gap-4 text-center text-xs text-muted-foreground md:flex-row">
          <div className="flex items-center justify-center md:justify-start gap-3">
            <span className="brand-mark h-6 text-foreground" aria-hidden />
            <a
              href="https://scientata.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="eyebrow transition-colors hover:text-highlight"
            >
              {t('landing_family')} ↗
            </a>
          </div>

          <p className="text-center">
            {t('developed_by')}{' '}
            <a
              href="https://gustavosimas.com"
              target="_blank"
              rel="noopener noreferrer"
              className="accent-serif text-base underline-offset-4 hover:underline"
            >
              Gustavo Simas
            </a>
          </p>
        </div>
      </footer>

      {/* Modal do tutorial */}
      {tutorialMounted && (
        <Suspense fallback={null}>
          <TutorialModal open={tutorialOpen} onOpenChange={setTutorialOpen} />
        </Suspense>
      )}

      {/* Tour guiado — iniciado pelo modal do tutorial */}
      {tourActive && (
        <ErrorBoundary label="tour" className="hidden">
          <Suspense fallback={null}>
            <GuidedTour />
          </Suspense>
        </ErrorBoundary>
      )}
    </div>
  );
}

/** Espaço da aba enquanto o chunk chega — ocupa a altura para o rodapé não pular. */
function TabFallback() {
  const t = useLocale((state) => state.t);
  return (
    <div className="min-h-[60vh]" aria-busy="true">
      <span className="sr-only">{t('loading')}</span>
    </div>
  );
}

/**
 * Estado do salvamento automático. Componente próprio (e não estado do `App`): cada
 * checkpoint muda esse estado duas ou três vezes, e no `App` isso re-renderizava a aba
 * aberta inteira a cada salvamento.
 */
function SaveStatus() {
  const t = useLocale((state) => state.t);
  const saveStatus = useProjectStore((state) => state.saveStatus);
  const lastSavedAt = useProjectStore((state) => state.lastSavedAt);
  const error = useProjectStore((state) => state.error);
  if (saveStatus === 'idle') return null;
  return (
    <div
      className="eyebrow hidden h-9 items-center px-2 animate-in fade-in-0 xl:flex"
      title={saveStatus === 'error' ? (error ?? undefined) : undefined}
      role="status"
    >
      {saveStatus === 'saving' && <span>{t('project_save_status_saving')}</span>}
      {saveStatus === 'saved' && lastSavedAt && (
        <span>{t('project_save_status_saved').replace('{time}', new Date(lastSavedAt).toLocaleTimeString())}</span>
      )}
      {saveStatus === 'error' && <span className="text-destructive">{t('project_save_status_error')}</span>}
    </div>
  );
}
