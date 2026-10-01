import { Suspense, useEffect, useState } from 'react';
import { FolderOpen } from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { SettingsButton } from '@/components/HeaderActions';
import { KpiTicker } from '@/components/KpiTicker';
import { TutorialTriggerButton } from '@/components/TutorialTriggerButton';
import { BuyMeCoffeeButton } from '@/components/BuyMeCoffeeButton';
import { DemoBanner } from '@/components/DemoBanner';
import { WorkspaceSidebar } from '@/components/WorkspaceSidebar';
import { WorkspaceBackdrop } from '@/components/AmbientBackdrop';
import { preloadBibliometricViews } from '@/features/bibliometrics/views';
import { LandingScreen } from '@/features/landing/LandingScreen';
import { numberLocale } from '@/lib/i18n/labels';
import { type TranslationKey } from '@/lib/i18n/translations';
import { lazyWithPreload, whenIdle } from '@/lib/lazy';
import { useHashRoute } from '@/lib/use-hash-route';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation, type WorkspaceTab } from '@/state/navigation.store';
import { useProjectStore } from '@/state/project.store';
import { useTour } from '@/state/tour.store';

// Cada aba é um chunk próprio: a landing (o primeiro paint) não baixa nem executa o
// código do workspace. Ao entrar no workspace, as abas são pré-carregadas no ócio, então a
// troca de aba não passa pelo fallback do Suspense.
// Análise Bibliométrica: Informações Principais, Redes, Análises Avançadas e Relatório, cada vista em seu chunk.
const DataTab = lazyWithPreload(() => import('@/features/data/DataTab'));
const BibliometricsTab = lazyWithPreload(() => import('@/features/bibliometrics/BibliometricsTab'));
const SearchTab = lazyWithPreload(() => import('@/features/search/SearchTab'));
const ReviewTab = lazyWithPreload(() => import('@/features/review/ReviewTab'));
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

const PANELS: Record<WorkspaceTab, { labelKey: TranslationKey; Panel: typeof DataTab }> = {
  data: { labelKey: 'nav_data', Panel: DataTab },
  search: { labelKey: 'tab_search', Panel: SearchTab },
  bibliometrics: { labelKey: 'tab_bibliometrics', Panel: BibliometricsTab },
  review: { labelKey: 'tab_review', Panel: ReviewTab },
};

export default function App() {
  const documentCount = useDataset((state) => state.active?.length ?? 0);
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const activeTab = useNavigation((state) => state.activeTab);
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
      void DataTab.preload();
      void BibliometricsTab.preload();
      preloadBibliometricViews();
      void SearchTab.preload();
      void ReviewTab.preload();
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

  const { labelKey: activeLabelKey, Panel: ActivePanel } = PANELS[activeTab];

  return (
    // Sem bg aqui: o fundo é o WorkspaceBackdrop, animado e desfocado. `isolate` cria o
    // contexto de empilhamento: sem ele o z-index -1 do fundo ficava atrás do bg do body.
    <div className="isolate flex min-h-screen text-foreground">
      <WorkspaceBackdrop />
      <WorkspaceSidebar />
      <div className="min-w-0 flex-1 flex flex-col justify-between relative">
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
                {documentCount > 0 && (
                  <div className="eyebrow flex h-9 items-center gap-2 px-2 text-foreground">
                    <span className="size-1.5 rounded-full bg-highlight shadow-[0_0_10px_var(--highlight)]" />
                    <span className="tabular-nums">{documentCount.toLocaleString(numberLocale(locale))}</span>
                    <span className="hidden text-muted-foreground xl:inline">{t('active_docs')}</span>
                  </div>
                )}

                <SaveStatus />

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

                <TutorialTriggerButton onClick={openTutorial} />
                <SettingsButton />
              </div>
            </div>
            <div data-tour="ticker">
              <KpiTicker />
            </div>
          </header>

          <DemoBanner />

          <main className="container py-6">
            {/* h1 (marca) → h2 (tela) → h3 (seções): sem ele, a hierarquia pulava do h1
                para o h3 (heading-order no Lighthouse). Só para leitores de tela. */}
            <h2 className="sr-only">{t(activeLabelKey)}</h2>
            {/* Uma tela que quebre (ou cujo chunk não baixe) não leva as outras junto. */}
            <ErrorBoundary key={activeTab} variant="page" label={t(activeLabelKey)}>
              <Suspense fallback={<TabFallback />}>
                <ActivePanel />
              </Suspense>
            </ErrorBoundary>
          </main>
        </div>

        {/* Widget Flutuante da Simi - Assistente Científica (FAB - Canto Inferior Direito).
            Só nas análises: em Dados não há o que perguntar, e no Motor de Busca a conversa
            já tem tela própria (a mesma conversa, pelo store compartilhado).
            Se quebrar, some sozinho em vez de derrubar o workspace. */}
        {(activeTab === 'bibliometrics' || activeTab === 'review') && (
          <ErrorBoundary label="Simi" className="hidden">
            <Suspense fallback={null}>
              <ChatWidget />
            </Suspense>
          </ErrorBoundary>
        )}

        {/* Botão Flutuante de Café Luminoso (Pague-me um café - Canto Inferior Esquerdo) */}
        <BuyMeCoffeeButton />

        {/* Rodapé com crédito de desenvolvimento centralizado */}
        <footer className="app-footer mt-20 border-t border-border bg-background/70 py-8 backdrop-blur-md">
          {/* Reserva a área dos botões flutuantes (Simi e café, canto inferior direito): embaixo
              no layout empilhado, à direita no layout em linha — o crédito nunca fica coberto. */}
          <div className="container flex flex-col items-center justify-between gap-4 pb-24 text-center text-xs text-muted-foreground md:flex-row md:pb-0 md:pr-24">
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
    </div>
  );
}

/** Espaço da aba enquanto o chunk chega — ocupa a altura para o rodapé não pular. */
function TabFallback() {
  const t = useLocale((state) => state.t);
  return (
    // data-tab-fallback: o rodapé fica invisível enquanto isto existe (ver index.css).
    <div className="min-h-[60vh]" aria-busy="true" data-tab-fallback>
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
