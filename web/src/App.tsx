import { useEffect, useState } from 'react';
import { BarChart3, FileText, FolderOpen, Network, Search } from 'lucide-react';

import { GithubButton, SettingsButton } from '@/components/HeaderActions';
import { KpiTicker } from '@/components/KpiTicker';
import { TutorialModal, TutorialTriggerButton } from '@/components/TutorialModal';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import OverviewTab from '@/features/overview/OverviewTab';
import NetworksTab from '@/features/networks/NetworksTab';
import SearchTab from '@/features/search/SearchTab';
import ReportTab from '@/features/report/ReportTab';
import { ChatWidget } from '@/features/chat/ChatWidget';
import { BuyMeCoffeeButton } from '@/components/BuyMeCoffeeButton';
import { LandingScreen } from '@/features/landing/LandingScreen';
import { type TranslationKey } from '@/lib/i18n/translations';
import { useHashRoute } from '@/lib/use-hash-route';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { useProjectStore } from '@/state/project.store';

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
  const [route, navigate] = useHashRoute();
  const saveStatus = useProjectStore((state) => state.saveStatus);
  const lastSavedAt = useProjectStore((state) => state.lastSavedAt);

  // Com uma base carregada, os gráficos virão: baixa seus bundles em segundo plano para
  // que o primeiro gráfico aberto não pisque em "Carregando gráfico…".
  useEffect(() => {
    if (documentCount === 0) return;
    const preload = (): void => {
      void import('@/components/charts/PlotlyChart');
      void import('@/components/charts/SigmaGraph');
      void import('@/components/charts/RadialGraph');
      void import('@/components/charts/WordCloud');
    };
    // Safari não tem requestIdleCallback; um atraso curto cumpre o mesmo papel.
    const timer = setTimeout(preload, 800);
    return () => clearTimeout(timer);
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
        <LandingScreen navigate={navigate} onOpenTutorial={() => setTutorialOpen(true)} />
        {/* Montado aqui também: sem isso, abrir o tutorial a partir da landing não
            renderizaria nada, já que este early-return substitui a árvore inteira do
            workspace (onde o modal normalmente vive, mais abaixo). */}
        <TutorialModal open={tutorialOpen} onOpenChange={setTutorialOpen} />
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
                <span className="eyebrow text-highlight">
                  {t('app_version')} · {t('landing_highlight_1_label')}
                </span>
                <span className="eyebrow max-w-[22rem] leading-snug">{t('app_subtitle')}</span>
              </span>
            </button>

            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => navigate('landing')} className="header-chip cursor-pointer">
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

              {saveStatus !== 'idle' && (
                <div
                  className="eyebrow hidden h-9 items-center px-2 animate-in fade-in-0 xl:flex"
                  title={saveStatus === 'error' ? (useProjectStore.getState().error ?? undefined) : undefined}
                >
                  {saveStatus === 'saving' && <span>{t('project_save_status_saving')}</span>}
                  {saveStatus === 'saved' && lastSavedAt && (
                    <span>
                      {t('project_save_status_saved').replace(
                        '{time}',
                        new Date(lastSavedAt).toLocaleTimeString(),
                      )}
                    </span>
                  )}
                  {saveStatus === 'error' && (
                    <span className="text-destructive">{t('project_save_status_error')}</span>
                  )}
                </div>
              )}

              <TutorialTriggerButton onClick={() => setTutorialOpen(true)} />
              <SettingsButton />
              <GithubButton />
            </div>
          </div>
          <KpiTicker />
        </header>

        <main className="container py-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="h-auto w-full flex-wrap justify-start gap-x-2">
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

            {TABS.map(({ value, Panel }) => (
              <TabsContent key={value} value={value} className="mt-5">
                <Panel />
              </TabsContent>
            ))}
          </Tabs>
        </main>
      </div>

      {/* Widget Flutuante da Simi - Assistente Científica (FAB - Canto Inferior Direito) */}
      <ChatWidget />

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
      <TutorialModal open={tutorialOpen} onOpenChange={setTutorialOpen} />
    </div>
  );
}
