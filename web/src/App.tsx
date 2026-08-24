import { useState } from 'react';
import { BarChart3, ClipboardList, FileText, Network, Search } from 'lucide-react';

import { AiSettingsButton, AiSettingsModal } from '@/components/AiSettingsModal';
import { LanguageToggle } from '@/components/LanguageToggle';
import { ThemeToggle } from '@/components/ThemeToggle';
import { TutorialModal, TutorialTriggerButton } from '@/components/TutorialModal';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import OverviewTab from '@/features/overview/OverviewTab';
import NetworksTab from '@/features/networks/NetworksTab';
import SearchTab from '@/features/search/SearchTab';
import ReportTab from '@/features/report/ReportTab';
import FeedbackTab from '@/features/feedback/FeedbackTab';
import { ChatWidget } from '@/features/chat/ChatWidget';
import { type TranslationKey } from '@/lib/i18n/translations';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';

const TABS = [
  {
    value: 'overview',
    labelKey: 'tab_overview' as TranslationKey,
    Icon: BarChart3,
    iconColor: 'text-blue-500 group-data-[state=active]:text-inherit',
    Panel: OverviewTab,
  },
  {
    value: 'networks',
    labelKey: 'tab_networks' as TranslationKey,
    Icon: Network,
    iconColor: 'text-purple-500 group-data-[state=active]:text-inherit',
    Panel: NetworksTab,
  },
  {
    value: 'search',
    labelKey: 'tab_search' as TranslationKey,
    Icon: Search,
    iconColor: 'text-cyan-600 group-data-[state=active]:text-inherit',
    Panel: SearchTab,
  },
  {
    value: 'report',
    labelKey: 'tab_report' as TranslationKey,
    Icon: FileText,
    iconColor: 'text-indigo-500 group-data-[state=active]:text-inherit',
    Panel: ReportTab,
  },
  {
    value: 'feedback',
    labelKey: 'tab_feedback' as TranslationKey,
    Icon: ClipboardList,
    iconColor: 'text-amber-500 group-data-[state=active]:text-inherit',
    Panel: FeedbackTab,
  },
] as const;

export default function App() {
  const documentCount = useDataset((state) => state.active?.length ?? 0);
  const t = useLocale((state) => state.t);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [aiSettingsOpen, setAiSettingsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col justify-between relative">
      <div>
        {/* Barra de destaque com gradiente no topo */}
        <div className="h-1.5 w-full bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600" />

        <header className="sticky top-0 z-40 border-b border-border/80 bg-card/90 shadow-2xs backdrop-blur-md">
          <div className="container flex flex-wrap items-center justify-between gap-4 py-3.5 sm:py-4">
            <div className="flex items-center gap-3.5">
              <div className="flex size-14 sm:size-16 shrink-0 items-center justify-center rounded-2xl border border-primary/25 bg-gradient-to-br from-blue-50 to-indigo-50/70 p-2 shadow-xs dark:from-blue-950 dark:to-indigo-950">
                <img
                  src="/simetrics-logo.png"
                  alt="Simetrics Logo"
                  className="h-11 sm:h-12 w-auto object-contain transition-transform duration-300 hover:scale-105"
                />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight text-foreground">
                    {t('app_title')}
                  </h1>
                  <span className="rounded-md bg-blue-100 px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider text-blue-700 dark:bg-blue-950 dark:text-blue-300 shadow-2xs">
                    {t('app_version')}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground font-medium">
                  {t('app_subtitle')}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {documentCount > 0 && (
                <div className="flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50/80 px-3.5 py-1.5 text-xs font-semibold text-blue-700 shadow-2xs dark:border-blue-900/60 dark:bg-blue-950/60 dark:text-blue-300">
                  <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="tabular-nums">{documentCount.toLocaleString('pt-BR')}</span>
                  <span className="font-normal text-muted-foreground">{t('active_docs')}</span>
                </div>
              )}

              <TutorialTriggerButton onClick={() => setTutorialOpen(true)} />
              <AiSettingsButton onClick={() => setAiSettingsOpen(true)} />
              <LanguageToggle />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className="container py-6">
          <Tabs defaultValue="overview">
            <TabsList className="h-auto flex-wrap gap-1.5 rounded-xl border border-border/80 bg-card p-1.5 shadow-xs">
              {TABS.map(({ value, labelKey, Icon, iconColor }) => (
                <TabsTrigger
                  key={value}
                  value={value}
                  className="group gap-2 rounded-lg px-3.5 py-2 text-xs sm:text-sm font-medium transition-all hover:bg-muted/80 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-sm"
                >
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

      {/* Widget Flutuante do Assistente Científico (FAB) */}
      <ChatWidget />

      {/* Rodapé com crédito de desenvolvimento */}
      <footer className="mt-20 border-t border-border/80 bg-card/60 py-6 backdrop-blur-xs">
        <div className="container flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted-foreground text-center sm:text-left">
          <div className="flex items-center gap-2">
            <img src="/simetrics-logo.png" alt="" className="h-5 w-auto object-contain" />
            <span>Simetrics · {t('app_subtitle')}</span>
          </div>
          <p>
            {t('developed_by')}{' '}
            <a
              href="https://gustavosimas.com"
              target="_blank"
              rel="noopener noreferrer"
              className="font-bold text-primary underline underline-offset-4 transition-colors hover:text-primary/80"
            >
              Gustavo Simas
            </a>
          </p>
        </div>
      </footer>

      {/* Modais de Tutorial e Configuração de IA */}
      <TutorialModal open={tutorialOpen} onOpenChange={setTutorialOpen} />
      <AiSettingsModal open={aiSettingsOpen} onOpenChange={setAiSettingsOpen} />
    </div>
  );
}
