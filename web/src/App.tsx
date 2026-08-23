import { BarChart3, Bot, ClipboardList, Network, Search } from 'lucide-react';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import OverviewTab from '@/features/overview/OverviewTab';
import NetworksTab from '@/features/networks/NetworksTab';
import SearchTab from '@/features/search/SearchTab';
import ChatTab from '@/features/chat/ChatTab';
import FeedbackTab from '@/features/feedback/FeedbackTab';
import { useDataset } from '@/state/dataset.store';

const TABS = [
  { value: 'overview', label: 'Informações Principais', Icon: BarChart3, Panel: OverviewTab },
  { value: 'networks', label: 'Redes e Grafos', Icon: Network, Panel: NetworksTab },
  { value: 'search', label: 'Motor de Busca', Icon: Search, Panel: SearchTab },
  { value: 'chat', label: 'Assistente Científico', Icon: Bot, Panel: ChatTab },
  { value: 'feedback', label: 'Feedback', Icon: ClipboardList, Panel: FeedbackTab },
] as const;

export default function App() {
  const documentCount = useDataset((state) => state.active?.length ?? 0);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container flex flex-wrap items-center justify-between gap-3 py-5">
          <div className="flex items-center gap-3">
            <img src="/simetrics-logo.png" alt="" className="h-10 w-auto" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">Simetrics</h1>
              <p className="text-sm text-muted-foreground">
                Análise Bibliométrica e Cientométrica
              </p>
            </div>
          </div>

          {documentCount > 0 && (
            <p className="text-sm text-muted-foreground tabular-nums">
              {documentCount.toLocaleString('pt-BR')} documentos na base ativa
            </p>
          )}
        </div>
      </header>

      <main className="container py-6">
        <Tabs defaultValue="overview">
          <TabsList className="h-auto flex-wrap">
            {TABS.map(({ value, label, Icon }) => (
              <TabsTrigger key={value} value={value} className="gap-2">
                <Icon className="size-4" aria-hidden />
                {label}
              </TabsTrigger>
            ))}
          </TabsList>

          {TABS.map(({ value, Panel }) => (
            <TabsContent key={value} value={value}>
              <Panel />
            </TabsContent>
          ))}
        </Tabs>
      </main>
    </div>
  );
}
