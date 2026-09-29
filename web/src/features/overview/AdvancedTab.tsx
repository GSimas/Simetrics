import { EmptyState } from '@/features/EmptyState';
import { SectionTitle } from '@/components/InfoTip';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { VisualAnalyses } from './VisualAnalyses';

/** Vista Análises Avançadas: as visualizações pesadas, antes num modal, agora numa aba. */
export default function AdvancedTab() {
  const active = useDataset((state) => state.active);
  const t = useLocale((state) => state.t);

  if (!active) return <EmptyState title={t('tab_advanced')} />;

  return (
    <div className="space-y-4" data-tour="visual">
      <SectionTitle title={t('visual_title')} info={t('visual_description')} />
      <VisualAnalyses dataset={active} />
    </div>
  );
}
