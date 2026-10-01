import { BarChart3, ClipboardList, Filter, GitBranch, ShieldCheck, TableProperties, type LucideIcon } from 'lucide-react';

import { REVIEW_STEPS, type StepProgress } from '@/core/review/progress';
import type { ReviewStep } from '@/state/review.store';
import type { ReviewCopy } from './copy';

/** As etapas na ordem do fluxo, com seus ícones — a barra lateral e a tela usam as mesmas. */
export const REVIEW_STEP_ITEMS: { value: ReviewStep; Icon: LucideIcon }[] = REVIEW_STEPS.map((value) => ({
  value,
  Icon: {
    protocol: ClipboardList,
    screening: Filter,
    quality: ShieldCheck,
    extraction: TableProperties,
    synthesis: BarChart3,
    prisma: GitBranch,
  }[value],
}));

const fill = (template: string, { done, total }: StepProgress) =>
  template.replace('{done}', done.toLocaleString()).replace('{total}', total.toLocaleString());

/** Uma linha sobre onde a etapa está: "5/7 itens", "Sem checklist no protocolo", "Pronta"… */
export function stepDetail(step: ReviewStep, progress: StepProgress, copy: ReviewCopy): string {
  switch (step) {
    case 'protocol':
      return fill(copy.progressItems, progress);
    case 'screening':
      return progress.state === 'waiting' ? copy.progressNoRecords : fill(copy.progressDecisions, progress);
    case 'quality':
    case 'extraction':
      if (progress.state === 'off') return step === 'quality' ? copy.progressNoChecklist : copy.progressNoForm;
      if (progress.state === 'waiting') return copy.progressAwaitingIncluded;
      return fill(copy.progressStudies, progress);
    case 'synthesis':
      return progress.state === 'done' ? copy.progressReady : copy.progressBuilding;
    case 'prisma':
      return progress.state === 'done' ? copy.progressReady : copy.progressAfterScreening;
  }
}
