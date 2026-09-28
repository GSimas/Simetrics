import { useMemo } from 'react';
import { BarChart3, ClipboardList, Eye, Filter, GitBranch, ShieldCheck, TableProperties } from 'lucide-react';

import { DemoCopyButton } from '@/components/DemoBanner';
import { createReview } from '@/core/review/state';
import { getDeviceId } from '@/lib/device-id';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useReview, useReviewNav, useReviewReadOnly, type ReviewStep } from '@/state/review.store';
import { REVIEW_COPY } from './copy';
import { ExtractionPanel } from './ExtractionPanel';
import { PANEL_ENTER, PRESS } from './motion';
import { PrismaPanel } from './PrismaPanel';
import { ProtocolPanel } from './ProtocolPanel';
import { QualityPanel } from './QualityPanel';
import { ScreeningPanel } from './ScreeningPanel';
import { SynthesisPanel } from './SynthesisPanel';

const STEPS: { value: ReviewStep; Icon: typeof ClipboardList }[] = [
  { value: 'protocol', Icon: ClipboardList },
  { value: 'screening', Icon: Filter },
  { value: 'quality', Icon: ShieldCheck },
  { value: 'extraction', Icon: TableProperties },
  { value: 'synthesis', Icon: BarChart3 },
  { value: 'prisma', Icon: GitBranch },
];

/**
 * Aba da revisão sistematizada no fluxo do Parsifal: planejamento (protocolo, checklist de
 * qualidade, formulário de extração) → condução (triagem, qualidade, extração) → síntese
 * e relato (PRISMA, relatório).
 * Trabalha sobre a mesma base do projeto que as análises bibliométricas.
 */
export default function ReviewTab() {
  const locale = useLocale((state) => state.locale);
  const copy = REVIEW_COPY[locale];
  const stored = useReview((state) => state.review);
  // Sem revisão ainda, mostra um protocolo em branco; a primeira edição o cria no store.
  const blank = useMemo(() => createReview(getDeviceId()), []);
  const review = stored ?? blank;
  const step = useReviewNav((state) => state.step);
  const setStep = useReviewNav((state) => state.setStep);
  const readOnly = useReviewReadOnly();
  const editProtocol = () => setStep('protocol');

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <p className="eyebrow text-highlight">{copy.eyebrow}</p>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{copy.intro}</p>
      </div>

      {readOnly && (
        <div
          role="status"
          className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-highlight/40 bg-highlight/5 px-4 py-3 shadow-[0_0_32px_-16px_var(--highlight)] animate-in fade-in-0 slide-in-from-top-1 duration-300"
        >
          <Eye className="size-4 shrink-0 text-highlight" aria-hidden />
          <p className="min-w-[14rem] flex-1 text-xs leading-relaxed text-muted-foreground sm:text-sm">{copy.demoReadOnly}</p>
          <DemoCopyButton />
        </div>
      )}

      <nav className="flex flex-wrap gap-2" aria-label={copy.eyebrow} data-tour="review-steps">
        {STEPS.map(({ value, Icon }, index) => {
          const active = step === value;
          return (
            <button
              key={value}
              type="button"
              aria-current={active ? 'step' : undefined}
              onClick={() => setStep(value)}
              className={cn(
                'group flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium sm:text-sm',
                PRESS,
                active
                  ? 'border-highlight bg-highlight/5 text-foreground shadow-[0_0_24px_-10px_var(--highlight)]'
                  : 'border-border text-muted-foreground hover:border-highlight/50 hover:text-foreground',
              )}
            >
              <span className={cn('font-mono text-[10px] transition-colors duration-200', active && 'text-highlight')}>
                {index + 1}
              </span>
              <Icon
                className={cn(
                  'size-4 transition-[color,transform] duration-200 group-hover:scale-110 motion-reduce:transform-none',
                  active && 'text-highlight',
                )}
                aria-hidden
              />
              {copy.steps[value]}
            </button>
          );
        })}
      </nav>

      {/* A chave nova a cada etapa refaz a entrada: a troca desliza em vez de piscar. */}
      <div key={step} className={PANEL_ENTER} data-tour={`review-step-${step}`}>
        {step === 'protocol' && <ProtocolPanel review={review} copy={copy} />}
        {step === 'screening' && <ScreeningPanel review={review} copy={copy} />}
        {step === 'quality' && <QualityPanel review={review} copy={copy} onEditProtocol={editProtocol} />}
        {step === 'extraction' && <ExtractionPanel review={review} copy={copy} onEditProtocol={editProtocol} />}
        {step === 'synthesis' && <SynthesisPanel review={review} copy={copy} />}
        {step === 'prisma' && <PrismaPanel review={review} copy={copy} />}
      </div>
    </div>
  );
}
