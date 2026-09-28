import { useMemo, useState } from 'react';
import { BarChart3, ClipboardList, Filter, GitBranch, ShieldCheck, TableProperties } from 'lucide-react';

import { createReview } from '@/core/review/state';
import { getDeviceId } from '@/lib/device-id';
import { cn } from '@/lib/utils';
import { DemoCopyButton } from '@/components/DemoBanner';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useReview } from '@/state/review.store';
import { REVIEW_COPY } from './copy';
import { ExtractionPanel } from './ExtractionPanel';
import { PrismaPanel } from './PrismaPanel';
import { ProtocolPanel } from './ProtocolPanel';
import { QualityPanel } from './QualityPanel';
import { ScreeningPanel } from './ScreeningPanel';
import { SynthesisPanel } from './SynthesisPanel';

type Step = 'protocol' | 'screening' | 'quality' | 'extraction' | 'synthesis' | 'prisma';

const STEPS: { value: Step; Icon: typeof ClipboardList }[] = [
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
  const [step, setStep] = useState<Step>('protocol');
  const isDemo = useDataset((state) => state.isDemo);

  // O exemplo é só visualização: uma revisão precisa de um projeto salvo onde viver.
  if (isDemo) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-xl border border-dashed border-border p-10 text-center">
        <p className="eyebrow text-highlight">{copy.eyebrow}</p>
        <p className="max-w-xl text-sm text-muted-foreground">{copy.demoReadOnly}</p>
        <DemoCopyButton size="default" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <p className="eyebrow text-highlight">{copy.eyebrow}</p>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{copy.intro}</p>
      </div>

      <nav className="flex flex-wrap gap-2" aria-label={copy.eyebrow}>
        {STEPS.map(({ value, Icon }, index) => (
          <button
            key={value}
            type="button"
            aria-current={step === value ? 'step' : undefined}
            onClick={() => setStep(value)}
            className={cn(
              'flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium transition-colors sm:text-sm',
              step === value
                ? 'border-highlight text-foreground'
                : 'border-border text-muted-foreground hover:text-foreground',
            )}
          >
            <span className={cn('font-mono text-[10px]', step === value && 'text-highlight')}>{index + 1}</span>
            <Icon className={cn('size-4', step === value && 'text-highlight')} aria-hidden />
            {copy.steps[value]}
          </button>
        ))}
      </nav>

      {step === 'protocol' && <ProtocolPanel review={review} copy={copy} />}
      {step === 'screening' && <ScreeningPanel review={review} copy={copy} />}
      {step === 'quality' && <QualityPanel review={review} copy={copy} onEditProtocol={() => setStep('protocol')} />}
      {step === 'extraction' && <ExtractionPanel review={review} copy={copy} onEditProtocol={() => setStep('protocol')} />}
      {step === 'synthesis' && <SynthesisPanel review={review} copy={copy} />}
      {step === 'prisma' && <PrismaPanel review={review} copy={copy} />}
    </div>
  );
}
