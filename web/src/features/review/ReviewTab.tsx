import { useMemo } from 'react';
import { ArrowRight, Eye, Lock, Sparkles } from 'lucide-react';

import { DemoCopyButton } from '@/components/DemoBanner';
import { STEPS_WITHOUT_DATA } from '@/core/review/progress';
import { createReview } from '@/core/review/state';
import { getDeviceId } from '@/lib/device-id';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNextReviewAction, useReview, useReviewNav, useReviewProgress, useReviewReadOnly } from '@/state/review.store';
import { REVIEW_COPY } from './copy';
import { ExtractionPanel } from './ExtractionPanel';
import { PANEL_ENTER, PRESS } from './motion';
import { goToNextAction, goToProtocolBlock } from './next-step';
import { ProgressRing } from './progress-parts';
import { REVIEW_STEP_ITEMS, stepDetail } from './progress-steps';
import { PrismaPanel } from './PrismaPanel';
import { ProtocolPanel } from './ProtocolPanel';
import { QualityPanel } from './QualityPanel';
import { ScreeningPanel } from './ScreeningPanel';
import { SynthesisPanel } from './SynthesisPanel';


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
  const readOnly = useReviewReadOnly();
  // Sem base, só o protocolo: uma etapa que dependa dela abre o protocolo no lugar.
  const hasData = useDataset((state) => state.active !== null);
  const chosen = useReviewNav((state) => state.step);
  const step = hasData || STEPS_WITHOUT_DATA.includes(chosen) ? chosen : 'protocol';

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

      {!readOnly && <NextStepCard />}

      <ReviewProgressPanel />

      {/* A chave nova a cada etapa refaz a entrada: a troca desliza em vez de piscar. */}
      <div key={step} className={PANEL_ENTER} data-tour={`review-step-${step}`}>
        {step === 'protocol' && <ProtocolPanel review={review} copy={copy} />}
        {step === 'screening' && <ScreeningPanel review={review} copy={copy} />}
        {step === 'quality' && <QualityPanel review={review} copy={copy} onEditProtocol={() => goToProtocolBlock('quality-checklist')} />}
        {step === 'extraction' && <ExtractionPanel review={review} copy={copy} onEditProtocol={() => goToProtocolBlock('extraction-form')} />}
        {step === 'synthesis' && <SynthesisPanel review={review} copy={copy} />}
        {step === 'prisma' && <PrismaPanel review={review} copy={copy} />}
      </div>
    </div>
  );
}

/**
 * Onde a revisão está: a barra do total e uma faixa por etapa, cada uma com a própria
 * completude — o avanço fica à vista a cada decisão. As etapas também são navegáveis aqui,
 * além da barra lateral.
 */
function ReviewProgressPanel() {
  const locale = useLocale((state) => state.locale);
  const copy = REVIEW_COPY[locale];
  const step = useReviewNav((state) => state.step);
  const setStep = useReviewNav((state) => state.setStep);
  const progress = useReviewProgress();
  const hasData = useDataset((state) => state.active !== null);
  const percent = Math.round(progress.overall * 100);

  return (
    <section
      data-tour="review-steps"
      aria-label={copy.progressTitle}
      className="space-y-4 rounded-xl border border-border bg-card p-4 sm:p-5"
    >
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="space-y-1">
          <p className="eyebrow">{copy.progressTitle}</p>
          <p className="max-w-xl text-xs text-muted-foreground">{copy.progressHint}</p>
        </div>
        <p className="text-4xl font-medium leading-none tracking-[-0.05em] tabular-nums">
          {percent}
          <span className="text-lg text-muted-foreground">%</span>
        </p>
      </div>

      <div
        role="progressbar"
        aria-label={copy.progressTitle}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-2 overflow-hidden rounded-full bg-muted"
      >
        <div
          className="h-full rounded-full bg-highlight shadow-[0_0_16px_var(--highlight)] transition-[width] duration-700 ease-out"
          style={{ width: `${percent}%` }}
        />
      </div>

      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
        {REVIEW_STEP_ITEMS.map(({ value, Icon }, index) => {
          const item = progress.steps[value];
          const locked = !hasData && !STEPS_WITHOUT_DATA.includes(value);
          const active = step === value && !locked;
          return (
            <li key={value}>
              <button
                type="button"
                aria-current={active ? 'step' : undefined}
                aria-disabled={locked || undefined}
                onClick={() => !locked && setStep(value)}
                className={cn(
                  'group flex h-full w-full flex-col gap-2 rounded-lg border p-3 text-left',
                  PRESS,
                  active
                    ? 'border-highlight bg-highlight/5 shadow-[0_0_24px_-10px_var(--highlight)]'
                    : 'border-border hover:border-highlight/50',
                  (item.state === 'off' || locked) && !active && 'opacity-60',
                  locked && 'cursor-not-allowed hover:border-border',
                )}
              >
                <span className="flex items-center gap-2">
                  <span className={cn('font-mono text-[10px] text-muted-foreground', active && 'text-highlight')}>
                    {index + 1}
                  </span>
                  <Icon className={cn('size-4 text-muted-foreground', active && 'text-highlight')} aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-xs font-medium sm:text-sm">{copy.steps[value]}</span>
                  {locked ? <Lock className="size-3.5 text-muted-foreground" aria-hidden /> : <ProgressRing progress={item} />}
                </span>
                <span className="h-1 overflow-hidden rounded-full bg-muted">
                  <span
                    className={cn(
                      'block h-full rounded-full transition-[width] duration-500 ease-out',
                      item.state === 'done' ? 'bg-highlight' : 'bg-highlight/70',
                    )}
                    style={{ width: `${Math.round(item.ratio * 100)}%` }}
                  />
                </span>
                <span className="text-[11px] leading-snug text-muted-foreground">{locked ? copy.progressNeedsData : stepDetail(value, item, copy)}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/**
 * O que falta fazer primeiro, com um clique para chegar lá: abre a etapa, rola até o
 * campo e o faz piscar em destaque (ver `goToNextAction`).
 */
function NextStepCard() {
  const locale = useLocale((state) => state.locale);
  const t = useLocale((state) => state.t);
  const copy = REVIEW_COPY[locale];
  const action = useNextReviewAction();
  const done = action.kind === 'report';
  const text = copy.nextActions[action.kind].replace('{count}', (action.count ?? 0).toLocaleString(locale === 'en' ? 'en' : 'pt-BR'));

  return (
    <button
      type="button"
      onClick={() => goToNextAction(action)}
      className={cn(
        'group flex w-full items-center gap-4 rounded-xl border border-highlight/50 bg-highlight/5 px-4 py-3.5 text-left',
        'shadow-[0_0_32px_-16px_var(--highlight)] hover:border-highlight hover:bg-highlight/10',
        PRESS,
      )}
    >
      <span className="grid size-9 shrink-0 place-items-center rounded-full bg-highlight text-ink">
        {done ? <Sparkles className="size-4.5" aria-hidden /> : <ArrowRight className="size-4.5" aria-hidden />}
      </span>
      <span className="min-w-0 flex-1 space-y-0.5">
        <span className="eyebrow block text-highlight">
          {copy.nextTitle} · {action.kind === 'import' ? t('nav_data') : copy.steps[action.step]}
        </span>
        {/* key: a frase nova entra com um fade quando o passo avança. */}
        <span key={action.kind} className="block text-sm font-medium text-foreground animate-in fade-in-0 slide-in-from-bottom-1 duration-300 sm:text-base">
          {text}
        </span>
      </span>
      <span className="eyebrow hidden shrink-0 items-center gap-1.5 text-foreground transition-colors group-hover:text-highlight sm:inline-flex">
        {copy.nextGo}
        <ArrowRight className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5" aria-hidden />
      </span>
    </button>
  );
}
