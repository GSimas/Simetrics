import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { ArrowLeft, ArrowRight, CircleCheckBig, LoaderCircle, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { useTour } from '@/state/tour.store';
import { TOUR_STEPS, type TourStep } from './tour-steps';

/** Folga entre o alvo e a borda do destaque. */
const PAD = 8;
/** Distância do cartão ao alvo e às bordas da tela. */
const GAP = 14;
const MARGIN = 16;
const CARD_WIDTH = 384;
/** Quanto esperar um alvo aparecer (aba trocando, gráfico carregando) antes de desistir. */
const TARGET_TIMEOUT = 7000;

interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

type Phase = 'loading-data' | 'searching' | 'ready' | 'missing';

/** Estado atrelado ao passo que o produziu: ao trocar de passo, volta sozinho ao inicial. */
interface ForStep<T> {
  index: number;
  value: T;
}

function findTarget(step: TourStep): HTMLElement | null {
  if (!step.target) return null;
  const element = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 ? element : null;
}

/** Fixo no cabeçalho ou flutuante: não há o que rolar para ele aparecer. */
function isPinned(element: HTMLElement): boolean {
  for (let node: HTMLElement | null = element; node; node = node.parentElement) {
    const position = getComputedStyle(node).position;
    if (position === 'fixed' || position === 'sticky') return true;
  }
  return false;
}

/**
 * Rola o alvo até logo abaixo do cabeçalho fixo. Alinhar pelo topo (e não centralizar)
 * deixa o espaço de baixo livre para o cartão — os blocos ocupam a largura toda, então
 * ao lado deles não cabe nada.
 */
function scrollToTarget(element: HTMLElement): void {
  if (isPinned(element)) return;
  const header = document.querySelector('header')?.getBoundingClientRect().bottom ?? 0;
  const rect = element.getBoundingClientRect();
  const top = window.scrollY + rect.top - header - (MARGIN + PAD);
  window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
}

/** Posição do cartão: abaixo, acima, ao lado do alvo — ou encostado num canto. */
function placeCard(target: Box | null, cardHeight: number): CSSProperties {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  if (vw < 640) {
    // No celular o cartão ocupa a largura toda, na metade oposta à do alvo.
    const targetLow = target !== null && target.top + target.height / 2 > vh / 2;
    return targetLow
      ? { left: MARGIN, right: MARGIN, top: MARGIN }
      : { left: MARGIN, right: MARGIN, bottom: MARGIN };
  }

  const width = Math.min(CARD_WIDTH, vw - MARGIN * 2);
  if (!target) {
    return { width, left: (vw - width) / 2, top: Math.max(MARGIN, (vh - cardHeight) / 2) };
  }

  const clampX = (x: number): number => Math.min(Math.max(MARGIN, x), vw - width - MARGIN);
  const clampY = (y: number): number => Math.min(Math.max(MARGIN, y), vh - cardHeight - MARGIN);
  const centeredX = clampX(target.left + target.width / 2 - width / 2);
  const bottom = target.top + target.height;
  const right = target.left + target.width;

  if (bottom + GAP + cardHeight <= vh - MARGIN) return { width, left: centeredX, top: bottom + GAP };
  if (target.top - GAP - cardHeight >= MARGIN) return { width, left: centeredX, top: target.top - GAP - cardHeight };
  if (right + GAP + width <= vw - MARGIN) return { width, left: right + GAP, top: clampY(target.top) };
  if (target.left - GAP - width >= MARGIN) return { width, left: target.left - GAP - width, top: clampY(target.top) };
  // Alvo ocupa a tela: o cartão fica no canto inferior direito, por cima dele.
  return { width, right: MARGIN, bottom: MARGIN };
}

/**
 * Tour guiado: escurece a página, recorta um "holofote" sobre o bloco da vez e mostra ao
 * lado um cartão com a explicação. Troca de aba, carrega a base de exemplo e abre um
 * perfil no Motor de Busca quando o passo precisa.
 *
 * O escurecimento é a sombra do próprio holofote, que não captura cliques: a página
 * segue utilizável, e o usuário pode experimentar o que está destacado.
 */
export function GuidedTour() {
  const active = useTour((state) => state.active);
  const index = useTour((state) => state.index);
  const direction = useTour((state) => state.direction);
  const goTo = useTour((state) => state.goTo);
  const stop = useTour((state) => state.stop);
  const locale = useLocale((state) => state.locale);
  const hasData = useDataset((state) => state.active !== null);
  const isIngesting = useDataset((state) => state.isIngesting);

  const [search, setSearch] = useState<ForStep<'ready' | 'missing'> | null>(null);
  const [tracked, setTracked] = useState<ForStep<Box> | null>(null);
  const [cardHeight, setCardHeight] = useState(0);
  const cardRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLButtonElement>(null);
  const targetRef = useRef<HTMLElement | null>(null);
  const titleId = useId();

  const step = TOUR_STEPS[index] ?? TOUR_STEPS[0]!;
  const copy = locale === 'en' ? step.en : step.pt;
  const isFirst = index === 0;
  const isLast = index === TOUR_STEPS.length - 1;
  const isEn = locale === 'en';

  const waitingData = step.needsData === true && !hasData;
  const phase: Phase = waitingData
    ? 'loading-data'
    : !step.target
      ? 'ready'
      : search?.index === index
        ? search.value
        : 'searching';
  const box = tracked?.index === index ? tracked.value : null;

  // Garante base e aba, depois procura o alvo até ele aparecer.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let timer = 0;
    targetRef.current = null;

    if (step.needsData && !useDataset.getState().active) {
      if (!useDataset.getState().isIngesting) void useDataset.getState().loadDemo();
      return;
    }

    if (step.tab && useNavigation.getState().activeTab !== step.tab) {
      useNavigation.getState().setActiveTab(step.tab);
    }
    step.prepare?.();

    if (!step.target) return;

    const startedAt = Date.now();
    const poll = (): void => {
      if (cancelled) return;
      const element = findTarget(step);
      if (element) {
        targetRef.current = element;
        scrollToTarget(element);
        setSearch({ index, value: 'ready' });
        return;
      }
      if (Date.now() - startedAt > TARGET_TIMEOUT) {
        if (step.optional) {
          const next = index + direction;
          if (next >= 0 && next < TOUR_STEPS.length) goTo(next);
          else setSearch({ index, value: 'missing' });
        } else {
          setSearch({ index, value: 'missing' });
        }
        return;
      }
      timer = window.setTimeout(poll, 120);
    };
    // A primeira busca já sai do corpo do efeito — nada de setState síncrono aqui.
    timer = window.setTimeout(poll, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // `hasData` refaz o passo quando a base de exemplo termina de carregar.
  }, [active, index, step, direction, goTo, hasData]);

  // Acompanha o alvo enquanto a página rola, muda de tamanho ou anima.
  useEffect(() => {
    if (!active || phase !== 'ready') return;
    let frame = 0;
    const track = (): void => {
      const element = targetRef.current;
      if (element?.isConnected) {
        const rect = element.getBoundingClientRect();
        const next = {
          top: rect.top - PAD,
          left: rect.left - PAD,
          width: rect.width + PAD * 2,
          height: rect.height + PAD * 2,
        };
        setTracked((previous) => {
          const current = previous?.index === index ? previous.value : null;
          const same =
            current !== null &&
            Math.abs(current.top - next.top) < 0.5 &&
            Math.abs(current.left - next.left) < 0.5 &&
            Math.abs(current.width - next.width) < 0.5 &&
            Math.abs(current.height - next.height) < 0.5;
          return same ? previous : { index, value: next };
        });
      } else if (element) {
        // O bloco foi remontado (troca de dados, de aba): procura o novo.
        targetRef.current = findTarget(step);
      }
      frame = requestAnimationFrame(track);
    };
    frame = requestAnimationFrame(track);
    return () => cancelAnimationFrame(frame);
  }, [active, phase, step, index]);

  useLayoutEffect(() => {
    if (!cardRef.current) return;
    setCardHeight(cardRef.current.offsetHeight);
  }, [index, phase, locale, active]);

  useEffect(() => {
    if (active && phase !== 'loading-data') nextRef.current?.focus({ preventScroll: true });
  }, [active, index, phase]);

  useEffect(() => {
    if (!active) return;
    const onKey = (event: KeyboardEvent): void => {
      // Teclas digitadas num campo pertencem ao campo.
      const target = event.target;
      if (target instanceof Element && target.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (event.key === 'Escape') stop();
      else if (event.key === 'ArrowRight') {
        if (isLast) stop();
        else goTo(index + 1);
      } else if (event.key === 'ArrowLeft' && !isFirst) goTo(index - 1);
      else return;
      event.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, index, isFirst, isLast, goTo, stop]);

  if (!active) return null;

  const spotlight = phase === 'ready' && box !== null;
  const Icon = step.Icon;

  return createPortal(
    <div className="fixed inset-0 z-[70] pointer-events-none" data-guided-tour>
      {spotlight ? (
        <div
          aria-hidden
          className="absolute rounded-lg ring-2 ring-highlight transition-[top,left,width,height] duration-300 ease-out"
          style={{
            top: box.top,
            left: box.left,
            width: box.width,
            height: box.height,
            boxShadow: '0 0 0 9999px rgb(7 17 15 / 0.72), 0 0 32px 4px color-mix(in srgb, var(--highlight) 45%, transparent)',
          }}
        />
      ) : (
        <div aria-hidden className="absolute inset-0 bg-[rgb(7_17_15/0.72)] animate-in fade-in-0" />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="false"
        aria-labelledby={titleId}
        className="pointer-events-auto fixed flex max-h-[min(70dvh,560px)] flex-col border border-border bg-popover text-popover-foreground shadow-2xl transition-[top,left,right,bottom] duration-300 ease-out"
        style={placeCard(spotlight ? box : null, cardHeight)}
      >
        <div className="h-1 w-full bg-muted">
          <div
            className="h-full bg-highlight transition-[width] duration-300"
            style={{ width: `${((index + 1) / TOUR_STEPS.length) * 100}%` }}
          />
        </div>

        <div key={index} className="min-h-0 space-y-3 overflow-y-auto p-4 duration-300 animate-in fade-in-0 sm:p-5">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="grid size-8 shrink-0 place-items-center border border-highlight/40 text-highlight">
                <Icon className="size-4" aria-hidden />
              </span>
              <span className="eyebrow">
                Tour · {String(index + 1).padStart(2, '0')} / {TOUR_STEPS.length}
              </span>
            </div>
            <button
              type="button"
              onClick={stop}
              className="grid size-7 shrink-0 cursor-pointer place-items-center text-muted-foreground transition-colors hover:text-foreground"
              aria-label={isEn ? 'Close tour' : 'Fechar tour'}
              title={isEn ? 'Close tour (Esc)' : 'Fechar tour (Esc)'}
            >
              <X className="size-4" aria-hidden />
            </button>
          </div>

          <h2 id={titleId} className="text-base font-semibold leading-snug tracking-tight sm:text-lg">
            {copy.title}
          </h2>

          {phase === 'loading-data' || (step.needsData && isIngesting) ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <LoaderCircle className="size-4 animate-spin text-highlight" aria-hidden />
              {isEn ? 'Loading the demo dataset…' : 'Carregando a base de exemplo…'}
            </p>
          ) : (
            <>
              <p className="text-sm leading-relaxed text-muted-foreground">{copy.body}</p>
              {copy.bullets && (
                <ul className="space-y-1.5 text-sm leading-snug text-muted-foreground">
                  {copy.bullets.map((bullet) => (
                    <li key={bullet} className="flex gap-2">
                      <span className="mt-[0.45em] size-1.5 shrink-0 bg-highlight" aria-hidden />
                      <span>{bullet}</span>
                    </li>
                  ))}
                </ul>
              )}
              {phase === 'searching' && step.target && (
                <p className="eyebrow flex items-center gap-2">
                  <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
                  {isEn ? 'Finding this block…' : 'Localizando o bloco…'}
                </p>
              )}
              {phase === 'missing' && (
                <p className="eyebrow">
                  {isEn ? 'This block is not on screen right now.' : 'Este bloco não está na tela agora.'}
                </p>
              )}
            </>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-border px-4 py-3 sm:px-5">
          <button
            type="button"
            onClick={stop}
            className="eyebrow cursor-pointer transition-colors hover:text-foreground"
          >
            {isEn ? 'Skip tour' : 'Pular tour'}
          </button>
          <div className="flex items-center gap-2">
            {!isFirst && (
              <Button variant="outline" size="sm" onClick={() => goTo(index - 1)} className="gap-1.5">
                <ArrowLeft className="size-3.5" aria-hidden />
                <span className="sr-only sm:not-sr-only">{isEn ? 'Back' : 'Voltar'}</span>
              </Button>
            )}
            <Button
              ref={nextRef}
              size="sm"
              onClick={() => (isLast ? stop() : goTo(index + 1))}
              className={cn('gap-1.5', phase === 'loading-data' && 'opacity-60')}
              disabled={phase === 'loading-data'}
            >
              {isLast ? (
                <>
                  <CircleCheckBig className="size-3.5" aria-hidden />
                  {isEn ? 'Finish' : 'Concluir'}
                </>
              ) : (
                <>
                  {isFirst ? (isEn ? 'Start' : 'Começar') : isEn ? 'Next' : 'Próximo'}
                  <ArrowRight className="size-3.5" aria-hidden />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
