import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/utils';

/**
 * Movimento e luz da revisão, no mesmo sistema do resto do Simetrics: entradas com
 * `animate-in` (fade + leve deslocamento, 200–300 ms), alturas pelo `Collapse`, clique com
 * `active:scale`, e brilho da cor do estado — destaque da marca por padrão, verde para
 * incluir, vermelho para excluir.
 */

/** Transição de qualquer controle clicável: cor, borda, fundo, sombra e o "afundar" do clique. */
export const PRESS =
  'transition-[color,background-color,border-color,box-shadow,transform,opacity] duration-200 ease-out active:scale-[0.97] motion-reduce:transform-none';

/** Botões "Adicionar…": o mesmo contorno do app, com o afundar do clique. */
export const ADD_BUTTON = 'active:scale-[0.97] motion-reduce:transform-none';

/** Item que acaba de aparecer (linha adicionada, estudo aberto). */
export const ENTER = 'animate-in fade-in-0 slide-in-from-bottom-1 duration-300';

/** Painel inteiro que entra (troca de etapa). */
export const PANEL_ENTER = 'animate-in fade-in-0 slide-in-from-bottom-2 duration-300';

/** Item pequeno que sai (etiqueta removida). */
export const EXIT = 'animate-out fade-out-0 zoom-out-95 fill-mode-forwards duration-200';

export type Tone = 'highlight' | 'include' | 'exclude' | 'warning' | 'neutral';

const ACTIVE: Record<Tone, string> = {
  highlight: 'border-highlight text-highlight bg-highlight/10 shadow-[0_0_18px_-6px_var(--highlight)]',
  include: 'border-include text-include bg-include/10 shadow-[0_0_20px_-6px_var(--glow-include)]',
  exclude: 'border-exclude text-exclude bg-exclude/10 shadow-[0_0_20px_-6px_var(--glow-exclude)]',
  warning:
    'border-amber-500 text-amber-700 bg-amber-500/10 shadow-[0_0_20px_-6px_rgb(245_158_11)] dark:text-amber-300',
  neutral: 'border-foreground/60 text-foreground bg-muted shadow-[0_0_18px_-8px_var(--foreground)]',
};

const HOVER: Record<Tone, string> = {
  highlight: 'hover:border-highlight/60 hover:text-foreground',
  include: 'hover:border-include/70 hover:text-include hover:shadow-[0_0_16px_-8px_var(--glow-include)]',
  exclude: 'hover:border-exclude/70 hover:text-exclude hover:shadow-[0_0_16px_-8px_var(--glow-exclude)]',
  warning: 'hover:border-amber-500/70 hover:text-amber-700 dark:hover:text-amber-300',
  neutral: 'hover:border-foreground/50 hover:text-foreground',
};

/** Botão-etiqueta de escolha (filtros, respostas, opções): borda fina, aceso quando ativo. */
export function chipClass(active: boolean, tone: Tone = 'highlight', size: 'sm' | 'md' = 'md'): string {
  return cn(
    'rounded-md border disabled:cursor-not-allowed disabled:opacity-60',
    size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1 text-xs',
    PRESS,
    active ? ACTIVE[tone] : cn('border-border text-muted-foreground', HOVER[tone]),
  );
}

/** Botão de decisão da triagem: grande, com a luz do estado quando é a decisão atual. */
export function decisionClass(active: boolean, tone: Tone): string {
  return cn(
    'inline-flex h-9 items-center justify-center gap-2 rounded-md border px-4 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-60 [&_svg]:size-4',
    PRESS,
    active ? ACTIVE[tone] : cn('border-input text-foreground', HOVER[tone]),
  );
}

/** Bloco de conteúdo: a borda acende de leve sob o cursor, como os lançadores da visão geral. */
export const BLOCK = 'rounded-xl border border-border/80 transition-[border-color,box-shadow] duration-300 hover:border-highlight/40 hover:shadow-[0_0_32px_-18px_var(--highlight)]';

/**
 * Remoção animada: o item fecha (altura e opacidade) antes de sair do estado. Devolve
 * quem está saindo e a função que dispara a saída seguida da ação real.
 */
const COLLAPSE_MS = 300;

export function useExitRemove(): { isLeaving: (id: string) => boolean; remove: (id: string, action: () => void) => void } {
  const [leaving, setLeaving] = useState<ReadonlySet<string>>(() => new Set());
  const timers = useRef<number[]>([]);

  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach((timer) => window.clearTimeout(timer));
  }, []);

  return {
    isLeaving: (id) => leaving.has(id),
    remove(id, action) {
      setLeaving((current) => new Set(current).add(id));
      const timer = window.setTimeout(() => {
        action();
        setLeaving((current) => {
          const next = new Set(current);
          next.delete(id);
          return next;
        });
      }, COLLAPSE_MS);
      timers.current.push(timer);
    },
  };
}

export type FlashKind = 'include' | 'exclude' | 'neutral';

export function useFlash(): [{ id: number; kind: FlashKind } | null, (kind: FlashKind) => void] {
  const [flash, setFlash] = useState<{ id: number; kind: FlashKind } | null>(null);
  return [flash, (kind) => setFlash((current) => ({ id: (current?.id ?? 0) + 1, kind }))];
}
