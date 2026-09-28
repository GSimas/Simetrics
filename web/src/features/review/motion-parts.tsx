import type { ReactNode } from 'react';

import { Collapse } from '@/components/Collapse';
import { cn } from '@/lib/utils';
import { ENTER, type FlashKind } from './motion';

/** Linha de lista que entra deslizando e sai recolhendo a altura. */
export function AnimatedItem({ leaving, children, className }: { leaving: boolean; children: ReactNode; className?: string }) {
  return (
    <Collapse open={!leaving} delayOpen={false}>
      {/* O respiro interno evita que o recorte do Collapse corte anéis de foco e brilhos. */}
      <div className={cn(ENTER, 'p-0.5', className)}>{children}</div>
    </Collapse>
  );
}

/**
 * Lampejo de luz sobre um cartão a cada decisão. O `id` muda a cada disparo, e a chave
 * nova reinicia a animação mesmo quando a mesma decisão se repete.
 */
export function DecisionFlash({ flash }: { flash: { id: number; kind: FlashKind } | null }) {
  if (!flash) return null;
  return (
    <span
      key={flash.id}
      aria-hidden
      className={cn(
        'decision-flash',
        flash.kind === 'include' && 'decision-flash--include',
        flash.kind === 'exclude' && 'decision-flash--exclude',
      )}
    />
  );
}
