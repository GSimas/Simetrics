import type { ReactNode } from 'react';

import { cn } from '@/lib/utils';

export interface CollapseProps {
  open: boolean;
  children: ReactNode;
  className?: string;
  /**
   * Espera 150 ms antes de abrir (padrão). Serve a indicadores de progresso; para um
   * "ver mais" clicado pelo usuário, passe `false` — a resposta deve ser imediata.
   */
  delayOpen?: boolean;
}

/**
 * Abre e fecha um trecho animando a altura (grid 0fr → 1fr), em vez de o conteúdo
 * aparecer de um quadro para o outro e empurrar a página.
 *
 * Por padrão a abertura espera 150 ms: uma operação que termina antes disso (deduplicar
 * por DOI, por exemplo) nem chega a mostrar a barra — que de outro modo só piscaria.
 */
export function Collapse({ open, children, className, delayOpen = true }: CollapseProps) {
  return (
    <div
      // Fechado, o conteúdo sai do foco e da árvore de acessibilidade.
      inert={!open}
      className={cn(
        'grid transition-[grid-template-rows,opacity,margin] duration-300 ease-out motion-reduce:transition-none',
        open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0',
        open && delayOpen && 'delay-150',
        className,
      )}
    >
      <div className="min-h-0 overflow-hidden">{children}</div>
    </div>
  );
}
