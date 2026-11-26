import { flushSync } from 'react-dom';

/** Menos movimento, pela opção do app (`data-motion`) ou pelo sistema. */
function reducedMotion(): boolean {
  return (
    document.documentElement.dataset.motion === 'reduce' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/**
 * Troca de tela com View Transition: a tela velha sai e a nova entra, em vez de trocar
 * num corte seco. `type` escolhe a animação no CSS (`:active-view-transition-type`);
 * sem ele, o navegador faz um cruzamento simples. Sem suporte, ou com movimento
 * reduzido, só aplica a mudança.
 */
export function viewTransition(update: () => void, type?: string): void {
  if (!document.startViewTransition || reducedMotion()) {
    update();
    return;
  }
  // `flushSync`: o "depois" que a transição captura já tem o React renderizado.
  const run = (): void => flushSync(update);
  try {
    document.startViewTransition(type ? ({ update: run, types: [type] } as never) : run);
  } catch {
    // Navegador com View Transitions, mas sem o formato com `types`.
    document.startViewTransition(run);
  }
}
