import { useEffect, useState } from 'react';

/** Duração das animações de saída (`animate-out duration-200`) com folga. */
export const EXIT_MS = 220;

/**
 * Mantém um elemento montado durante a animação de saída.
 *
 * Desmontar no mesmo quadro em que `open` vira `false` corta a animação: o elemento some
 * de uma vez. Aqui ele fica em `closing` por `EXIT_MS` — o tempo de a classe `animate-out`
 * (com `fill-mode-forwards`, para não voltar a aparecer no fim) terminar.
 */
export function usePresence(open: boolean): { mounted: boolean; closing: boolean } {
  const [mounted, setMounted] = useState(open);
  // Abrir é imediato: ajuste durante a renderização, sem esperar um efeito.
  if (open && !mounted) setMounted(true);

  useEffect(() => {
    if (open || !mounted) return;
    const timer = window.setTimeout(() => setMounted(false), EXIT_MS);
    return () => window.clearTimeout(timer);
  }, [open, mounted]);

  return { mounted, closing: mounted && !open };
}
