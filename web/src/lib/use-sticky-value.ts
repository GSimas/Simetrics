import { useState } from 'react';

/**
 * Devolve o último valor não nulo enquanto o atual é `null`.
 *
 * As análises derivadas voltam a `null` sempre que a base muda (deduplicação, temas) e
 * só reaparecem quando o worker termina. Mostrar o resultado anterior nesse intervalo
 * evita que KPIs e blocos sumam e voltem — o "piscar" —, e `stale` permite sinalizar
 * que os números estão sendo atualizados.
 */
export function useStickyValue<T>(value: T | null): { value: T | null; stale: boolean } {
  const [last, setLast] = useState<T | null>(value);
  // Estado derivado de renders anteriores: ajustar durante a renderização é o padrão
  // recomendado pelo React, sem o render extra de um efeito.
  if (value !== null && value !== last) setLast(value);
  return { value: value ?? last, stale: value === null && last !== null };
}
