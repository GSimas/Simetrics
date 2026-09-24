import type { Dataset } from '@/lib/types';
import { splitTokens } from './text';

/** Quantos termos a nuvem de palavras-chave exibe. */
const DEFAULT_TOP_N = 150;

export interface WordFrequency {
  text: string;
  value: number;
}

/**
 * Palavras-chave mais frequentes, cada uma inteira: "water distribution" conta como um
 * termo, não como "water" e "distribution". As bases separam as palavras-chave por `;`.
 * Uma palavra-chave repetida no mesmo documento conta uma vez.
 */
export function keywordFrequencies(
  rows: Dataset,
  column: string,
  topN: number = DEFAULT_TOP_N,
): WordFrequency[] {
  const counts = new Map<string, number>();

  for (const doc of rows) {
    for (const keyword of new Set(splitTokens(doc[column], 'lower'))) {
      counts.set(keyword, (counts.get(keyword) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([text, value]) => ({ text, value }))
    .sort((left, right) => right.value - left.value || left.text.localeCompare(right.text))
    .slice(0, topN);
}
