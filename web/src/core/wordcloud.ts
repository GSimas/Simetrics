import type { Dataset } from '@/lib/types';
import {
  SIMETRICS_EXTRA_STOP_WORDS,
  WORDCLOUD_STOP_WORDS,
} from './ml/wordcloud-stop-words';

/**
 * Frequência de termos para a nuvem de palavras — ⇄ `gerar_nuvem_echarts` (utils.py:2682).
 */

/** Palavras com ao menos 3 caracteres — ⇄ `re.findall(r'\b\w{3,}\b', texto)`. */
const WORD_PATTERN = /[\p{L}\p{N}_]{3,}/gu;

/** Quantos termos a nuvem exibe. */
const DEFAULT_TOP_N = 150;

const STOP_WORDS = new Set<string>([...WORDCLOUD_STOP_WORDS, ...SIMETRICS_EXTRA_STOP_WORDS]);

export interface WordFrequency {
  text: string;
  value: number;
}

/**
 * Termos mais frequentes de uma coluna, já filtrados por stop words.
 *
 * Diferente do TF-IDF, aqui a contagem é bruta: a nuvem comunica volume, não relevância
 * relativa. É por isso que a lista de stop words importa tanto — sem ela a nuvem exibe
 * preposições.
 */
export function wordFrequencies(
  rows: Dataset,
  column: string,
  topN: number = DEFAULT_TOP_N,
): WordFrequency[] {
  const counts = new Map<string, number>();

  for (const doc of rows) {
    const value = doc[column];
    if (value === null || value === undefined) continue;

    const text = String(value).toLowerCase();
    WORD_PATTERN.lastIndex = 0;

    let match: RegExpExecArray | null;
    while ((match = WORD_PATTERN.exec(text)) !== null) {
      const word = match[0];
      if (STOP_WORDS.has(word)) continue;
      counts.set(word, (counts.get(word) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([text, value]) => ({ text, value }))
    // Desempate alfabético mantém a nuvem estável entre renderizações.
    .sort((left, right) => right.value - left.value || left.text.localeCompare(right.text))
    .slice(0, topN);
}
