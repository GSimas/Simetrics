import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { collectColumns, pickColumn, splitTokens, toNumeric } from '../text';

/**
 * Genética das ideias — ⇄ `calcular_genetica_palavras` (utils.py:341).
 *
 * Trata cada palavra-chave como um organismo e mede seu ciclo de vida: quando nasceu
 * (primeira aparição), quando foi vista pela última vez, quantas vezes se replicou e
 * quanto impacto acumulou. Termos com longevidade alta e replicação alta são o núcleo
 * estável da área; termos recentes com replicação rápida são fronteiras emergentes.
 */

export interface KeywordGenetics {
  keyword: string;
  /** Ano da primeira aparição. */
  birthYear: number;
  /** Ano da última aparição. */
  lastYear: number;
  /** Anos entre a primeira e a última aparição. */
  lifespan: number;
  /** Quantas vezes o termo aparece na base. */
  occurrences: number;
  /** Citações acumuladas pelos documentos que usam o termo. */
  citations: number;
}

export function keywordGenetics(rows: Dataset): KeywordGenetics[] {
  const columns = collectColumns(rows);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  if (!keywordsColumn || !columns.has(FIELD.YEAR_CLEAN)) return [];

  interface Accumulator {
    birthYear: number;
    lastYear: number;
    occurrences: number;
    citations: number;
  }

  const byKeyword = new Map<string, Accumulator>();

  for (const doc of rows) {
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (year === null) continue;

    const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;

    // Minúsculas, como o `.str.lower()` do Python: "Memetics" e "memetics" são o mesmo termo.
    for (const keyword of splitTokens(doc[keywordsColumn], 'lower')) {
      const current = byKeyword.get(keyword);

      if (current) {
        current.birthYear = Math.min(current.birthYear, year);
        current.lastYear = Math.max(current.lastYear, year);
        current.occurrences += 1;
        current.citations += citations;
      } else {
        byKeyword.set(keyword, {
          birthYear: year,
          lastYear: year,
          occurrences: 1,
          citations,
        });
      }
    }
  }

  return [...byKeyword.entries()]
    .map(([keyword, data]) => ({
      keyword,
      birthYear: data.birthYear,
      lastYear: data.lastYear,
      lifespan: data.lastYear - data.birthYear,
      occurrences: data.occurrences,
      citations: data.citations,
    }))
    .sort((left, right) => right.occurrences - left.occurrences);
}
