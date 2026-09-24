import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, toNumeric } from '../text';

/**
 * Visão enxuta de um documento para a classificação híbrida: só título, palavras-chave,
 * resumo e ano. A documentação do Jev é explícita que a acurácia cai com contexto
 * irrelevante, então afiliações, referências e demais campos ficam de fora.
 */
export interface HybridDoc {
  /** Posição do documento na base ativa. */
  index: number;
  title: string;
  keywords: string;
  abstract: string;
  year: number | null;
}

function text(value: unknown): string {
  return isNullLike(value) ? '' : String(value).trim();
}

export function toHybridDocs(rows: Dataset): HybridDoc[] {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  const abstractColumn = pickColumn(columns, FIELD_CANDIDATES.abstract);
  const yearColumn = pickColumn(columns, FIELD_CANDIDATES.year);

  return rows.map((doc, index) => {
    const year = yearColumn ? toNumeric(doc[yearColumn]) : null;
    return {
      index,
      title: titleColumn ? text(doc[titleColumn]) : '',
      keywords: keywordsColumn ? text(doc[keywordsColumn]) : '',
      abstract: abstractColumn ? text(doc[abstractColumn]) : text(doc[FIELD.ABSTRACT]),
      year: year === null ? null : Math.trunc(year),
    };
  });
}

/** Corta um texto num limite de caracteres sem partir a última palavra ao meio. */
export function truncate(value: string, limit: number): string {
  if (value.length <= limit) return value;
  const cut = value.slice(0, limit);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}
