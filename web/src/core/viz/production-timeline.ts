import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { docsPerYear } from '../summary';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from '../text';

/**
 * Produção ao longo do tempo, opcionalmente quebrada por categoria — o gráfico
 * "Produção ao longo do tempo" da Visão Geral.
 *
 * Mesma filosofia do boxplot (core/viz/boxplot.ts): resolve a coluna certa para a
 * dimensão pedida e conta documentos por (categoria, ano). 'Total' é o caso trivial —
 * uma série só, delegada a `docsPerYear` — as demais recortam as categorias mais
 * frequentes, porque um país ou tipo de trabalho isolado sem nenhum documento não
 * ajuda a leitura do gráfico.
 */

export type ProductionCategory = 'Total' | 'Países' | 'Temas (IA)' | 'Base de Dados' | 'Tipo de Trabalho';

export interface ProductionSeries {
  category: string;
  points: { year: number; count: number }[];
}

/** Teto de séries simultâneas — mesmo tamanho da paleta qualitativa (viz-shared.ts),
 * então nenhuma categoria fica sem cor própria. */
export const MAX_PRODUCTION_SERIES = 8;

function resolveDimensionColumn(rows: Dataset, category: ProductionCategory): string | null {
  const columns = collectColumns(rows);

  switch (category) {
    case 'Países':
      return columns.has(FIELD.COUNTRY) ? FIELD.COUNTRY : null;
    case 'Temas (IA)':
      return columns.has(FIELD.THEME) ? FIELD.THEME : null;
    case 'Base de Dados':
      return columns.has(FIELD.DATABASE) ? FIELD.DATABASE : null;
    case 'Tipo de Trabalho':
      return pickColumn(columns, FIELD_CANDIDATES.documentType);
    default:
      return null;
  }
}

/** Categorias de um documento na dimensão escolhida. Só "Países" é multivalorado
 * (autores de países diferentes no mesmo trabalho) — as demais têm um valor só. */
function categoriesOf(doc: SimetricsDoc, column: string, category: ProductionCategory): string[] {
  if (category === 'Países') {
    return splitTokens(doc[column]).filter((entity) => !isNullLike(entity));
  }

  const value = String(doc[column] ?? '').trim();
  return value && !isNullLike(value) ? [value] : [];
}

/**
 * Série(s) de documentos por ano, quebradas pela categoria escolhida.
 *
 * Com `category !== 'Total'`, mantém só as `MAX_PRODUCTION_SERIES` categorias com mais
 * documentos no total — não é paginação, é o teto que mantém o gráfico legível (e a
 * paleta de cores suficiente). Devolve `[]` quando a base não tem a coluna necessária
 * (ex.: "Temas (IA)" antes do mapeamento temático rodar).
 */
export function productionTimeline(rows: Dataset, category: ProductionCategory): ProductionSeries[] {
  if (category === 'Total') {
    return [{ category: 'Total', points: docsPerYear(rows) }];
  }

  const column = resolveDimensionColumn(rows, category);
  if (!column) return [];

  const totals = new Map<string, number>();
  for (const doc of rows) {
    for (const entity of categoriesOf(doc, column, category)) {
      totals.set(entity, (totals.get(entity) ?? 0) + 1);
    }
  }

  const top = [...totals.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, MAX_PRODUCTION_SERIES)
    .map(([entity]) => entity);

  if (top.length === 0) return [];

  const wanted = new Set(top);
  const byCategory = new Map<string, Map<number, number>>();
  for (const entity of top) byCategory.set(entity, new Map());

  for (const doc of rows) {
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (year === null) continue;
    const yearKey = Math.trunc(year);

    for (const entity of categoriesOf(doc, column, category)) {
      if (!wanted.has(entity)) continue;
      const byYear = byCategory.get(entity) as Map<number, number>;
      byYear.set(yearKey, (byYear.get(yearKey) ?? 0) + 1);
    }
  }

  // Preserva a ordem por volume (maior categoria primeiro) e descarta quem não tem
  // nenhum ano válido — pode acontecer se todo documento da categoria tiver ano ausente.
  return top
    .map((entity) => ({
      category: entity,
      points: [...(byCategory.get(entity) as Map<number, number>)]
        .sort((left, right) => left[0] - right[0])
        .map(([year, count]) => ({ year, count })),
    }))
    .filter((series) => series.points.length > 0);
}
