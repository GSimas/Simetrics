import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from '../text';

/**
 * Distribuição estatística comparativa — ⇄ o bloco de boxplot de Geral.py:899.
 *
 * Compara a dispersão de uma métrica entre até cinco entidades. O boxplot mostra o que
 * uma média esconde: dois países podem ter a mesma média de citações e distribuições
 * completamente diferentes — um consistente, outro carregado por um único outlier.
 */

/** Como os documentos são agrupados no eixo X. */
export type BoxplotDimension = 'Países' | 'Palavras-chave' | 'Temas (IA)';

/** O que é medido no eixo Y. */
export type BoxplotMetric =
  | 'Documentos por autor'
  | 'Documentos por ano'
  | 'Citações por documento'
  | 'Citações por autor'
  | 'Citações por ano';

export interface BoxplotSeries {
  entity: string;
  values: number[];
  /** Rótulo de cada ponto, para o tooltip. */
  labels: string[];
}

/** Máximo de entidades comparáveis de uma vez — mais que isso a leitura se perde. */
export const MAX_BOXPLOT_ITEMS = 5;

/** Entidades disponíveis para comparação, ordenadas por frequência. */
export function boxplotOptions(
  rows: Dataset,
  dimension: BoxplotDimension,
  limit = 300,
): string[] {
  const column = resolveDimensionColumn(rows, dimension);
  if (!column) return [];

  const counts = new Map<string, number>();

  for (const doc of rows) {
    for (const entity of entitiesOf(doc, column, dimension)) {
      counts.set(entity, (counts.get(entity) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit)
    .map(([entity]) => entity);
}

function resolveDimensionColumn(rows: Dataset, dimension: BoxplotDimension): string | null {
  const columns = collectColumns(rows);

  switch (dimension) {
    case 'Países':
      return columns.has(FIELD.COUNTRY) ? FIELD.COUNTRY : null;
    case 'Palavras-chave':
      return pickColumn(columns, FIELD_CANDIDATES.keywords);
    case 'Temas (IA)':
      return columns.has(FIELD.THEME) ? FIELD.THEME : null;
    default:
      return null;
  }
}

/** Entidades de um documento na dimensão escolhida. */
function entitiesOf(
  doc: SimetricsDoc,
  column: string,
  dimension: BoxplotDimension,
): string[] {
  // Temas não são multivalorados: cada documento pertence a exatamente um.
  if (dimension === 'Temas (IA)') {
    const theme = String(doc[column] ?? '').trim();
    return theme && !isNullLike(theme) ? [theme] : [];
  }

  return splitTokens(doc[column], 'title').filter((entity) => !isNullLike(entity));
}

/**
 * Séries prontas para o boxplot.
 *
 * Cada métrica define uma unidade de observação diferente — por autor, por ano ou por
 * documento — e é isso que muda o que a caixa representa. "Citações por autor" mede a
 * desigualdade entre pesquisadores; "citações por documento", entre trabalhos.
 */
export function boxplotSeries(
  rows: Dataset,
  dimension: BoxplotDimension,
  metric: BoxplotMetric,
  selected: readonly string[],
): BoxplotSeries[] {
  const column = resolveDimensionColumn(rows, dimension);
  if (!column || selected.length === 0) return [];

  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const wanted = new Set(selected);

  const series = new Map<string, BoxplotSeries>();
  for (const entity of selected) series.set(entity, { entity, values: [], labels: [] });

  // Acumuladores por (entidade, unidade de observação), para as métricas agregadas.
  const byAuthor = new Map<string, Map<string, number>>();
  const byYear = new Map<string, Map<number, number>>();

  for (const doc of rows) {
    const entities = entitiesOf(doc, column, dimension).filter((entity) => wanted.has(entity));
    if (entities.length === 0) continue;

    const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);

    for (const entity of entities) {
      if (metric === 'Citações por documento') {
        const bucket = series.get(entity);
        if (!bucket) continue;
        bucket.values.push(citations);
        bucket.labels.push(titleColumn ? String(doc[titleColumn] ?? '') : '');
        continue;
      }

      if (metric === 'Documentos por autor' || metric === 'Citações por autor') {
        if (!authorsColumn) continue;
        let authors = byAuthor.get(entity);
        if (!authors) byAuthor.set(entity, (authors = new Map()));

        for (const author of splitTokens(doc[authorsColumn])) {
          const increment = metric === 'Documentos por autor' ? 1 : citations;
          authors.set(author, (authors.get(author) ?? 0) + increment);
        }
        continue;
      }

      if (year === null) continue;
      let years = byYear.get(entity);
      if (!years) byYear.set(entity, (years = new Map()));

      const increment = metric === 'Documentos por ano' ? 1 : citations;
      const key = Math.trunc(year);
      years.set(key, (years.get(key) ?? 0) + increment);
    }
  }

  if (metric === 'Documentos por autor' || metric === 'Citações por autor') {
    for (const [entity, authors] of byAuthor) {
      const bucket = series.get(entity);
      if (!bucket) continue;
      for (const [author, value] of authors) {
        bucket.values.push(value);
        bucket.labels.push(author);
      }
    }
  } else if (metric === 'Documentos por ano' || metric === 'Citações por ano') {
    for (const [entity, years] of byYear) {
      const bucket = series.get(entity);
      if (!bucket) continue;
      for (const [year, value] of [...years].sort((left, right) => left[0] - right[0])) {
        bucket.values.push(value);
        bucket.labels.push(String(year));
      }
    }
  }

  // Preserva a ordem em que o usuário selecionou, e descarta séries sem observação.
  return selected
    .map((entity) => series.get(entity))
    .filter((entry): entry is BoxplotSeries => entry !== undefined && entry.values.length > 0);
}
