import { FIELD, FIELD_CANDIDATES, currentYear } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { computeIndices } from './scientometrics';
import { mean, median, pyRound, std, sum } from './stats';
import { collectColumns, isNullLike, pickColumn, splitTokens, titleCase, toNumeric } from './text';

/**
 * Geradores das tabelas analíticas — ⇄ `gerar_tabela_{autores,paises,venues,keywords}`
 * (utils.py:862-1096).
 *
 * As quatro tabelas do Python repetem quase o mesmo corpo. Aqui a estrutura comum é
 * fatorada num único motor de agregação, e cada tabela só descreve como extrai a
 * entidade e quais colunas específicas monta.
 */

/** Linha comum a todas as tabelas de entidade. */
export interface EntityRow {
  entity: string;
  docCount: number;
  citations: number;
  h: number;
  g: number;
  i10: number;
  m: number;
  meanCitations: number;
  medianCitations: number;
  stdCitations: number;
  /** Especialização por Quociente Locacional, quando há temas categorizados. */
  topSpecialization: string;
  /** "2024: Título (12 citações) | 2023: ..." */
  timeline: string;
  /** Documento mais citado do grupo. */
  topDocument: string;
  authors: string[];
  countries: string[];
  coauthors: string[];
  documents: string[];
}

interface Group {
  entity: string;
  docs: SimetricsDoc[];
}

/** Agrupa por entidade explodindo campos multivalorados separados por ';'. */
function groupByExploded(
  rows: Dataset,
  column: string,
  transform: 'title' | 'upper' | 'none',
): Group[] {
  const groups = new Map<string, SimetricsDoc[]>();

  for (const doc of rows) {
    const raw = doc[column];
    const tokens = transform === 'none' ? [String(raw ?? '').trim()] : splitTokens(raw);

    for (const token of tokens) {
      const entity =
        transform === 'title'
          ? titleCase(token)
          : transform === 'upper'
            ? token.toUpperCase()
            : token;

      // O Python filtra vazios e o literal que o pandas produz ao converter nulos.
      if (!entity || isNullLike(entity)) continue;

      let bucket = groups.get(entity);
      if (!bucket) groups.set(entity, (bucket = []));
      bucket.push(doc);
    }
  }

  return [...groups.entries()].map(([entity, docs]) => ({ entity, docs }));
}

/** Citações numéricas do grupo, com ausências em 0 — ⇄ `.fillna(0)`. */
function citationsOf(docs: readonly SimetricsDoc[]): number[] {
  return docs.map((doc) => toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0);
}

/**
 * ⇄ `_format_timeline` (utils.py:819): agrupa por ano decrescente, deduplicando por
 * (ano, título) para não repetir o mesmo documento dentro do grupo.
 */
function formatTimeline(docs: readonly SimetricsDoc[], titleColumn: string | null): string {
  const byYear = new Map<string, string[]>();
  const seen = new Set<string>();

  for (const doc of docs) {
    const yearValue = toNumeric(doc[FIELD.YEAR_CLEAN]);
    const year = yearValue === null ? 'S/D' : String(Math.trunc(yearValue));
    const title = titleColumn ? String(doc[titleColumn] ?? '').trim() : 'Sem título';

    const key = `${year} ${title}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const citations = Math.trunc(toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0);
    let bucket = byYear.get(year);
    if (!bucket) byYear.set(year, (bucket = []));
    bucket.push(`${title} (${citations} citações)`);
  }

  return [...byYear.keys()]
    .sort((a, b) => (a < b ? 1 : a > b ? -1 : 0))
    .map((year) => `${year}: ${(byYear.get(year) as string[]).join('; ')}`)
    .join(' | ');
}

/** ⇄ `_get_top_doc` (utils.py:844): documento de maior citação, primeiro em caso de empate. */
function topDocument(docs: readonly SimetricsDoc[], titleColumn: string | null): string {
  if (docs.length === 0) return '';

  let best = 0;
  let bestCitations = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < docs.length; i += 1) {
    const citations = toNumeric((docs[i] as SimetricsDoc)[FIELD.TOTAL_CITATIONS]) ?? 0;
    if (citations > bestCitations) {
      bestCitations = citations;
      best = i;
    }
  }

  const doc = docs[best] as SimetricsDoc;
  const title = titleColumn ? String(doc[titleColumn] ?? '').trim() : 'Sem Título';
  return `${title} (${Math.trunc(bestCitations)} citações)`;
}

/**
 * Contexto do Quociente Locacional, calculado uma vez por dataset.
 *
 * QL = (Qik / Qk) / (Qi / Q), onde Qik são os documentos da entidade k no tema i,
 * Qk o total da entidade, Qi o total do tema e Q o total geral. Acima de 1 indica
 * especialização: a entidade publica naquele tema mais do que a media da base.
 */
interface QuotientContext {
  total: number;
  docsPerTheme: Map<string, number>;
}

function buildQuotientContext(rows: Dataset): QuotientContext | null {
  const columns = collectColumns(rows);
  if (!columns.has(FIELD.THEME)) return null;

  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const docsPerTheme = new Map<string, number>();
  const seenTitles = new Set<string>();

  for (const doc of rows) {
    // O Python conta temas sobre documentos únicos por título.
    if (titleColumn) {
      const title = String(doc[titleColumn] ?? '');
      if (seenTitles.has(title)) continue;
      seenTitles.add(title);
    }

    const theme = String(doc[FIELD.THEME] ?? '');
    if (isNullLike(theme)) continue;
    docsPerTheme.set(theme, (docsPerTheme.get(theme) ?? 0) + 1);
  }

  const total = titleColumn ? seenTitles.size : rows.length;
  return total > 0 ? { total, docsPerTheme } : null;
}

/** Tema de maior QL do grupo, formatado como "Tema (QL: 1.83)". */
function topSpecialization(
  docs: readonly SimetricsDoc[],
  context: QuotientContext | null,
): string {
  if (!context) return 'Não Categorizado';

  const groupSize = docs.length;
  if (groupSize === 0) return 'Não Categorizado';

  const docsPerThemeInGroup = new Map<string, number>();
  for (const doc of docs) {
    const theme = String(doc[FIELD.THEME] ?? '');
    if (isNullLike(theme)) continue;
    docsPerThemeInGroup.set(theme, (docsPerThemeInGroup.get(theme) ?? 0) + 1);
  }

  let bestQuotient = -1;
  let bestTheme = '';

  for (const [theme, countInGroup] of docsPerThemeInGroup) {
    const themeTotal = context.docsPerTheme.get(theme) ?? 0;
    if (themeTotal <= 0) continue;

    const quotient = countInGroup / groupSize / (themeTotal / context.total);
    if (quotient > bestQuotient) {
      bestQuotient = quotient;
      bestTheme = theme;
    }
  }

  if (bestQuotient < 0) return 'Não Categorizado';
  return `${bestTheme} (QL: ${bestQuotient.toFixed(2)})`;
}

/** Constrói a linha comum a partir de um grupo já formado. */
function buildRow(
  group: Group,
  titleColumn: string | null,
  context: QuotientContext | null,
  baseYear: number,
): EntityRow {
  const citations = citationsOf(group.docs);
  const years = group.docs.map((doc) => doc[FIELD.YEAR_CLEAN]);
  const indices = computeIndices(citations, years, baseYear);

  const authors = new Set<string>();
  const countries = new Set<string>();
  for (const doc of group.docs) {
    for (const author of splitTokens(doc[FIELD.AUTHORS], 'title')) authors.add(author);
    for (const country of splitTokens(doc[FIELD.COUNTRY], 'title')) countries.add(country);
  }

  const coauthors = new Set(authors);
  coauthors.delete(group.entity);

  return {
    entity: group.entity,
    docCount: group.docs.length,
    citations: sum(citations),
    ...indices,
    meanCitations: pyRound(mean(citations), 2),
    medianCitations: pyRound(median(citations), 2),
    // O Python zera o desvio para grupos de um único documento, onde o std amostral é NaN.
    stdCitations: group.docs.length > 1 ? pyRound(std(citations), 2) : 0,
    topSpecialization: topSpecialization(group.docs, context),
    timeline: formatTimeline(group.docs, titleColumn),
    topDocument: topDocument(group.docs, titleColumn),
    authors: [...authors],
    countries: [...countries],
    coauthors: [...coauthors],
    documents: titleColumn
      ? group.docs.map((doc) => String(doc[titleColumn] ?? '')).filter(Boolean)
      : [],
  };
}

/** Ordenação padrão das tabelas: índice h, depois total de citações. */
function sortRows(rows: EntityRow[]): EntityRow[] {
  return rows.sort((left, right) => right.h - left.h || right.citations - left.citations);
}

function buildTable(
  rows: Dataset,
  column: string | null,
  transform: 'title' | 'upper' | 'none',
  baseYear: number,
): EntityRow[] {
  if (!column) return [];

  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const context = buildQuotientContext(rows);
  const groups = groupByExploded(rows, column, transform);

  return sortRows(groups.map((group) => buildRow(group, titleColumn, context, baseYear)));
}

export function authorsTable(rows: Dataset, baseYear = currentYear()): EntityRow[] {
  const column = pickColumn(collectColumns(rows), FIELD_CANDIDATES.authors);
  return buildTable(rows, column, 'title', baseYear);
}

export function countriesTable(rows: Dataset, baseYear = currentYear()): EntityRow[] {
  const columns = collectColumns(rows);
  return buildTable(rows, columns.has(FIELD.COUNTRY) ? FIELD.COUNTRY : null, 'title', baseYear);
}

/** Venues não são multivaloradas: o nome inteiro é a entidade, em MAIÚSCULAS. */
export function venuesTable(rows: Dataset, baseYear = currentYear()): EntityRow[] {
  const column = pickColumn(collectColumns(rows), FIELD_CANDIDATES.venue);
  return buildTable(rows, column, 'upper', baseYear);
}

export function keywordsTable(rows: Dataset, baseYear = currentYear()): EntityRow[] {
  const column = pickColumn(collectColumns(rows), FIELD_CANDIDATES.keywords);
  return buildTable(rows, column, 'title', baseYear);
}
