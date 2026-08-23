import { FIELD, FIELD_CANDIDATES, currentYear } from '@/lib/schema';
import type {
  BibliometrixMetrics,
  Dataset,
  DatasetSummary,
  MetadataCompleteness,
} from '@/lib/types';
import { mean, pyRound } from './stats';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from './text';

/** Anos válidos do dataset, já truncados para inteiro. */
function validYears(rows: Dataset): number[] {
  const years: number[] = [];
  for (const doc of rows) {
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (year !== null) years.push(Math.trunc(year));
  }
  return years;
}

/**
 * Métricas do relatório "Main Information" do Bibliometrix — ⇄
 * `calcular_metricas_bibliometrix` (utils.py:2404).
 */
export function bibliometrixMetrics(
  rows: Dataset,
  baseYear = currentYear(),
): BibliometrixMetrics {
  const columns = collectColumns(rows);
  const years = validYears(rows);

  // Taxa de crescimento anual composta entre o primeiro e o último ano com publicação.
  let growthRate = 0;
  const distinctYears = new Set(years);
  if (distinctYears.size > 1) {
    const firstYear = Math.min(...years);
    const lastYear = Math.max(...years);

    let docsFirstYear = 0;
    let docsLastYear = 0;
    for (const year of years) {
      if (year === firstYear) docsFirstYear += 1;
      if (year === lastYear) docsLastYear += 1;
    }

    const span = lastYear - firstYear;
    if (docsFirstYear > 0 && span > 0) {
      growthRate = ((docsLastYear / docsFirstYear) ** (1 / span) - 1) * 100;
    }
  }

  // Citações por ano de vida do documento; documentos do ano corrente contam 1 ano.
  let avgCitPerYear = 0;
  if (columns.has(FIELD.TOTAL_CITATIONS) && columns.has(FIELD.YEAR_CLEAN)) {
    const perYear: number[] = [];
    for (const doc of rows) {
      const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
      if (year === null) continue;

      const age = Math.max(baseYear - year + 1, 1);
      const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
      const ratio = citations / age;
      if (Number.isFinite(ratio)) perYear.push(ratio);
    }
    if (perYear.length > 0) avgCitPerYear = pyRound(mean(perYear), 2);
  }

  // MCP: documentos com colaboração internacional (mais de um país distinto).
  let mcp = 0;
  if (columns.has(FIELD.COUNTRY)) {
    for (const doc of rows) {
      if (new Set(splitTokens(doc[FIELD.COUNTRY])).size > 1) mcp += 1;
    }
  }

  let coauthIndex = 0;
  let singleAuthorDocs = 0;
  if (columns.has(FIELD.AUTHORS)) {
    const counts: number[] = [];
    for (const doc of rows) {
      const count = splitTokens(doc[FIELD.AUTHORS]).length;
      // O Python descarta documentos sem autor antes de calcular a média.
      if (count > 0) counts.push(count);
      if (count === 1) singleAuthorDocs += 1;
    }
    if (counts.length > 0) coauthIndex = pyRound(mean(counts), 2);
  }

  return {
    growthRate: pyRound(growthRate, 2),
    mcp,
    scp: rows.length - mcp,
    coauthIndex,
    singleAuthorDocs,
    avgCitPerYear,
  };
}

/** Campos verificados no painel de qualidade — ⇄ `campos_verificacao` (utils.py:608). */
const COMPLETENESS_FIELDS: readonly (readonly [string, string])[] = [
  [FIELD.AUTHORS, 'Author (AU)'],
  [FIELD.DOCUMENT_TYPE, 'Document Type (DT)'],
  [FIELD.ABSTRACT, 'Abstract (AB)'],
  [FIELD.COUNTRY, 'Affiliation/Country (C1)'],
  [FIELD.DOI, 'DOI (DI)'],
  [FIELD.TITLE, 'Title (TI)'],
  [FIELD.SECONDARY_TITLE, 'Journal/Source (SO)'],
  [FIELD.YEAR_CLEAN, 'Publication Year (PY)'],
  [FIELD.TOTAL_CITATIONS, 'Total Citation (TC)'],
  [FIELD.KEYWORDS, 'Keywords (DE/ID)'],
  [FIELD.REFERENCES_UNIFIED, 'Cited References (CR)'],
];

function completenessStatus(missingPct: number): MetadataCompleteness['status'] {
  if (missingPct === 0) return 'Excelente';
  if (missingPct <= 10) return 'Bom';
  if (missingPct <= 20) return 'Aceitável';
  return 'Ruim';
}

/** ⇄ `analisar_completude_metadados` (utils.py:607). */
export function metadataCompleteness(rows: Dataset): MetadataCompleteness[] {
  const columns = collectColumns(rows);
  const total = rows.length;

  const report = COMPLETENESS_FIELDS.map(([column, label]) => {
    let missing = total;

    if (columns.has(column)) {
      missing = 0;
      for (const doc of rows) {
        const value = doc[column];
        // Ausente = nulo, NaN, ou texto vazio depois do trim.
        if (value === null || value === undefined) missing += 1;
        else if (typeof value === 'number') {
          if (Number.isNaN(value)) missing += 1;
        } else if (String(value).trim() === '') missing += 1;
      }
    }

    const missingPct = total > 0 ? (missing / total) * 100 : 0;
    return { field: label, missing, missingPct, status: completenessStatus(missingPct) };
  });

  return report.sort((left, right) => left.missingPct - right.missingPct);
}

/** ⇄ `resumir_base_bibliometrica` (utils.py:663). */
export function summarize(rows: Dataset, baseYear = currentYear()): DatasetSummary {
  const columns = collectColumns(rows);
  const years = validYears(rows);

  let timespan = 'N/S';
  let avgAge: number | null = null;
  if (years.length > 0) {
    timespan = `${Math.min(...years)}:${Math.max(...years)}`;
    avgAge = pyRound(baseYear - mean(years), 2);
  }

  const distinct = (column: string, transform?: 'lower'): number => {
    if (!columns.has(column)) return 0;
    const values = new Set<string>();
    for (const doc of rows) {
      for (const token of splitTokens(doc[column], transform)) values.add(token);
    }
    return values.size;
  };

  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);

  // Venues não são multivaloradas: a coluna inteira é a entidade.
  //
  // A normalização para MAIÚSCULAS antes de contar é uma divergência deliberada em
  // relação ao Python, pelo mesmo motivo da tabela de venues: `nunique()` sobre o valor
  // bruto trata "IEEE Transactions on..." e "IEEE TRANSACTIONS ON..." como dois
  // periódicos distintos, o que é a regra ao misturar exports de bases diferentes.
  // Sem isso, o indicador anunciaria 719 venues acima de uma tabela com 589 linhas.
  let venuesCount = 0;
  if (venueColumn) {
    const venues = new Set<string>();
    for (const doc of rows) {
      const venue = String(doc[venueColumn] ?? '').trim();
      if (venue && !isNullLike(venue)) venues.add(venue.toUpperCase());
    }
    venuesCount = venues.size;
  }

  return {
    totalDocs: rows.length,
    timespan,
    avgAge,
    authorsCount: distinct(FIELD.AUTHORS),
    countriesCount: distinct(FIELD.COUNTRY),
    keywordsCount: keywordsColumn ? distinct(keywordsColumn, 'lower') : 0,
    venuesCount,
    bibliometrix: bibliometrixMetrics(rows, baseYear),
  };
}

/** Documentos por ano, crescente — base dos gráficos de produção temporal. */
export function docsPerYear(rows: Dataset): { year: number; count: number }[] {
  const counts = new Map<number, number>();
  for (const year of validYears(rows)) {
    counts.set(year, (counts.get(year) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([year, count]) => ({ year, count }))
    .sort((left, right) => left.year - right.year);
}

/** Documentos por autor, para a Lei de Lotka. */
export function docsPerAuthor(rows: Dataset): number[] {
  const counts = new Map<string, number>();
  for (const doc of rows) {
    for (const author of splitTokens(doc[FIELD.AUTHORS])) {
      counts.set(author, (counts.get(author) ?? 0) + 1);
    }
  }
  return [...counts.values()];
}
