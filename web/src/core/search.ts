import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SearchEntityType, SimetricsDoc } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, splitTokens } from './text';

/**
 * Motor de busca por entidade — ⇄ `preparar_opcoes_busca` e `filtrar_por_entidade`
 * (utils.py:722 e 759).
 */

export interface SearchOptions {
  documents: string[];
  authors: string[];
  countries: string[];
  venues: string[];
  themes: string[];
}

/** Todas as opções selecionáveis, ordenadas, para os seletores da aba de busca. */
export function buildSearchOptions(rows: Dataset): SearchOptions {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);

  const documents = new Set<string>();
  const authors = new Set<string>();
  const countries = new Set<string>();
  const venues = new Set<string>();
  const themes = new Set<string>();

  for (const doc of rows) {
    if (titleColumn) {
      const title = String(doc[titleColumn] ?? '').trim();
      if (title && !isNullLike(title)) documents.add(title);
    }

    if (authorsColumn) {
      for (const author of splitTokens(doc[authorsColumn])) authors.add(author);
    }

    if (columns.has(FIELD.COUNTRY)) {
      for (const country of splitTokens(doc[FIELD.COUNTRY])) countries.add(country);
    }

    if (venueColumn) {
      const venue = String(doc[venueColumn] ?? '').trim();
      if (venue && !isNullLike(venue)) venues.add(venue);
    }

    if (columns.has(FIELD.THEME)) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme && !isNullLike(theme)) themes.add(theme);
    }
  }

  const sorted = (values: Set<string>): string[] =>
    [...values].sort((left, right) => left.localeCompare(right, 'pt-BR'));

  return {
    documents: sorted(documents),
    authors: sorted(authors),
    countries: sorted(countries),
    venues: sorted(venues),
    themes: sorted(themes),
  };
}

/**
 * Documentos ligados a uma entidade.
 *
 * Documento, venue e tema casam pelo valor exato da coluna; autor e país são campos
 * multivalorados, e por isso a comparação é por token — não por substring, que casaria
 * "Silva, A." dentro de "Silva, A.B." e traria documentos alheios.
 */
export function filterByEntity(
  rows: Dataset,
  term: string,
  type: SearchEntityType,
): Dataset {
  if (!term) return [];

  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);

  const matches = (doc: SimetricsDoc): boolean => {
    switch (type) {
      case 'Documento':
        return titleColumn ? String(doc[titleColumn] ?? '').trim() === term : false;
      case 'Autor':
        return authorsColumn ? splitTokens(doc[authorsColumn]).includes(term) : false;
      case 'País':
        return columns.has(FIELD.COUNTRY) ? splitTokens(doc[FIELD.COUNTRY]).includes(term) : false;
      case 'Local de Publicação (Venue)':
        return venueColumn ? String(doc[venueColumn] ?? '').trim() === term : false;
      case 'Tema':
        return columns.has(FIELD.THEME) ? String(doc[FIELD.THEME] ?? '').trim() === term : false;
      default:
        return false;
    }
  };

  return rows.filter(matches);
}

/** Opções disponíveis para um tipo de entidade, na ordem em que a UI as apresenta. */
export function optionsForType(options: SearchOptions, type: SearchEntityType): string[] {
  switch (type) {
    case 'Documento':
      return options.documents;
    case 'Autor':
      return options.authors;
    case 'País':
      return options.countries;
    case 'Local de Publicação (Venue)':
      return options.venues;
    case 'Tema':
      return options.themes;
    default:
      return [];
  }
}

/** Tipos oferecidos no seletor, omitindo os que a base não tem como preencher. */
export function availableTypes(options: SearchOptions): SearchEntityType[] {
  const types: SearchEntityType[] = [];
  if (options.documents.length > 0) types.push('Documento');
  if (options.authors.length > 0) types.push('Autor');
  if (options.countries.length > 0) types.push('País');
  if (options.venues.length > 0) types.push('Local de Publicação (Venue)');
  // Temas só existem depois da categorização por IA.
  if (options.themes.length > 0) types.push('Tema');
  return types;
}
