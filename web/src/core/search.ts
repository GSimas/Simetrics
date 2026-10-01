import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SearchEntityType, SimetricsDoc } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, splitTokens } from './text';

/** Ordem alfabética do português, com um único comparador para a lista inteira. */
const PT_BR_COLLATOR = new Intl.Collator('pt-BR');

/**
 * Motor de busca por entidade — ⇄ `preparar_opcoes_busca` e `filtrar_por_entidade`
 * (utils.py:722 e 759).
 */

export interface SearchOptions {
  documents: string[];
  authors: string[];
  countries: string[];
  venues: string[];
  keywords: string[];
  themes: string[];
}

/** Todas as opções selecionáveis, ordenadas, para os seletores da aba de busca. */
export function buildSearchOptions(rows: Dataset): SearchOptions {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);

  const documents = new Set<string>();
  const authors = new Set<string>();
  const countries = new Set<string>();
  const venues = new Set<string>();
  const keywords = new Set<string>();
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

    if (keywordsColumn) {
      for (const kw of splitTokens(doc[keywordsColumn])) {
        if (kw && !isNullLike(kw)) keywords.add(kw);
      }
    }

    if (columns.has(FIELD.THEME)) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme && !isNullLike(theme)) themes.add(theme);
    }
  }

  // Um Collator só: `localeCompare(x, 'pt-BR')` monta um comparador de idioma a cada
  // comparação — era a maior tarefa da thread principal ao carregar uma base. Mesma ordem.
  const sorted = (values: Set<string>): string[] => [...values].sort(PT_BR_COLLATOR.compare);

  return {
    documents: sorted(documents),
    authors: sorted(authors),
    countries: sorted(countries),
    venues: sorted(venues),
    keywords: sorted(keywords),
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
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);

  const termLower = term.toLowerCase().trim();

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
      case 'Palavra-chave':
        return keywordsColumn
          ? splitTokens(doc[keywordsColumn]).some((kw) => kw.toLowerCase().trim() === termLower)
          : false;
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
    case 'Palavra-chave':
      return options.keywords;
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
  if (options.keywords.length > 0) types.push('Palavra-chave');
  // Temas só existem depois da categorização por IA.
  if (options.themes.length > 0) types.push('Tema');
  return types;
}

export interface EntityMatch {
  type: SearchEntityType;
  term: string;
}

/** Minúsculas e sem acentos: "Análise" casa com "analise". */
function fold(text: string): string {
  return text.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
}

// Grafias dobradas por conjunto de opções: dobrar ~30 mil termos a cada tecla custaria caro.
const foldedCache = new WeakMap<SearchOptions, Map<SearchEntityType, string[]>>();

function foldedFor(options: SearchOptions, type: SearchEntityType): string[] {
  let byType = foldedCache.get(options);
  if (!byType) foldedCache.set(options, (byType = new Map()));
  let folded = byType.get(type);
  if (!folded) byType.set(type, (folded = optionsForType(options, type).map(fold)));
  return folded;
}

/**
 * Entidades cujo nome contém a consulta, em todos os tipos — as sugestões da caixa de
 * busca. Começo do nome primeiro, depois começo de palavra, depois qualquer posição; no
 * empate, o nome mais curto (o mais próximo do que foi digitado).
 */
export function matchEntities(options: SearchOptions, query: string, limit = 8): EntityMatch[] {
  const needle = fold(query.trim());
  if (needle.length < 2) return [];

  const ranked: { match: EntityMatch; rank: number; length: number }[] = [];
  for (const type of availableTypes(options)) {
    const originals = optionsForType(options, type);
    const folded = foldedFor(options, type);
    // Palavra-chave casa sem diferenciar maiúsculas (`filterByEntity`): "Memetic algorithm"
    // e "MEMETIC ALGORITHM" abrem o mesmo dossiê, então viram uma sugestão só.
    const seen = new Set<string>();
    folded.forEach((name, index) => {
      const at = name.indexOf(needle);
      if (at < 0) return;
      if (type === 'Palavra-chave') {
        const key = originals[index]!.toLowerCase().trim();
        if (seen.has(key)) return;
        seen.add(key);
      }
      const rank = at === 0 ? 0 : /[\s,.;:(\-/]/.test(name[at - 1] ?? '') ? 1 : 2;
      ranked.push({ match: { type, term: originals[index]! }, rank, length: name.length });
    });
  }
  return ranked
    .sort((left, right) => left.rank - right.rank || left.length - right.length)
    .slice(0, limit)
    .map((entry) => entry.match);
}
