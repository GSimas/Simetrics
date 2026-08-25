import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SearchEntityType, SimilarityHit } from '@/lib/types';
import { pyRound } from './stats';
import { collectColumns, pickColumn, splitTokens } from './text';

/**
 * Similaridade de Jaccard entre entidades — ⇄ `calcular_similares_biblio` (utils.py:2588).
 *
 * A ideia é tratar cada entidade como um "DNA acadêmico": o conjunto de traços que a
 * caracteriza. Duas entidades são parecidas na medida em que compartilham traços.
 *
 *     J(A, B) = |A ∩ B| / |A ∪ B|
 *
 * O que entra no perfil muda conforme o tipo, e a escolha não é arbitrária — o perfil de
 * um autor inclui seus coautores mas não ele mesmo, senão todo autor seria similar a si.
 */

/** Quantos resultados devolver, e o teto de traços exibidos por resultado. */
const MAX_RESULTS = 15;
const MAX_SHARED_TRAITS = 4;

interface Profiles {
  documents: Map<string, Set<string>>;
  authors: Map<string, Set<string>>;
  countries: Map<string, Set<string>>;
  venues: Map<string, Set<string>>;
  keywords: Map<string, Set<string>>;
}

function addAll(target: Map<string, Set<string>>, key: string, values: Iterable<string>): void {
  let bucket = target.get(key);
  if (!bucket) target.set(key, (bucket = new Set()));
  for (const value of values) bucket.add(value);
}

/** Monta os perfis de todas as entidades numa única passagem pela base. */
export function buildProfiles(rows: Dataset): Profiles {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);

  const profiles: Profiles = {
    documents: new Map(),
    authors: new Map(),
    countries: new Map(),
    venues: new Map(),
    keywords: new Map(),
  };

  for (const doc of rows) {
    const title = titleColumn ? String(doc[titleColumn] ?? '').trim() : '';
    const authors = new Set(authorsColumn ? splitTokens(doc[authorsColumn]) : []);
    const rawKeywords = keywordsColumn ? splitTokens(doc[keywordsColumn]) : [];
    // Palavras-chave entram em minúsculas para match de traço: "Memetics" e "memetics" são o mesmo traço.
    const keywords = new Set(keywordsColumn ? splitTokens(doc[keywordsColumn], 'lower') : []);
    const countries = new Set(columns.has(FIELD.COUNTRY) ? splitTokens(doc[FIELD.COUNTRY]) : []);
    const venue = venueColumn ? String(doc[venueColumn] ?? '').trim() : '';
    const venueSet = venue ? [venue] : [];

    // Documento: o que ele trata, quem escreveu e onde saiu.
    if (title) {
      addAll(profiles.documents, title, [...keywords, ...authors, ...venueSet]);
    }

    for (const author of authors) {
      // Coautores entram no perfil, o próprio autor não.
      const coauthors = [...authors].filter((other) => other !== author);
      addAll(profiles.authors, author, [...keywords, ...venueSet, ...coauthors]);
    }

    for (const country of countries) {
      addAll(profiles.countries, country, [...keywords, ...authors, ...venueSet]);
    }

    if (venue) {
      addAll(profiles.venues, venue, [...keywords, ...authors, ...countries]);
    }

    for (const kw of rawKeywords) {
      const otherKws = [...keywords].filter((other) => other !== kw.toLowerCase());
      addAll(profiles.keywords, kw, [...otherKws, ...authors, ...venueSet]);
    }
  }

  return profiles;
}

function profilesFor(profiles: Profiles, type: SearchEntityType): Map<string, Set<string>> | null {
  switch (type) {
    case 'Documento':
      return profiles.documents;
    case 'Autor':
      return profiles.authors;
    case 'País':
      return profiles.countries;
    case 'Local de Publicação (Venue)':
      return profiles.venues;
    case 'Palavra-chave':
      return profiles.keywords;
    default:
      // Temas não têm perfil próprio: são um rótulo atribuído pela IA, não uma entidade
      // com traços observáveis na base.
      return null;
  }
}

/**
 * Entidades mais parecidas com `term`, do mesmo tipo, em ordem decrescente.
 *
 * @param profiles Perfis pré-computados. Reutilizá-los entre buscas evita varrer a base
 *   inteira a cada consulta do usuário.
 */
export function findSimilar(
  profiles: Profiles,
  term: string,
  type: SearchEntityType,
): SimilarityHit[] {
  const pool = profilesFor(profiles, type);
  if (!pool || !term) return [];

  const target = pool.get(term);
  if (!target || target.size === 0) return [];

  const results: SimilarityHit[] = [];

  for (const [candidate, profile] of pool) {
    if (candidate === term) continue;

    const shared: string[] = [];
    for (const trait of target) {
      if (profile.has(trait)) shared.push(trait);
    }
    if (shared.length === 0) continue;

    // |A ∪ B| = |A| + |B| − |A ∩ B|, sem materializar a união.
    const union = target.size + profile.size - shared.length;
    if (union === 0) continue;

    results.push({
      item: candidate,
      similarity: pyRound((shared.length / union) * 100, 1),
      sharedTraits: shared.sort().slice(0, MAX_SHARED_TRAITS).join(' | '),
    });
  }

  results.sort((left, right) => right.similarity - left.similarity);
  return results.slice(0, MAX_RESULTS);
}
