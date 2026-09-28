import type { SearchConcept } from './types';

/**
 * String de busca a partir dos conceitos do protocolo: termos de um conceito unidos por
 * OR, conceitos unidos por AND — e cada base com a própria sintaxe de campo. O Parsifal
 * gera uma string genérica só; aqui ela já sai pronta para colar em cada base.
 */

export interface SearchTarget {
  id: string;
  name: string;
}

export const SEARCH_TARGETS: readonly SearchTarget[] = [
  { id: 'generic', name: 'Genérica' },
  { id: 'scopus', name: 'Scopus' },
  { id: 'wos', name: 'Web of Science' },
  { id: 'pubmed', name: 'PubMed' },
  { id: 'cochrane', name: 'Cochrane Library' },
];

export const DEFAULT_SEARCH_TARGETS = ['generic', 'scopus', 'wos'];

/** Frases (com espaço ou hífen) vão entre aspas; termos simples e com curinga, não. */
function quote(term: string): string {
  const clean = term.replace(/"/g, '').trim();
  return /[\s-]/.test(clean) ? `"${clean}"` : clean;
}

function usableConcepts(concepts: readonly SearchConcept[]): string[][] {
  return concepts
    .map((concept) => concept.terms.map((term) => term.trim()).filter(Boolean))
    .filter((terms) => terms.length > 0);
}

function joinGroups(groups: string[][], format: (term: string) => string): string {
  return groups
    .map((terms) => {
      const joined = terms.map(format).join(' OR ');
      // Um grupo de um termo só dispensa parênteses — a string fica mais legível.
      return terms.length > 1 ? `(${joined})` : joined;
    })
    .join(' AND ');
}

export function buildSearchString(concepts: readonly SearchConcept[], target: string): string {
  const groups = usableConcepts(concepts);
  if (groups.length === 0) return '';
  // Dentro do campo da base, um conceito só dispensa os próprios parênteses.
  const fielded = groups.length === 1 ? groups[0]!.map(quote).join(' OR ') : joinGroups(groups, quote);

  switch (target) {
    case 'scopus':
      return `TITLE-ABS-KEY(${fielded})`;
    case 'wos':
      return `TS=(${fielded})`;
    case 'pubmed':
      // No PubMed o campo vai em cada termo; [tiab] = título e resumo.
      return joinGroups(groups, (term) => `${quote(term)}[tiab]`);
    case 'cochrane':
      return groups
        .map((terms) => `(${terms.map(quote).join(' OR ')}):ti,ab,kw`)
        .join(' AND ');
    default:
      return joinGroups(groups, quote);
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Expressão que acha os termos da busca num título ou resumo, para destacá-los na
 * triagem. O curinga `*` vira "qualquer continuação da palavra", como nas bases.
 */
export function buildHighlighter(concepts: readonly SearchConcept[]): RegExp | null {
  const patterns = usableConcepts(concepts)
    .flat()
    .map((term) => term.replace(/"/g, '').trim())
    .filter((term) => term.replace(/\*/g, '').length >= 2)
    .sort((a, b) => b.length - a.length)
    .map((term) => escapeRegExp(term).replace(/\\\*/g, '[\\p{L}\\p{N}]*').replace(/\s+/g, '\\s+'));

  if (patterns.length === 0) return null;
  return new RegExp(`(?<![\\p{L}\\p{N}])(?:${patterns.join('|')})(?![\\p{L}\\p{N}])`, 'giu');
}
