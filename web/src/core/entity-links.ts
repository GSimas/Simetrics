import type { SearchEntityType } from '@/lib/types';
import { availableTypes, optionsForType, type SearchOptions } from './search';

/**
 * Encontra, num texto livre (a resposta da Simi), os nomes que existem na base — autores,
 * países, venues, palavras-chave, temas e títulos — para virarem links ao dossiê.
 *
 * Compara sequências de palavras, sem caixa, acentos nem pontuação: "Price, I." no texto
 * casa com "PRICE I" na base. Entre candidatos que começam na mesma palavra vence o mais
 * longo, então um título inteiro não vira três palavras-chave soltas.
 */

export interface EntityMention {
  start: number;
  end: number;
  type: SearchEntityType;
  term: string;
}

export interface EntityIndex {
  /** Sequência de palavras normalizadas → entidade. */
  byKey: Map<string, { type: SearchEntityType; term: string }>;
  /** Primeira palavra → comprimentos (em palavras) dos nomes que começam por ela, do maior ao menor. */
  lengthsByFirst: Map<string, number[]>;
}

const WORD = /[\p{L}\p{N}]+/gu;

function fold(word: string): string {
  return word.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
}

function words(text: string): { word: string; start: number; end: number }[] {
  const found: { word: string; start: number; end: number }[] = [];
  for (const match of text.matchAll(WORD)) {
    found.push({ word: fold(match[0]), start: match.index, end: match.index + match[0].length });
  }
  return found;
}

/**
 * Na colisão entre tipos, o mais específico fica: um título é um documento, e um nome
 * que é autor e palavra-chave é mais provavelmente o autor.
 */
const PRIORITY: SearchEntityType[] = ['Documento', 'Autor', 'Local de Publicação (Venue)', 'País', 'Tema', 'Palavra-chave'];

export function buildEntityIndex(options: SearchOptions): EntityIndex {
  const byKey = new Map<string, { type: SearchEntityType; term: string }>();
  const lengths = new Map<string, Set<number>>();
  const types = availableTypes(options).sort((a, b) => PRIORITY.indexOf(a) - PRIORITY.indexOf(b));

  for (const type of types) {
    for (const term of optionsForType(options, type)) {
      const parts = words(term).map((part) => part.word);
      const key = parts.join(' ');
      // Curto demais ou só números (anos, volumes): ligaria o texto inteiro por acaso.
      if (key.length < 3 || /^[\d ]+$/.test(key) || byKey.has(key)) continue;
      byKey.set(key, { type, term });
      const first = parts[0]!;
      if (!lengths.has(first)) lengths.set(first, new Set());
      lengths.get(first)!.add(parts.length);
    }
  }

  const lengthsByFirst = new Map<string, number[]>();
  for (const [first, set] of lengths) lengthsByFirst.set(first, [...set].sort((a, b) => b - a));
  return { byKey, lengthsByFirst };
}

export function findEntityMentions(text: string, index: EntityIndex): EntityMention[] {
  const tokens = words(text);
  const mentions: EntityMention[] = [];
  let i = 0;
  while (i < tokens.length) {
    let matched = 0;
    for (const length of index.lengthsByFirst.get(tokens[i]!.word) ?? []) {
      if (i + length > tokens.length) continue;
      const key = tokens
        .slice(i, i + length)
        .map((token) => token.word)
        .join(' ');
      const entity = index.byKey.get(key);
      if (!entity) continue;
      mentions.push({ start: tokens[i]!.start, end: tokens[i + length - 1]!.end, ...entity });
      matched = length;
      break;
    }
    i += matched || 1;
  }
  return mentions;
}
