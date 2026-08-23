import { kmeans } from 'ml-kmeans';
import { PCA } from 'ml-pca';

import { FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { collectColumns, pickColumn, splitTokens, titleCase } from '../text';

/**
 * Mapa conceitual por PCA — ⇄ `gerar_mapas_conceituais` (utils.py:1280).
 *
 * Posiciona TERMOS, e não documentos, num espaço reduzido: dois termos ficam próximos
 * quando aparecem nos mesmos documentos. O resultado é a topologia da área — ilhas de
 * conceitos que andam juntos, e termos de fronteira entre elas.
 *
 * A matriz é transposta em relação ao clustering temático: lá agrupamos documentos por
 * seus termos, aqui agrupamos termos por seus documentos.
 *
 * Os termos vêm inteiros do campo de palavras-chave, separados por ';' — nunca quebrados
 * por espaço. É o que preserva expressões compostas como "Knowledge Management" em vez de
 * dissolvê-las em duas palavras sem sentido isolado.
 */

const SEED = 42;

export interface ConceptTerm {
  term: string;
  /** Coordenadas nos três primeiros componentes principais. */
  x: number;
  y: number;
  z: number;
  cluster: number;
  /** Documentos em que o termo aparece. */
  frequency: number;
}

export interface ConceptMapOptions {
  /** Termos mais frequentes considerados. */
  topTerms?: number;
  /** Agrupamentos a formar. */
  clusters?: number;
}

export function conceptMap(rows: Dataset, options: ConceptMapOptions = {}): ConceptTerm[] {
  const topTerms = options.topTerms ?? 50;
  const clusterCount = options.clusters ?? 4;

  const keywordsColumn = pickColumn(collectColumns(rows), FIELD_CANDIDATES.keywords);
  if (!keywordsColumn) return [];

  // Termos por documento, em caixa de título — ⇄ o `custom_tokenizer` do Python.
  const perDocument: Set<string>[] = [];
  const frequency = new Map<string, number>();

  for (const doc of rows) {
    const terms = new Set(splitTokens(doc[keywordsColumn]).map(titleCase));
    if (terms.size === 0) continue;

    perDocument.push(terms);
    for (const term of terms) frequency.set(term, (frequency.get(term) ?? 0) + 1);
  }

  if (perDocument.length === 0) return [];

  const selected = [...frequency.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, topTerms)
    .map(([term]) => term);

  // PCA precisa de ao menos três dimensões de saída e mais termos que agrupamentos.
  if (selected.length < Math.max(clusterCount, 4)) return [];

  // Matriz termo × documento: cada linha é o perfil de ocorrência de um termo,
  // normalizada para comprimento unitário.
  //
  // A normalização é o que torna o mapa legível, e é uma divergência deliberada em
  // relação ao Python, que agrupa sobre as contagens brutas. Sem ela, o comprimento do
  // vetor é proporcional à frequência do termo, e o K-Means acaba separando por QUÃO
  // COMUM o termo é em vez de COM QUEM ele aparece — que é a pergunta do mapa conceitual.
  //
  // Medido na base de exemplo: sobre contagens brutas os quatro agrupamentos saem com
  // 3, 1, 45 e 1 termos, ou seja, uma mancha única. Normalizado, saem 2, 23, 20 e 5, e
  // correspondem a teoria memética, algoritmos de otimização, cultura e cognição, e
  // comunicação digital.
  const matrix = selected.map((term) => {
    const profile = perDocument.map((terms) => (terms.has(term) ? 1 : 0));
    let squared = 0;
    for (const value of profile) squared += value * value;
    const norm = Math.sqrt(squared) || 1;
    return profile.map((value) => value / norm);
  });

  const partition = kmeans(matrix, Math.min(clusterCount, selected.length), {
    seed: SEED,
    initialization: 'kmeans++',
    maxIterations: 100,
  });

  const pca = new PCA(matrix, { center: true });
  const projected = pca.predict(matrix, { nComponents: 3 }).to2DArray();

  return selected.map((term, index) => {
    const coordinates = projected[index] ?? [];
    return {
      term,
      x: coordinates[0] ?? 0,
      y: coordinates[1] ?? 0,
      z: coordinates[2] ?? 0,
      cluster: partition.clusters[index] ?? 0,
      frequency: frequency.get(term) ?? 0,
    };
  });
}
