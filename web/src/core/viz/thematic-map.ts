import Graph from 'graphology';
import louvain from 'graphology-communities-louvain';

import type { Dataset } from '@/lib/types';
import { WORDCLOUD_STOP_WORDS } from '../ml/wordcloud-stop-words';
import { mean } from '../stats';

/**
 * Mapa temático — ⇄ `gerar_mapa_tematico` (utils.py:2450).
 *
 * Posiciona cada agrupamento de termos em dois eixos, no estilo do Bibliometrix:
 *
 * - **Centralidade** (eixo X): força das ligações do agrupamento com o RESTO da rede.
 *   Mede o quanto o tema conversa com os demais.
 * - **Densidade** (eixo Y): força das ligações INTERNAS ao agrupamento. Mede o quanto o
 *   tema está desenvolvido internamente.
 *
 * Os quadrantes formados pelas médias dão a leitura: alta/alta são temas motores;
 * baixa centralidade e alta densidade são nichos isolados; alta centralidade e baixa
 * densidade são temas básicos e transversais; baixa/baixa são emergentes ou em declínio.
 */

/** Termos genéricos de escrita acadêmica, que apareceriam em todo agrupamento. */
const EXTRA_STOP_WORDS = [
  'research',
  'study',
  'analysis',
  'results',
  'using',
  'paper',
  'article',
  'author',
  'may',
  'can',
  'will',
] as const;

const STOP_WORDS = new Set<string>([...WORDCLOUD_STOP_WORDS, ...EXTRA_STOP_WORDS]);

/** Palavras com ao menos 3 caracteres — ⇄ `re.findall(r'\b\w{3,}\b', ...)`. */
const WORD_PATTERN = /[\p{L}\p{N}_]{3,}/gu;

export interface ThematicCluster {
  id: number;
  /** Rótulo curto: os três termos mais frequentes. */
  label: string;
  /** Os seis termos mais frequentes, para o tooltip. */
  terms: string[];
  /** Soma das ligações internas — eixo Y. */
  density: number;
  /** Soma das ligações externas — eixo X. */
  centrality: number;
  /** Frequência total dos termos, usada como tamanho da bolha. */
  frequency: number;
}

export interface ThematicMap {
  clusters: ThematicCluster[];
  /** Médias que delimitam os quadrantes. */
  meanCentrality: number;
  meanDensity: number;
}

export function thematicMap(
  rows: Dataset,
  column: string,
  topWords = 150,
): ThematicMap | null {
  // Tokeniza cada documento, descartando stop words.
  const documents: string[][] = [];
  const totalFrequency = new Map<string, number>();

  for (const doc of rows) {
    const text = String(doc[column] ?? '').toLowerCase();
    if (!text.trim()) continue;

    const words: string[] = [];
    WORD_PATTERN.lastIndex = 0;

    let match: RegExpExecArray | null;
    while ((match = WORD_PATTERN.exec(text)) !== null) {
      const word = match[0];
      if (STOP_WORDS.has(word)) continue;
      words.push(word);
      totalFrequency.set(word, (totalFrequency.get(word) ?? 0) + 1);
    }

    if (words.length > 0) documents.push(words);
  }

  const selected = new Set(
    [...totalFrequency.entries()]
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, topWords)
      .map(([word]) => word),
  );

  if (selected.size === 0) return null;

  // Rede de coocorrência entre os termos selecionados.
  const graph = new Graph({ type: 'undirected', multi: false });

  for (const words of documents) {
    const present = words.filter((word) => selected.has(word));

    for (let i = 0; i < present.length; i += 1) {
      for (let j = i + 1; j < present.length; j += 1) {
        const source = present[i] as string;
        const target = present[j] as string;
        if (source === target) continue;

        if (!graph.hasNode(source)) graph.addNode(source);
        if (!graph.hasNode(target)) graph.addNode(target);

        if (graph.hasEdge(source, target)) {
          graph.updateEdgeAttribute(source, target, 'weight', (weight) => (weight as number) + 1);
        } else {
          graph.addEdge(source, target, { weight: 1 });
        }
      }
    }
  }

  if (graph.order === 0 || graph.size === 0) return null;

  // Louvain no lugar do `greedy_modularity_communities`: ambos otimizam modularidade,
  // e o Louvain é substancialmente mais rápido.
  const communities = louvain(graph, { resolution: 1 });

  const membersByCommunity = new Map<number, string[]>();
  for (const [node, community] of Object.entries(communities)) {
    let bucket = membersByCommunity.get(community);
    if (!bucket) membersByCommunity.set(community, (bucket = []));
    bucket.push(node);
  }

  const clusters: ThematicCluster[] = [];
  let index = 0;

  for (const [, members] of membersByCommunity) {
    // Agrupamentos de um termo só não têm densidade interna a medir.
    if (members.length < 2) continue;

    const memberSet = new Set(members);
    let internal = 0;
    let external = 0;

    for (const node of members) {
      graph.forEachNeighbor(node, (neighbor, attributes) => {
        const weight = Number(attributes['weight'] ?? 1);
        if (memberSet.has(neighbor)) internal += weight;
        else external += weight;
      });
    }

    // As ligações internas foram contadas pelas duas pontas.
    internal /= 2;

    const ordered = [...members].sort(
      (left, right) => (totalFrequency.get(right) ?? 0) - (totalFrequency.get(left) ?? 0),
    );

    index += 1;
    clusters.push({
      id: index,
      label: ordered.slice(0, 3).join('<br>'),
      terms: ordered.slice(0, 6),
      density: internal,
      centrality: external,
      frequency: members.reduce((total, word) => total + (totalFrequency.get(word) ?? 0), 0),
    });
  }

  if (clusters.length === 0) return null;

  return {
    clusters,
    meanCentrality: mean(clusters.map((cluster) => cluster.centrality)),
    meanDensity: mean(clusters.map((cluster) => cluster.density)),
  };
}
