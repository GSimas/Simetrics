import Graph from 'graphology';

import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, NodeKind } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, splitTokens } from '../text';

/**
 * Construção dos grafos bibliométricos.
 *
 * Duas topologias distintas, com propósitos distintos:
 *
 * - **Heterogênea** (`buildHeterogeneousGraph`): documentos ligados a seus autores,
 *   países e venues. É o ecossistema completo, usado nas métricas de ecologia profunda.
 * - **Coocorrência** (`buildCooccurrenceGraph`): entidades do mesmo tipo ligadas quando
 *   aparecem no mesmo documento, com peso pela frequência. É a rede de coautoria ou de
 *   palavras-chave que o usuário visualiza.
 */

/** Grafo heterogêneo com o tipo de cada nó — ⇄ `gerar_tabela_metricas_completas`. */
export interface HeterogeneousGraph {
  graph: Graph;
  nodeTypes: Map<string, NodeKind>;
}

export function buildHeterogeneousGraph(
  rows: Dataset,
  onProgress?: (ratio: number) => void,
): HeterogeneousGraph {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);
  const hasCountry = columns.has(FIELD.COUNTRY);

  const graph = new Graph({ type: 'undirected', multi: false });
  const nodeTypes = new Map<string, NodeKind>();

  if (!titleColumn) return { graph, nodeTypes };

  const ensureNode = (key: string, kind: NodeKind): void => {
    if (!graph.hasNode(key)) graph.addNode(key);
    // Última escrita vence, como o `node_types[x] = ...` do Python. Importa quando a mesma
    // string é documento e venue ao mesmo tempo — o caso dos livros, cujo título de
    // publicação repete o título da obra.
    nodeTypes.set(key, kind);
  };

  const ensureEdge = (source: string, target: string): void => {
    // Laços próprios são preservados de propósito. Eles aparecem quando o venue tem o
    // mesmo nome do documento (livros), e o NetworkX os mantém, contando-os como grau 2.
    // Descartá-los mudaria silenciosamente o grau desses nós na tabela SNA.
    if (!graph.hasEdge(source, target)) graph.addEdge(source, target);
  };

  rows.forEach((doc, position) => {
    if (onProgress && position % 500 === 0) onProgress(position / rows.length);

    const title = String(doc[titleColumn] ?? '').trim();
    if (!title || isNullLike(title)) return;

    ensureNode(title, 'Documento');

    if (authorsColumn) {
      for (const author of splitTokens(doc[authorsColumn])) {
        ensureNode(author, 'Autor');
        ensureEdge(title, author);
      }
    }

    if (hasCountry) {
      for (const country of splitTokens(doc[FIELD.COUNTRY])) {
        ensureNode(country, 'País');
        ensureEdge(title, country);
      }
    }

    if (venueColumn) {
      const venue = String(doc[venueColumn] ?? '').trim();
      if (venue && !isNullLike(venue)) {
        ensureNode(venue, 'Local de Publicação (Venue)');
        ensureEdge(title, venue);
      }
    }
  });

  return { graph, nodeTypes };
}

export interface CooccurrenceOptions {
  /** Mantém apenas as `topN` entidades mais frequentes. */
  topN?: number;
}

/**
 * Rede de coocorrência ponderada — ⇄ `criar_grafo_e_metricas` (utils.py:2183).
 *
 * O recorte por `topN` acontece ANTES de formar as arestas, e não depois: uma rede de
 * coautoria completa tem milhares de nós periféricos que só poluem a visualização, e
 * filtrar depois deixaria arestas apontando para nós inexistentes.
 */
export function buildCooccurrenceGraph(
  rows: Dataset,
  column: string,
  options: CooccurrenceOptions = {},
): Graph {
  const topN = options.topN ?? 30;

  // Lista de entidades por documento, preservada para a segunda passagem.
  const perDocument: string[][] = [];
  const frequency = new Map<string, number>();

  for (const doc of rows) {
    const items = splitTokens(doc[column]);
    if (items.length === 0) continue;

    perDocument.push(items);
    for (const item of items) frequency.set(item, (frequency.get(item) ?? 0) + 1);
  }

  const top = new Set(
    [...frequency.entries()]
      .sort((left, right) => right[1] - left[1] || (left[0] < right[0] ? -1 : 1))
      .slice(0, topN)
      .map(([item]) => item),
  );

  const graph = new Graph({ type: 'undirected', multi: false });
  for (const item of top) graph.addNode(item, { count: frequency.get(item) ?? 0 });

  for (const items of perDocument) {
    // Únicos e ordenados: um autor repetido na mesma string não deve gerar laço próprio,
    // e a ordem estável mantém o par (u, v) canônico.
    const present = [...new Set(items.filter((item) => top.has(item)))].sort();
    if (present.length < 2) continue;

    for (let i = 0; i < present.length; i += 1) {
      for (let j = i + 1; j < present.length; j += 1) {
        const source = present[i] as string;
        const target = present[j] as string;

        if (graph.hasEdge(source, target)) {
          graph.updateEdgeAttribute(source, target, 'weight', (weight) => (weight as number) + 1);
        } else {
          graph.addEdge(source, target, { weight: 1 });
        }
      }
    }
  }

  return graph;
}

/** Coluna do dataset correspondente a cada tipo de rede oferecido na interface. */
export const COOCCURRENCE_SOURCES = {
  Coautoria: FIELD_CANDIDATES.authors,
  'Palavras-chave': FIELD_CANDIDATES.keywords,
  Países: [FIELD.COUNTRY],
} as const satisfies Record<string, readonly string[]>;

export type CooccurrenceKind = keyof typeof COOCCURRENCE_SOURCES;

/** Resolve a coluna disponível para o tipo de rede pedido. */
export function resolveCooccurrenceColumn(rows: Dataset, kind: CooccurrenceKind): string | null {
  return pickColumn(collectColumns(rows), COOCCURRENCE_SOURCES[kind]);
}
