import * as Comlink from 'comlink';

import { clusterDocuments, type ClusteringResult } from '@/core/clustering';
import {
  buildIndex,
  search,
  toContextDocuments,
  type ContextDocument,
  type RetrievalIndex,
} from '@/core/retrieval';
import { summarize } from '@/core/summary';
import { authorsTable, countriesTable, keywordsTable, venuesTable } from '@/core/tables';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, WorkerProgress } from '@/lib/types';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import { executeAnalyticalTool, type ToolExecutionResponse } from '@/core/tools';

/**
 * Worker de IA: agrupamento temático, seleção de contexto e execução de ferramentas analíticas.
 *
 * As tarefas ficam aqui porque compartilham o custo pesado — vetorização TF-IDF
 * e tabelas cienciométricas sobre a base inteira — sem bloquear a interface.
 */

export type ProgressCallback = (progress: WorkerProgress) => void;

/** Índice BM25 mantido entre perguntas, para não reindexar a cada consulta. */
let cachedIndex: { dataset: Dataset; index: RetrievalIndex } | null = null;

function getIndex(dataset: Dataset): RetrievalIndex {
  // A comparação é por identidade de referência: o store cria um novo array sempre que a
  // base muda (deduplicação, novo upload), então isso já detecta invalidação.
  if (cachedIndex && cachedIndex.dataset === dataset) return cachedIndex.index;

  const index = buildIndex(dataset);
  cachedIndex = { dataset, index };
  return index;
}

/**
 * Panorama agregado da base, enviado ao modelo junto com os documentos recuperados.
 *
 * Dá visão macro a um custo mínimo de tokens.
 */
export interface AggregateSummary {
  totalDocuments: number;
  timespan: string;
  totalAuthors: number;
  totalVenues: number;
  totalCountries: number;
  topAuthors: { name: string; documents: number; citations: number; h: number }[];
  topVenues: { name: string; documents: number; citations: number }[];
  topCountries: { name: string; documents: number; citations: number }[];
  topKeywords: { name: string; documents: number }[];
  mostCited: { title: string; year: number | null; citations: number }[];
  themes: { name: string; documents: number }[];
  productionPerYear: { year: number; documents: number }[];
}

const TOP_N = 25;

function buildAggregate(dataset: Dataset): AggregateSummary {
  const summary = summarize(dataset);
  const columns = collectColumns(dataset);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);

  const mostCited = [...dataset]
    .sort(
      (left, right) =>
        (toNumeric(right[FIELD.TOTAL_CITATIONS]) ?? 0) -
        (toNumeric(left[FIELD.TOTAL_CITATIONS]) ?? 0),
    )
    .slice(0, TOP_N)
    .map((doc) => ({
      title: titleColumn ? String(doc[titleColumn] ?? '') : '',
      year: toNumeric(doc[FIELD.YEAR_CLEAN]),
      citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
    }));

  const themeCounts = new Map<string, number>();
  if (columns.has(FIELD.THEME)) {
    for (const doc of dataset) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme) themeCounts.set(theme, (themeCounts.get(theme) ?? 0) + 1);
    }
  }

  // Anos de produção
  const yearCounts = new Map<number, number>();
  for (const doc of dataset) {
    const y = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (y !== null) {
      const year = Math.trunc(y);
      yearCounts.set(year, (yearCounts.get(year) ?? 0) + 1);
    }
  }
  const productionPerYear = [...yearCounts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([year, documents]) => ({ year, documents }));

  return {
    totalDocuments: summary.totalDocs,
    timespan: summary.timespan,
    totalAuthors: summary.authorsCount,
    totalVenues: summary.venuesCount,
    totalCountries: summary.countriesCount,
    topAuthors: authorsTable(dataset)
      .slice(0, TOP_N)
      .map((row) => ({
        name: row.entity,
        documents: row.docCount,
        citations: row.citations,
        h: row.h,
      })),
    topVenues: venuesTable(dataset)
      .slice(0, TOP_N)
      .map((row) => ({ name: row.entity, documents: row.docCount, citations: row.citations })),
    topCountries: countriesTable(dataset)
      .slice(0, TOP_N)
      .map((row) => ({ name: row.entity, documents: row.docCount, citations: row.citations })),
    topKeywords: keywordsTable(dataset)
      .slice(0, TOP_N)
      .map((row) => ({ name: row.entity, documents: row.docCount })),
    mostCited,
    themes: [...themeCounts.entries()]
      .sort((left, right) => right[1] - left[1])
      .map(([name, documents]) => ({ name, documents })),
    productionPerYear,
  };
}

/** Agregado guardado por base, já que recalculá-lo custa as quatro tabelas analíticas. */
let cachedAggregate: { dataset: Dataset; summary: AggregateSummary } | null = null;

export interface ChatContext {
  documents: ContextDocument[];
  aggregate: AggregateSummary;
}

const api = {
  /**
   * Seleciona o contexto de uma pergunta: documentos relevantes por BM25 mais o
   * panorama agregado da base.
   */
  buildChatContext(dataset: Dataset, question: string, topN = 40): ChatContext {
    const index = getIndex(dataset);
    const hits = search(index, question, topN);

    if (!cachedAggregate || cachedAggregate.dataset !== dataset) {
      cachedAggregate = { dataset, summary: buildAggregate(dataset) };
    }

    return {
      documents: toContextDocuments(dataset, hits),
      aggregate: cachedAggregate.summary,
    };
  },

  /** Executa uma ferramenta analítica de forma isolada no worker */
  executeTool(
    toolName: string,
    args: Record<string, unknown>,
    dataset: Dataset,
  ): ToolExecutionResponse {
    return executeAnalyticalTool(toolName, args, dataset);
  },

  /** Agrupa a base por similaridade textual, devolvendo amostras para nomeação. */
  cluster(
    dataset: Dataset,
    maxClusters: number,
    onProgress?: ProgressCallback,
  ): ClusteringResult | null {
    return clusterDocuments(dataset, {
      maxClusters,
      onProgress: (ratio, phase) => onProgress?.({ phase, ratio }),
    });
  },
};

export type AiWorkerApi = typeof api;

Comlink.expose(api);
