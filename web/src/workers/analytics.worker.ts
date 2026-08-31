import * as Comlink from 'comlink';

import { topQuotientsByTheme, type QuotientEntry } from '@/core/locational-quotient';
import { lotkaDistribution, type LotkaDistribution } from '@/core/scientometrics';
import { docsPerAuthor, docsPerYear, metadataCompleteness, summarize } from '@/core/summary';
import { authorsTable, countriesTable, keywordsTable, venuesTable, type EntityRow } from '@/core/tables';
import {
  boxplotOptions,
  boxplotSeries,
  type BoxplotDimension,
  type BoxplotMetric,
  type BoxplotSeries,
} from '@/core/viz/boxplot';
import { collaborationNetwork, type CollaborationNetwork } from '@/core/viz/collaboration';
import { conceptMap, type ConceptMapOptions, type ConceptTerm } from '@/core/viz/concept-map';
import { keywordGenetics, type KeywordGenetics } from '@/core/viz/genetics';
import { historiograph, type HistoriographData } from '@/core/viz/historiograph';
import {
  productionTimeline,
  type ProductionCategory,
  type ProductionSeries,
} from '@/core/viz/production-timeline';
import {
  sankeyEvolution,
  suggestPeriods,
  type Period,
  type SankeyData,
} from '@/core/viz/sankey';
import { thematicMap, type ThematicMap } from '@/core/viz/thematic-map';
import { FIELD_CANDIDATES, currentYear } from '@/lib/schema';
import { collectColumns, pickColumn } from '@/core/text';
import type { Dataset, DatasetSummary, MetadataCompleteness } from '@/lib/types';

/**
 * Worker de análise: índices cientométricos, tabelas de entidade, QL e Lotka.
 *
 * As quatro tabelas percorrem a base inteira explodindo campos multivalorados — em
 * 10.000 documentos isso são centenas de milhares de linhas intermediárias.
 */

export interface AnalyticsBundle {
  summary: DatasetSummary;
  completeness: MetadataCompleteness[];
  docsPerYear: { year: number; count: number }[];
  lotka: LotkaDistribution | null;
}

export interface EntityTables {
  authors: EntityRow[];
  countries: EntityRow[];
  venues: EntityRow[];
  keywords: EntityRow[];
}

export interface QuotientTables {
  authors: [string, QuotientEntry][];
  countries: [string, QuotientEntry][];
  venues: [string, QuotientEntry][];
}

const api = {
  /** Painel de visão geral: KPIs, completude, produção por ano e Lotka. */
  overview(dataset: Dataset, baseYear: number = currentYear()): AnalyticsBundle {
    return {
      summary: summarize(dataset, baseYear),
      completeness: metadataCompleteness(dataset),
      docsPerYear: docsPerYear(dataset),
      lotka: lotkaDistribution(docsPerAuthor(dataset)),
    };
  },

  /**
   * Produção ao longo do tempo, quebrada pela categoria escolhida (país, tema de IA,
   * base de dados ou tipo de trabalho) — separado de `overview` porque só é calculado
   * quando o usuário troca a categoria, não a cada carregamento da base.
   */
  productionTimeline(dataset: Dataset, category: ProductionCategory): ProductionSeries[] {
    return productionTimeline(dataset, category);
  },

  /** As quatro tabelas analíticas, com índices h/g/i10/m por entidade. */
  tables(dataset: Dataset, baseYear: number = currentYear()): EntityTables {
    return {
      authors: authorsTable(dataset, baseYear),
      countries: countriesTable(dataset, baseYear),
      venues: venuesTable(dataset, baseYear),
      keywords: keywordsTable(dataset, baseYear),
    };
  },

  /** Ciclo de vida das palavras-chave — nascimento, longevidade, replicação, impacto. */
  genetics(dataset: Dataset): KeywordGenetics[] {
    return keywordGenetics(dataset);
  },

  /** Períodos sugeridos para o Sankey, cobrindo o intervalo da base em três fatias. */
  sankeyPeriods(dataset: Dataset): [Period, Period, Period] | null {
    return suggestPeriods(dataset);
  },

  /** Fluxo de evolução temática entre três períodos. */
  sankey(
    dataset: Dataset,
    periods: [Period, Period, Period],
    topN: number,
  ): SankeyData | null {
    return sankeyEvolution(dataset, periods, topN);
  },

  /** Linha do tempo de citações diretas entre os documentos mais citados. */
  historiograph(dataset: Dataset, topN: number): HistoriographData | null {
    return historiograph(dataset, topN);
  },

  /** Mapa conceitual: termos projetados por PCA sobre sua coocorrência em documentos. */
  conceptMap(dataset: Dataset, options: ConceptMapOptions): ConceptTerm[] {
    return conceptMap(dataset, options);
  },

  /** Rede de colaboração internacional, com coordenadas para o mapa. */
  collaboration(dataset: Dataset, topN: number): CollaborationNetwork {
    return collaborationNetwork(dataset, topN);
  },

  /**
   * Mapa temático de centralidade × densidade.
   *
   * Roda sobre resumos quando existirem, e sobre palavras-chave como reserva: o resumo dá
   * um vocabulário muito mais rico para a rede de coocorrência.
   */
  thematicMap(dataset: Dataset, source: 'abstract' | 'keywords', topWords: number): ThematicMap | null {
    const columns = collectColumns(dataset);
    const column =
      source === 'abstract' ? 'ABSTRACT' : pickColumn(columns, FIELD_CANDIDATES.keywords);
    return column ? thematicMap(dataset, column, topWords) : null;
  },

  /** Entidades disponíveis para comparação no boxplot. */
  boxplotOptions(dataset: Dataset, dimension: BoxplotDimension): string[] {
    return boxplotOptions(dataset, dimension);
  },

  /** Séries de distribuição para o boxplot. */
  boxplot(
    dataset: Dataset,
    dimension: BoxplotDimension,
    metric: BoxplotMetric,
    selected: string[],
  ): BoxplotSeries[] {
    return boxplotSeries(dataset, dimension, metric, selected);
  },

  /**
   * Entidade de maior QL por tema.
   *
   * Os `Map` são serializados como pares porque a clonagem estrutural do Comlink os
   * preserva, mas a UI consome como lista de qualquer forma.
   */
  quotients(dataset: Dataset): QuotientTables {
    const top = topQuotientsByTheme(dataset);
    return {
      authors: [...top.authors],
      countries: [...top.countries],
      venues: [...top.venues],
    };
  },
};

export type AnalyticsWorkerApi = typeof api;

Comlink.expose(api);
