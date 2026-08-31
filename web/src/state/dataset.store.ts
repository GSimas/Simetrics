import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import type { AnalyticsBundle, EntityTables } from '@/workers/analytics.worker';
import type { CooccurrenceReport, SnaReport } from '@/core/graph';
import type { UploadedFile } from '@/core/parsers';
import type { RisSource } from '@/core/parsers/pipeline-ris';
import { applyThemes, fallbackThemeName, type ClusteringResult } from '@/core/clustering';
import { buildSearchOptions, type SearchOptions } from '@/core/search';
import { labelCluster } from '@/lib/ai-client';
import { DEMO_FILES } from '@/lib/demo';
import type { DatabaseName } from '@/lib/schema';
import type { Dataset, DuplicateRecord, WorkerProgress } from '@/lib/types';
import {
  getAiWorker,
  getAnalyticsWorker,
  getGraphWorker,
  getIngestWorker,
  proxyProgress,
  terminateWorkers,
} from '@/workers/client';

/**
 * Estado global da base carregada — ⇄ o `st.session_state` do Streamlit.
 *
 * A diferença de fundo em relação ao Streamlit: lá o script inteiro reexecuta a cada
 * interação e o cache decide o que recalcular. Aqui nada reexecuta sozinho, então cada
 * derivação é disparada explicitamente e guardada. Trocar a base invalida tudo o que
 * dela derivava, e é o que `resetDerived` faz.
 */

export type DedupStrategy = 'none' | 'doi' | 'similarity' | 'both';

/** Metadados de origem de um arquivo carregado — tudo que sobra após a ingestão, usado
 * só para exibição (nome do projeto padrão, cartão de projeto na tela inicial). */
export interface DatasetSourceFile {
  name: string;
  database: DatabaseName | string;
}

interface DatasetState {
  /** Base como veio dos arquivos, antes de qualquer deduplicação. */
  original: Dataset | null;
  /** Base ativa: o que as análises consomem. */
  active: Dataset | null;
  duplicates: DuplicateRecord[];
  dedupStrategy: DedupStrategy;
  /** Limiar usado na última deduplicação por similaridade — `null` quando não se aplica
   * (estratégia 'none'/'doi'). Existe só para o Projeto lembrar o valor ao reabrir. */
  dedupThreshold: number | null;
  /** Arquivos de origem da base ativa — preenchido por `loadFiles`/`loadDemo`, os únicos
   * dois pontos de entrada de dados no app. Usado pela camada de Projetos. */
  sourceFiles: DatasetSourceFile[];

  overview: AnalyticsBundle | null;
  tables: EntityTables | null;
  sna: SnaReport | null;
  network: CooccurrenceReport | null;
  searchOptions: SearchOptions | null;
  /** Resultado da categorização temática, quando já executada. */
  clustering: ClusteringResult | null;

  isIngesting: boolean;
  isDeduplicating: boolean;
  isCategorizingThemes: boolean;
  progress: WorkerProgress | null;
  snaProgress: WorkerProgress | null;
  error: string | null;

  loadFiles: (files: UploadedFile[]) => Promise<void>;
  loadDemo: () => Promise<void>;
  applyDedup: (strategy: DedupStrategy, threshold?: number) => Promise<void>;
  computeOverview: () => Promise<void>;
  computeTables: () => Promise<void>;
  computeSna: () => Promise<void>;
  categorizeThemes: (maxClusters?: number) => Promise<void>;
  computeNetwork: (
    kind: Parameters<Awaited<ReturnType<typeof getGraphWorker>>['cooccurrence']>[1],
    topN: number,
    sizeMetric: Parameters<Awaited<ReturnType<typeof getGraphWorker>>['cooccurrence']>[3],
  ) => Promise<void>;
  reset: () => void;
}

/** Tudo o que deixa de valer quando a base ativa muda. Exportado porque
 * `state/project.store.ts` reaproveita a mesma forma ao hidratar um projeto salvo, em
 * vez de duplicar a lista de campos e arriscar as duas ficarem dessincronizadas. */
export const DERIVED_RESET = {
  overview: null,
  tables: null,
  sna: null,
  network: null,
  searchOptions: null,
  clustering: null,
} as const;

function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

export const useDataset = create<DatasetState>()(subscribeWithSelector((set, get) => ({
  original: null,
  active: null,
  duplicates: [],
  dedupStrategy: 'none',
  dedupThreshold: null,
  sourceFiles: [],
  ...DERIVED_RESET,
  isIngesting: false,
  isDeduplicating: false,
  isCategorizingThemes: false,
  progress: null,
  snaProgress: null,
  error: null,

  async loadFiles(files) {
    set({ isIngesting: true, progress: { phase: 'Lendo arquivos', ratio: 0 }, error: null });

    try {
      const worker = getIngestWorker();
      const dataset = await worker.ingest(
        files,
        proxyProgress((update: WorkerProgress) => set({ progress: update })),
      );

      set({
        original: dataset,
        active: dataset,
        duplicates: [],
        dedupStrategy: 'none',
        dedupThreshold: null,
        sourceFiles: files.map(({ name, database }) => ({ name, database })),
        ...DERIVED_RESET,
        searchOptions: buildSearchOptions(dataset),
      });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ isIngesting: false, progress: null });
    }
  },

  async loadDemo() {
    set({ isIngesting: true, progress: { phase: 'Carregando base de demonstração', ratio: 0 }, error: null });

    try {
      const sources: RisSource[] = await Promise.all(
        DEMO_FILES.map(async ({ name, database }) => {
          const response = await fetch(`${import.meta.env.BASE_URL}demo/${name}`);
          if (!response.ok) throw new Error(`Falha ao carregar ${name}: HTTP ${response.status}`);
          return { name, database, text: await response.text() };
        }),
      );

      const worker = getIngestWorker();
      const dataset = await worker.ingestRis(sources);

      set({
        original: dataset,
        active: dataset,
        duplicates: [],
        dedupStrategy: 'none',
        dedupThreshold: null,
        sourceFiles: DEMO_FILES.map(({ name, database }) => ({ name, database })),
        ...DERIVED_RESET,
        searchOptions: buildSearchOptions(dataset),
      });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ isIngesting: false, progress: null });
    }
  },

  async applyDedup(strategy, threshold = 0.9) {
    const { original } = get();
    if (!original) return;

    set({ isDeduplicating: true, progress: { phase: 'Deduplicando', ratio: 0 }, error: null });

    try {
      if (strategy === 'none') {
        set({
          active: original,
          duplicates: [],
          dedupStrategy: 'none',
          dedupThreshold: null,
          ...DERIVED_RESET,
          searchOptions: buildSearchOptions(original),
        });
        return;
      }

      const worker = getIngestWorker();
      // A deduplicação parte SEMPRE da base original, nunca da já deduplicada: aplicar
      // uma estratégia sobre o resultado da outra acumularia remoções e impediria o
      // usuário de voltar atrás.
      let result: { kept: Dataset; removed: DuplicateRecord[] };

      if (strategy === 'doi') {
        result = await worker.dedupByDoi(original);
      } else if (strategy === 'similarity') {
        result = await worker.dedupBySimilarity(
          original,
          threshold,
          proxyProgress((update: WorkerProgress) => set({ progress: update })),
        );
      } else if (strategy === 'both') {
        result = await worker.dedupBoth(
          original,
          threshold,
          proxyProgress((update: WorkerProgress) => set({ progress: update })),
        );
      } else {
        result = { kept: original, removed: [] };
      }

      set({
        active: result.kept,
        duplicates: result.removed,
        dedupStrategy: strategy,
        dedupThreshold: strategy === 'similarity' || strategy === 'both' ? threshold : null,
        ...DERIVED_RESET,
        searchOptions: buildSearchOptions(result.kept),
      });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ isDeduplicating: false, progress: null });
    }
  },

  async computeOverview() {
    const { active, overview } = get();
    if (!active || overview) return;

    try {
      const result = await getAnalyticsWorker().overview(active);
      set({ overview: result });
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async computeTables() {
    const { active, tables } = get();
    if (!active || tables) return;

    try {
      const result = await getAnalyticsWorker().tables(active);
      set({ tables: result });
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async computeSna() {
    const { active, sna } = get();
    if (!active || sna) return;

    try {
      set({ snaProgress: { phase: 'Iniciando análise de redes', ratio: 0 } });
      const result = await getGraphWorker().heterogeneous(
        active,
        proxyProgress((update: WorkerProgress) => set({ snaProgress: update })),
      );
      set({ sna: result });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ snaProgress: null });
    }
  },

  /**
   * Agrupa os documentos por similaridade e nomeia cada tema com o Gemini.
   *
   * O agrupamento roda inteiro no navegador; só a nomeação sai para a rede, e ainda
   * assim uma requisição por tema. É o que respeita o timeout: o laço equivalente no
   * Python leva ~25 s numa chamada só (utils.py:1215), acima do limite da Netlify.
   *
   * Quando a nomeação falha — chave ausente, cota, indisponibilidade — o tema recebe um
   * rótulo derivado dos próprios termos característicos. Perder o nome bonito não pode
   * custar o agrupamento inteiro, que é a parte cara.
   */
  async categorizeThemes(maxClusters = 10) {
    const { active } = get();
    if (!active) return;

    set({ isCategorizingThemes: true, progress: { phase: 'Agrupando documentos', ratio: 0 }, error: null });

    try {
      const result = await getAiWorker().cluster(
        active,
        maxClusters,
        proxyProgress((update: WorkerProgress) => set({ progress: update })),
      );

      if (!result) {
        set({ error: 'A base é pequena demais para identificar agrupamentos temáticos.' });
        return;
      }

      const names = new Map<number, string>();
      let failures = 0;

      for (const [position, cluster] of result.clusters.entries()) {
        set({
          progress: {
            phase: 'Nomeando temas',
            ratio: (position + 1) / result.clusters.length,
            detail: `${position + 1} de ${result.clusters.length}`,
          },
        });

        try {
          names.set(cluster.clusterId, await labelCluster({
            samples: cluster.samples,
            topTerms: cluster.topTerms,
          }));
        } catch {
          names.set(cluster.clusterId, fallbackThemeName(cluster));
          failures += 1;
        }
      }

      const themed = applyThemes(active, result.assignments, names);

      set({
        active: themed,
        ...DERIVED_RESET,
        clustering: result,
        searchOptions: buildSearchOptions(themed),
        error:
          failures > 0
            ? `${failures} de ${result.clusters.length} temas ficaram sem nome da IA e receberam rótulo automático a partir dos termos característicos.`
            : null,
      });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ isCategorizingThemes: false, progress: null });
    }
  },

  async computeNetwork(kind, topN, sizeMetric) {
    const { active } = get();
    if (!active) return;

    try {
      const result = await getGraphWorker().cooccurrence(active, kind, topN, sizeMetric);
      set({ network: result });
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  reset() {
    terminateWorkers();
    set({
      original: null,
      active: null,
      duplicates: [],
      dedupStrategy: 'none',
      dedupThreshold: null,
      sourceFiles: [],
      isIngesting: false,
      isDeduplicating: false,
      isCategorizingThemes: false,
      progress: null,
      snaProgress: null,
      error: null,
      ...DERIVED_RESET,
    });
  },
})));

/** True quando há base carregada — usado para liberar as abas de análise. */
export function useHasDataset(): boolean {
  return useDataset((state) => state.active !== null && state.active.length > 0);
}
