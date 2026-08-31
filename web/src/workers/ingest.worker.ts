import * as Comlink from 'comlink';

import { dedupBoth, dedupByDoi, dedupBySimilarity } from '@/core/dedup';
import { processFiles, type UploadedFile } from '@/core/parsers';
import { processRisFiles, type RisSource } from '@/core/parsers/pipeline-ris';
import { MAX_DOCUMENTS } from '@/lib/schema';
import type { Dataset, DedupResult, WorkerProgress } from '@/lib/types';

/**
 * Worker de ingestão: parse → normalização → deduplicação.
 *
 * Vive fora da thread principal porque ler 10.000 registros RIS e rodar a similaridade de
 * títulos leva segundos — tempo em que a UI ficaria congelada se isso rodasse inline.
 */

export type ProgressCallback = (progress: WorkerProgress) => void;

const api = {
  /**
   * Lê os arquivos enviados e devolve a base normalizada.
   *
   * Os `ArrayBuffer` chegam por transferência (zero-copy); o dataset volta por cópia
   * estruturada, que é o custo inevitável de trazer objetos para a thread principal.
   */
  ingest(files: UploadedFile[], onProgress?: ProgressCallback): Dataset {
    onProgress?.({ phase: 'Lendo arquivos', ratio: 0 });

    // Reserva os últimos 5% para o tique final de consolidação — as fases reais de
    // cada arquivo (leitura, enriquecimento, padronização; ver processFiles/
    // processRisFiles) já preenchem os outros 95% com progresso proporcional.
    const dataset = processFiles(files, (progress) => {
      onProgress?.({ ...progress, ratio: progress.ratio * 0.95 });
    });

    onProgress?.({ phase: 'Consolidando estrutura', ratio: 1, detail: `${dataset.length} documentos` });

    // O teto anunciado na UI existe para proteger a memória do navegador; documentos
    // além dele são descartados em vez de travarem a aba mais adiante.
    return dataset.length > MAX_DOCUMENTS ? dataset.slice(0, MAX_DOCUMENTS) : dataset;
  },

  /**
   * Atalho para fontes que já se sabe serem RIS — a base de demonstração.
   * Evita o roteamento por extensão e a conversão para ArrayBuffer.
   */
  ingestRis(sources: RisSource[], onProgress?: ProgressCallback): Dataset {
    const dataset = processRisFiles(sources, onProgress);
    return dataset.length > MAX_DOCUMENTS ? dataset.slice(0, MAX_DOCUMENTS) : dataset;
  },

  dedupByDoi(dataset: Dataset): DedupResult {
    return dedupByDoi(dataset);
  },

  dedupBySimilarity(
    dataset: Dataset,
    threshold: number,
    onProgress?: ProgressCallback,
  ): DedupResult {
    return dedupBySimilarity(dataset, {
      threshold,
      onProgress: (ratio) => onProgress?.({ phase: 'Comparando títulos', ratio }),
    });
  },

  dedupBoth(
    dataset: Dataset,
    threshold: number,
    onProgress?: ProgressCallback,
  ): DedupResult {
    onProgress?.({ phase: 'Deduplicando por DOI', ratio: 0.1 });
    return dedupBoth(dataset, {
      threshold,
      onProgress: (ratio) =>
        onProgress?.({
          phase: 'Comparando títulos por similaridade',
          ratio: 0.2 + ratio * 0.8,
        }),
    });
  },
};

export type IngestWorkerApi = typeof api;

Comlink.expose(api);
