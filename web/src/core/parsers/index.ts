import type { DatabaseName } from '@/lib/schema';
import type { Dataset, WorkerProgress } from '@/lib/types';
import { processCochrane } from './cochrane';
import { processPubmed } from './pubmed';
import { processScopusCsv } from './scopus-csv';
import { processRisFiles } from './pipeline-ris';
import { processWosExcel } from './wos-excel';

export type { RisSource } from './pipeline-ris';

/** Um arquivo enviado pelo usuário, já lido para memória. */
export interface UploadedFile {
  name: string;
  /** Conteúdo binário — necessário para Excel; os demais parsers usam o texto. */
  buffer: ArrayBuffer;
  database: DatabaseName | string;
}

/** Sugere a base de origem a partir da extensão e do nome — ⇄ o `def_idx` de Geral.py:245. */
export function suggestDatabase(fileName: string): DatabaseName {
  const lower = fileName.toLowerCase();

  if (lower.includes('cochrane')) return 'Cochrane';
  if (lower.endsWith('.csv')) return 'Scopus';
  if (lower.endsWith('.xls') || lower.endsWith('.xlsx')) return 'Web of Science';
  if (lower.endsWith('.txt') || lower.endsWith('.nbib')) return 'PubMed';
  return 'Outra';
}

function decode(buffer: ArrayBuffer): string {
  // `errors='ignore'` do Python: bytes inválidos viram U+FFFD em vez de derrubar o parse.
  return new TextDecoder('utf-8').decode(buffer);
}

/**
 * Roteia um arquivo para o parser correto — ⇄ o motor de roteamento de Geral.py:277.
 *
 * A escolha do usuário no seletor de base tem prioridade sobre a extensão, porque o mesmo
 * `.ris` sai de fontes diferentes com convenções incompatíveis.
 */
export function processFile(
  file: UploadedFile,
  onProgress?: (progress: WorkerProgress) => void,
): Dataset {
  const lower = file.name.toLowerCase();

  if (file.database === 'Cochrane') {
    return processCochrane(file.name, decode(file.buffer), onProgress);
  }

  if (file.database === 'PubMed' || lower.endsWith('.txt') || lower.endsWith('.nbib')) {
    return processPubmed(decode(file.buffer), onProgress);
  }

  if (lower.endsWith('.csv')) {
    return processScopusCsv(decode(file.buffer), onProgress);
  }

  if (lower.endsWith('.xls') || lower.endsWith('.xlsx')) {
    return processWosExcel(file.buffer, onProgress);
  }

  // Fallback: RIS genérico (WoS, Mendeley, SciELO).
  return processRisFiles(
    [{ name: file.name, text: decode(file.buffer), database: String(file.database) }],
    onProgress,
  );
}

/**
 * Processa vários arquivos e concatena, marcando a base de origem de cada documento.
 *
 * O progresso de cada arquivo (fase real: leitura, enriquecimento, padronização — ver
 * `processRisFiles`) é reescalado para a fatia `[índice/total, (índice+1)/total]` que
 * esse arquivo ocupa no conjunto, para múltiplos arquivos renderem uma única barra
 * contínua em vez de saltar de 0% a 100% a cada arquivo.
 */
export function processFiles(
  files: readonly UploadedFile[],
  onProgress?: (progress: WorkerProgress) => void,
): Dataset {
  const all: Dataset = [];
  const total = files.length;

  files.forEach((file, index) => {
    const base = index / total;
    const span = 1 / total;

    try {
      const docs = processFile(file, (inner) => {
        onProgress?.({
          phase: inner.phase,
          ratio: base + inner.ratio * span,
          detail: inner.detail ?? file.name,
        });
      });
      for (const doc of docs) {
        doc['BASE DE DADOS'] = file.database;
        all.push(doc);
      }
    } catch {
      // Um arquivo corrompido não deve invalidar os demais — comportamento do Python.
    }
  });

  return all;
}
