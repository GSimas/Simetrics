import type { ClusteringResult } from '@/core/clustering';
import type { DatabaseName } from '@/lib/schema';
import type { Dataset, DuplicateRecord } from '@/lib/types';

/**
 * Um "Projeto" é o snapshot salvável/exportável de uma base carregada no Simetrics.
 *
 * Guarda só o que é caro ou impossível de refazer sem os arquivos originais — `original`
 * (a dedup sempre parte dela, nunca do `active` anterior), `active`/`duplicates` (a
 * deduplicação por similaridade roda TF-IDF, não é barata) e `clustering` (custou
 * chamadas ao Gemini). As análises derivadas (overview, tabelas, SNA, rede) ficam de
 * fora de propósito: são baratas e o `dataset.store` já as recalcula sob demanda.
 */

export const PROJECT_SCHEMA_VERSION = 1;

/** Mesmos valores de `DedupStrategy` (state/dataset.store.ts), duplicado aqui de
 * propósito para `lib/project.ts` não depender de `state/` — união estrutural, então
 * qualquer `DedupStrategy` é atribuível aqui sem conversão. */
export type ProjectDedupStrategy = 'none' | 'doi' | 'similarity' | 'both';

export interface ProjectSourceFile {
  name: string;
  database: DatabaseName | string;
}

export interface ProjectRecord {
  id: string;
  schemaVersion: number;
  name: string;
  createdAt: string;
  updatedAt: string;
  sourceFiles: ProjectSourceFile[];
  dedupStrategy: ProjectDedupStrategy;
  dedupThreshold: number | null;
  original: Dataset;
  active: Dataset;
  duplicates: DuplicateRecord[];
  clustering: ClusteringResult | null;
}

/** Campos leves para listar projetos sem desserializar datasets inteiros. */
export type ProjectMeta = Pick<
  ProjectRecord,
  'id' | 'name' | 'createdAt' | 'updatedAt' | 'sourceFiles' | 'dedupStrategy'
> & { docCount: number };

export interface ProjectEnvelope {
  kind: 'simetrics-project';
  schemaVersion: number;
  exportedAt: string;
  project: ProjectRecord;
}

const ENVELOPE_KIND = 'simetrics-project';

export function toProjectMeta(project: ProjectRecord): ProjectMeta {
  return {
    id: project.id,
    name: project.name,
    createdAt: project.createdAt,
    updatedAt: project.updatedAt,
    sourceFiles: project.sourceFiles,
    dedupStrategy: project.dedupStrategy,
    docCount: project.active.length,
  };
}

/** Monta o envelope exportável — usado pelo chamador junto de `downloadBlob`/
 * `timestampedFilename` (core/export.ts) para virar um arquivo `.json` baixável. */
export function encodeProjectEnvelope(project: ProjectRecord): ProjectEnvelope {
  return {
    kind: ENVELOPE_KIND,
    schemaVersion: PROJECT_SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    project,
  };
}

function isDataset(value: unknown): value is Dataset {
  return Array.isArray(value);
}

/**
 * Valida e normaliza um envelope de projeto importado. Rejeita qualquer JSON que não
 * seja reconhecidamente um export do Simetrics, com mensagens específicas — mesmo
 * tratamento descritivo que o resto do app usa para arquivo de entrada inválido.
 *
 * Deliberadamente NÃO decide política de importação (id novo, nome duplicado): isso é
 * responsabilidade de quem chama (`project.store.ts`), que tem acesso à lista de
 * projetos já salvos. Esta função só garante que o formato é válido e utilizável.
 */
export function parseProjectEnvelope(raw: unknown): ProjectRecord {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Arquivo inválido: não é um JSON de projeto do Simetrics.');
  }

  const envelope = raw as Partial<ProjectEnvelope>;

  if (envelope.kind !== ENVELOPE_KIND) {
    throw new Error('Arquivo inválido: não é um projeto exportado do Simetrics.');
  }

  if (typeof envelope.schemaVersion !== 'number') {
    throw new Error('Arquivo inválido: versão do projeto ausente.');
  }

  const project = envelope.project as Partial<ProjectRecord> | undefined;
  if (!project || !isDataset(project.original) || !isDataset(project.active)) {
    throw new Error('Arquivo inválido: dados do projeto ausentes ou corrompidos.');
  }

  switch (envelope.schemaVersion) {
    case 1:
      return normalizeV1(project);
    default:
      throw new Error(
        `Versão de projeto não suportada (${envelope.schemaVersion}). Atualize o Simetrics.`,
      );
  }
}

/** Preenche campos ausentes com padrões seguros — defesa contra JSON editado à mão ou
 * de uma versão futura que tenha adicionado campos opcionais. */
function normalizeV1(project: Partial<ProjectRecord>): ProjectRecord {
  const now = new Date().toISOString();
  return {
    id: typeof project.id === 'string' && project.id ? project.id : crypto.randomUUID(),
    schemaVersion: PROJECT_SCHEMA_VERSION,
    name: typeof project.name === 'string' && project.name.trim() ? project.name : 'Projeto importado',
    createdAt: typeof project.createdAt === 'string' ? project.createdAt : now,
    updatedAt: typeof project.updatedAt === 'string' ? project.updatedAt : now,
    sourceFiles: Array.isArray(project.sourceFiles) ? project.sourceFiles : [],
    dedupStrategy: project.dedupStrategy ?? 'none',
    dedupThreshold: typeof project.dedupThreshold === 'number' ? project.dedupThreshold : null,
    original: project.original as Dataset,
    active: project.active as Dataset,
    duplicates: Array.isArray(project.duplicates) ? project.duplicates : [],
    clustering: project.clustering ?? null,
  };
}
