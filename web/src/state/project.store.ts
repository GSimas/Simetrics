import { create } from 'zustand';

import { buildSearchOptions } from '@/core/search';
import { downloadBlob, timestampedFilename } from '@/core/export';
import {
  encodeProjectEnvelope,
  parseProjectEnvelope,
  type ProjectMeta,
  type ProjectRecord,
} from '@/lib/project';
import {
  deleteProject as deleteProjectRecord,
  getAllProjectMeta,
  getProject,
  putProject,
} from '@/lib/project-db';
import { useLocale } from './locale.store';
import { DERIVED_RESET, useDataset, type DatasetSourceFile } from './dataset.store';

/**
 * Camada de "Projetos" sobre o `dataset.store`: lista leve de projetos salvos + o
 * checkpoint automático que os mantém atualizados. Não guarda os datasets em memória —
 * só o `dataset.store` guarda a base ativa; este store guarda metadados e orquestra o
 * IndexedDB (lib/project-db.ts).
 */

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

interface ProjectState {
  projects: ProjectMeta[];
  activeProjectId: string | null;
  isLoadingList: boolean;
  saveStatus: SaveStatus;
  lastSavedAt: string | null;
  error: string | null;

  refreshList: () => Promise<void>;
  open: (id: string) => Promise<void>;
  rename: (id: string, name: string) => Promise<void>;
  duplicate: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  exportToFile: (id: string) => Promise<void>;
  importFromFile: (file: File) => Promise<void>;
  clearError: () => void;
}

function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function isDatasetBusy(): boolean {
  const state = useDataset.getState();
  return state.isIngesting || state.isDeduplicating || state.isCategorizingThemes;
}

function deriveDefaultName(sourceFiles: DatasetSourceFile[]): string {
  const t = useLocale.getState().t;
  if (sourceFiles.length === 0) return t('project_untitled');
  const base = sourceFiles[0]!.name.replace(/\.[^./]+$/, '');
  return sourceFiles.length > 1 ? `${base} +${sourceFiles.length - 1}` : base;
}

/**
 * Grava o estado atual de `dataset.store` como o projeto ativo — cria um rascunho na
 * primeira vez (nenhum `activeProjectId` ainda), atualiza nas seguintes. Preserva
 * `name`/`createdAt` do projeto já existente lendo `projects` (a lista leve já reflete
 * o último checkpoint, não precisa reler o registro completo do IndexedDB).
 */
async function checkpoint(): Promise<void> {
  const ds = useDataset.getState();
  if (!ds.active || !ds.original) return;

  const { activeProjectId, projects } = useProjectStore.getState();
  const existingMeta = activeProjectId ? projects.find((p) => p.id === activeProjectId) : undefined;
  const now = new Date().toISOString();
  const id = activeProjectId ?? crypto.randomUUID();

  const record: ProjectRecord = {
    id,
    schemaVersion: 1,
    name: existingMeta?.name ?? deriveDefaultName(ds.sourceFiles),
    createdAt: existingMeta?.createdAt ?? now,
    updatedAt: now,
    sourceFiles: ds.sourceFiles,
    dedupStrategy: ds.dedupStrategy,
    dedupThreshold: ds.dedupThreshold,
    original: ds.original,
    active: ds.active,
    duplicates: ds.duplicates,
    clustering: ds.clustering,
  };

  useProjectStore.setState({ saveStatus: 'saving' });

  try {
    await putProject(record);
    useProjectStore.setState({
      activeProjectId: id,
      saveStatus: 'saved',
      lastSavedAt: now,
      error: null,
    });
    await useProjectStore.getState().refreshList();
  } catch (cause) {
    const quota = cause instanceof DOMException && cause.name === 'QuotaExceededError';
    const t = useLocale.getState().t;
    useProjectStore.setState({
      saveStatus: 'error',
      error: quota ? t('project_save_quota_error') : t('project_save_error'),
    });
  }
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  projects: [],
  activeProjectId: null,
  isLoadingList: false,
  saveStatus: 'idle',
  lastSavedAt: null,
  error: null,

  async refreshList() {
    set({ isLoadingList: true });
    try {
      const projects = await getAllProjectMeta();
      set({ projects });
    } catch (cause) {
      set({ error: describeError(cause) });
    } finally {
      set({ isLoadingList: false });
    }
  },

  async open(id) {
    if (isDatasetBusy()) {
      set({ error: useLocale.getState().t('project_busy_error') });
      return;
    }

    let record: ProjectRecord | undefined;
    try {
      record = await getProject(id);
    } catch (cause) {
      set({ error: describeError(cause) });
      return;
    }

    if (!record) {
      set({ error: useLocale.getState().t('project_not_found_error') });
      return;
    }

    // Marca o projeto como ativo ANTES de tocar no dataset.store: o assinante de
    // checkpoint reage à troca de `active` que vem a seguir, e precisa encontrar o
    // `activeProjectId` já correto para não criar um rascunho novo por engano.
    set({ activeProjectId: id, saveStatus: 'saved', lastSavedAt: record.updatedAt, error: null });

    useDataset.setState({
      original: record.original,
      active: record.active,
      duplicates: record.duplicates,
      dedupStrategy: record.dedupStrategy,
      dedupThreshold: record.dedupThreshold,
      sourceFiles: record.sourceFiles,
      ...DERIVED_RESET,
      clustering: record.clustering,
      searchOptions: buildSearchOptions(record.active),
      error: null,
    });
  },

  async rename(id, name) {
    const trimmed = name.trim();
    if (!trimmed) return;

    try {
      const record = await getProject(id);
      if (!record) return;
      await putProject({ ...record, name: trimmed, updatedAt: new Date().toISOString() });
      await get().refreshList();
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async duplicate(id) {
    try {
      const record = await getProject(id);
      if (!record) return;

      const now = new Date().toISOString();
      const copy: ProjectRecord = {
        ...record,
        id: crypto.randomUUID(),
        name: `${record.name} ${useLocale.getState().t('project_copy_suffix')}`,
        createdAt: now,
        updatedAt: now,
      };
      await putProject(copy);
      await get().refreshList();
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async remove(id) {
    try {
      await deleteProjectRecord(id);
      if (get().activeProjectId === id) {
        set({ activeProjectId: null, saveStatus: 'idle', lastSavedAt: null });
      }
      await get().refreshList();
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async exportToFile(id) {
    try {
      const record = await getProject(id);
      if (!record) return;

      const envelope = encodeProjectEnvelope(record);
      downloadBlob(
        timestampedFilename(record.name, 'json'),
        new Blob([JSON.stringify(envelope, null, 2)], { type: 'application/json' }),
      );
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  async importFromFile(file) {
    let raw: unknown;
    try {
      raw = JSON.parse(await file.text());
    } catch {
      set({ error: useLocale.getState().t('project_import_invalid_json') });
      return;
    }

    let parsed: ProjectRecord;
    try {
      parsed = parseProjectEnvelope(raw);
    } catch (cause) {
      set({ error: describeError(cause) });
      return;
    }

    // Todo import é tratado como um projeto novo e independente — nunca sobrescreve
    // silenciosamente um registro local com o mesmo id do arquivo exportado.
    const existingNames = new Set(get().projects.map((p) => p.name));
    let name = parsed.name;
    while (existingNames.has(name)) {
      name = `${name} ${useLocale.getState().t('project_import_suffix')}`;
    }

    try {
      await putProject({ ...parsed, id: crypto.randomUUID(), name });
      await get().refreshList();
    } catch (cause) {
      set({ error: describeError(cause) });
    }
  },

  clearError() {
    set({ error: null });
  },
}));

useDataset.subscribe(
  (state) => state.active,
  (active, prevActive) => {
    if (active && active !== prevActive) void checkpoint();
    if (!active) {
      useProjectStore.setState({ activeProjectId: null, saveStatus: 'idle', lastSavedAt: null });
    }
  },
);
