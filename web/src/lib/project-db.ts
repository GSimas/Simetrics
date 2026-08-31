import { toProjectMeta, type ProjectMeta, type ProjectRecord } from './project';

/**
 * Persistência de projetos em IndexedDB — não `localStorage`: um `ProjectRecord` de
 * base grande passa fácil da cota de ~5-10MB do localStorage (10.000 documentos giram
 * em torno de dezenas de MB). Wrapper nativo, sem lib (`idb` etc.): a superfície aqui é
 * pequena e estável, e o resto do app já resolve persistência de plataforma à mão
 * (ai-config.store.ts, locale.store.ts, ThemeToggle.tsx fazem o mesmo com localStorage).
 *
 * Dois object stores, não um: `projects` guarda o registro completo (datasets inclusos);
 * `projectsMeta` guarda só os campos leves de `ProjectMeta`, indexados por `updatedAt`,
 * para a tela inicial listar/ordenar projetos sem desserializar datasets inteiros.
 */

const DB_NAME = 'simetrics-projects';
const DB_VERSION = 1;
const STORE_PROJECTS = 'projects';
const STORE_META = 'projectsMeta';
const INDEX_UPDATED_AT = 'by-updatedAt';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  dbPromise ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_PROJECTS)) {
        db.createObjectStore(STORE_PROJECTS, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_META)) {
        const metaStore = db.createObjectStore(STORE_META, { keyPath: 'id' });
        metaStore.createIndex(INDEX_UPDATED_AT, 'updatedAt');
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Falha ao abrir o banco de projetos.'));
  });

  return dbPromise;
}

function wrapRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Falha na operação de armazenamento local.'));
  });
}

/** Grava o registro completo e sua versão leve numa única transação — ambos os stores
 * ficam sempre em sincronia, nunca um sem o outro. */
export async function putProject(record: ProjectRecord): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([STORE_PROJECTS, STORE_META], 'readwrite');
    tx.objectStore(STORE_PROJECTS).put(record);
    tx.objectStore(STORE_META).put(toProjectMeta(record));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Falha ao salvar o projeto.'));
    tx.onabort = () => reject(tx.error ?? new Error('Operação de salvamento cancelada.'));
  });
}

export async function getProject(id: string): Promise<ProjectRecord | undefined> {
  const db = await openDb();
  return wrapRequest(db.transaction(STORE_PROJECTS, 'readonly').objectStore(STORE_PROJECTS).get(id));
}

/** Mais recentes primeiro, direto do índice — sem carregar datasets nem ordenar em memória. */
export async function getAllProjectMeta(): Promise<ProjectMeta[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_META, 'readonly');
    const index = tx.objectStore(STORE_META).index(INDEX_UPDATED_AT);
    const results: ProjectMeta[] = [];
    const cursorRequest = index.openCursor(null, 'prev');

    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;
      if (cursor) {
        results.push(cursor.value as ProjectMeta);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    cursorRequest.onerror = () => reject(cursorRequest.error ?? new Error('Falha ao listar projetos.'));
  });
}

export async function deleteProject(id: string): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([STORE_PROJECTS, STORE_META], 'readwrite');
    tx.objectStore(STORE_PROJECTS).delete(id);
    tx.objectStore(STORE_META).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Falha ao excluir o projeto.'));
    tx.onabort = () => reject(tx.error ?? new Error('Operação de exclusão cancelada.'));
  });
}
