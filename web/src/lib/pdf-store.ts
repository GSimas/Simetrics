import { useLocale } from '@/state/locale.store';

import { deletePdfFromCloud, downloadPdfFromCloud, uploadPdfToCloud } from './cloud-sync';
import type { LocalWriteOptions } from './project-db';

/**
 * PDFs dos textos completos, num banco próprio do IndexedDB, endereçados pelo SHA-256 do
 * arquivo. Separados dos projetos de propósito: um projeto (e o arquivo de decisões trocado
 * entre revisores) não pode inchar com dezenas de MB de artigos nem levar um artigo
 * protegido por direito autoral junto sem que se perceba. Quem recebe um projeto sem os
 * PDFs anexa a própria cópia, e o hash a reconhece.
 *
 * Junto do arquivo vai o texto de cada página, extraído uma vez — a IA e a busca do trecho
 * não precisam abrir o PDF de novo.
 */

const DB_NAME = 'simetrics-pdfs';
const DB_VERSION = 1;
const STORE = 'files';

export interface StoredPdf {
  hash: string;
  name: string;
  data: Blob;
  pageTexts: string[];
  storedAt: string;
}

function fail(pt: string, en: string): Error {
  return new Error(useLocale.getState().locale === 'en' ? en : pt);
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  dbPromise ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE, { keyPath: 'hash' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? fail('Falha ao abrir o banco de PDFs.', 'Failed to open the PDF database.'));
  });
  return dbPromise;
}

function run<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const request = action(db.transaction(STORE, mode).objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? fail('Falha no armazenamento de PDFs.', 'PDF storage operation failed.'));
      }),
  );
}

export async function sha256(data: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function putPdf(pdf: StoredPdf, options: LocalWriteOptions = {}): Promise<IDBValidKey> {
  // PDFs ocupam espaço: pede ao navegador que não apague este banco sob pressão de disco.
  void navigator.storage?.persist?.().catch(() => false);
  const key = await run('readwrite', (store) => store.put(pdf));
  if (!options.localOnly) uploadPdfToCloud(pdf);
  return key;
}

/** Sem o PDF neste navegador, busca a cópia da conta (quando logado) e a guarda aqui. */
export async function getPdf(hash: string, options: LocalWriteOptions = {}): Promise<StoredPdf | undefined> {
  const local = await run('readonly', (store) => store.get(hash) as IDBRequest<StoredPdf | undefined>);
  if (local || options.localOnly) return local;
  const remote = await downloadPdfFromCloud(hash).catch(() => undefined);
  if (remote) await putPdf(remote, { localOnly: true });
  return remote;
}

export async function getAllPdfHashes(): Promise<string[]> {
  return (await run('readonly', (store) => store.getAllKeys())) as string[];
}

export async function deletePdf(hash: string): Promise<undefined> {
  await run('readwrite', (store) => store.delete(hash));
  deletePdfFromCloud(hash);
  return undefined;
}

/** Espaço usado e disponível para o site, quando o navegador informa. */
export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  const estimate = await navigator.storage?.estimate?.().catch(() => null);
  return estimate?.usage !== undefined && estimate.quota !== undefined ? { usage: estimate.usage, quota: estimate.quota } : null;
}
