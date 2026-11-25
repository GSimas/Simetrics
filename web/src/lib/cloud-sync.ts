import { create } from 'zustand';

import { encodeProjectEnvelope, parseProjectEnvelope, toProjectMeta, type ProjectRecord } from './project';
import { deleteProject, getAllProjectMeta, getProject, putProject } from './project-db';
import { getAllPdfHashes, getPdf, type StoredPdf } from './pdf-store';
import { CLOUD_BUCKET, supabase } from './supabase';

/**
 * Projetos e PDFs na nuvem para quem entrou na conta. O IndexedDB continua sendo o
 * armazenamento principal (o app funciona igual sem conta); aqui só se espelha cada
 * gravação local no Supabase e, ao entrar, se reconcilia os dois lados pelo `updatedAt`.
 *
 * ponytail: último a gravar vence, por projeto inteiro. Dois aparelhos editando o mesmo
 * projeto ao mesmo tempo perdem as edições do mais antigo; se isso virar uso real, trocar
 * por versão/etag no `simetrics_projects` e recusar gravação sobre versão velha.
 */

export type CloudStatus = 'off' | 'syncing' | 'synced' | 'error';

export const useCloudSync = create<{ status: CloudStatus; lastSyncAt: string | null }>(() => ({
  status: 'off',
  lastSyncAt: null,
}));

let userId: string | null = null;

/** Chamado pelo auth.store a cada troca de sessão. */
export function setCloudUser(id: string | null): void {
  userId = id;
  if (!id) useCloudSync.setState({ status: 'off', lastSyncAt: null });
}

const projectPath = (uid: string, id: string) => `${uid}/projects/${id}.json.gz`;
const pdfPath = (uid: string, hash: string, ext: 'pdf' | 'json') => `${uid}/pdfs/${hash}.${ext}`;

// Ids que já estiveram na nuvem nesta conta: sumiram de lá = excluídos em outro aparelho.
const syncedKey = (uid: string) => `simetrics-cloud-synced:${uid}`;
function readSynced(uid: string): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(syncedKey(uid)) ?? '[]') as string[]);
  } catch {
    return new Set();
  }
}
function markSynced(uid: string, id: string, synced: boolean): void {
  const ids = readSynced(uid);
  if (synced) ids.add(id);
  else ids.delete(id);
  try {
    localStorage.setItem(syncedKey(uid), JSON.stringify([...ids]));
  } catch {
    // Sem armazenamento: a próxima reconciliação só não detecta exclusões remotas.
  }
}

async function gzip(text: string): Promise<Blob> {
  const stream = new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Response(stream).blob();
}
async function gunzip(blob: Blob): Promise<string> {
  return new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

function fail(message: string, cause: unknown): never {
  useCloudSync.setState({ status: 'error' });
  throw cause instanceof Error ? cause : new Error(`${message}: ${String(cause)}`);
}

async function uploadProject(uid: string, record: ProjectRecord): Promise<void> {
  const body = await gzip(JSON.stringify(encodeProjectEnvelope(record)));
  const { error: uploadError } = await supabase.storage
    .from(CLOUD_BUCKET)
    .upload(projectPath(uid, record.id), body, { upsert: true, contentType: 'application/gzip' });
  if (uploadError) fail('upload do projeto', uploadError);
  const meta = toProjectMeta(record);
  const row = {
    id: meta.id,
    user_id: uid,
    name: meta.name,
    created_at: meta.createdAt,
    updated_at: meta.updatedAt,
    doc_count: meta.docCount,
    source_files: meta.sourceFiles,
    dedup_strategy: meta.dedupStrategy,
  };
  // A chave é (user_id, id): o mesmo id pode existir em outra conta usada neste navegador.
  const { error } = await supabase.from('simetrics_projects').upsert(row, { onConflict: 'user_id,id' });
  if (error) fail('metadados do projeto', error);
  markSynced(uid, record.id, true);
}

async function downloadProject(uid: string, id: string): Promise<ProjectRecord> {
  const { data, error } = await supabase.storage.from(CLOUD_BUCKET).download(projectPath(uid, id));
  if (error || !data) fail('download do projeto', error);
  return parseProjectEnvelope(JSON.parse(await gunzip(data)));
}

// Checkpoints chegam a cada edição da revisão; sobe só a versão mais recente, agrupada.
const UPLOAD_DELAY_MS = 4000;
const pendingUploads = new Map<string, ReturnType<typeof setTimeout>>();

/** Depois de cada gravação local de projeto. */
export function scheduleProjectUpload(id: string): void {
  if (!userId) return;
  clearTimeout(pendingUploads.get(id));
  pendingUploads.set(
    id,
    setTimeout(() => {
      pendingUploads.delete(id);
      const uid = userId;
      if (!uid) return;
      useCloudSync.setState({ status: 'syncing' });
      void getProject(id)
        .then((record) => record && uploadProject(uid, record))
        .then(() => useCloudSync.setState({ status: 'synced', lastSyncAt: new Date().toISOString() }))
        .catch((cause) => console.error('Simetrics: falha ao enviar projeto à nuvem', cause));
    }, UPLOAD_DELAY_MS),
  );
}

/** Depois de excluir um projeto localmente. */
export function deleteCloudProject(id: string): void {
  const uid = userId;
  if (!uid) return;
  clearTimeout(pendingUploads.get(id));
  pendingUploads.delete(id);
  void Promise.all([
    supabase.storage.from(CLOUD_BUCKET).remove([projectPath(uid, id)]),
    supabase.from('simetrics_projects').delete().eq('user_id', uid).eq('id', id),
  ])
    .then(() => markSynced(uid, id, false))
    .catch((cause) => console.error('Simetrics: falha ao excluir projeto da nuvem', cause));
}

/**
 * Ao sair da conta: sobe o que ainda não chegou à nuvem e tira deste navegador os projetos
 * dessa conta. Assim quem entrar depois no mesmo navegador não herda nem envia para a
 * própria conta os projetos de outra pessoa; eles descem de novo quando a dona voltar.
 * Projetos que não conseguiram subir ficam aqui, para não se perderem.
 * ponytail: os PDFs locais ficam (são endereçados por hash e não aparecem sem o projeto).
 */
export async function releaseAccount(): Promise<string[]> {
  const uid = userId;
  if (!uid) return [];
  for (const timer of pendingUploads.values()) clearTimeout(timer);
  pendingUploads.clear();

  const synced = readSynced(uid);
  const local = (await getAllProjectMeta()).filter((meta) => synced.has(meta.id));
  const { data: rows, error } = await supabase.from('simetrics_projects').select('id, updated_at');
  if (error) throw error;
  const remote = new Map(rows.map((row) => [row.id as string, row.updated_at as string]));

  const keep = new Set<string>();
  for (const { id, action } of planSync(local, remote, new Set())) {
    if (action !== 'upload') continue;
    const record = await getProject(id);
    if (record) await uploadProject(uid, record).catch(() => keep.add(id));
  }
  const removed = local.map((meta) => meta.id).filter((id) => !keep.has(id));
  for (const id of removed) await deleteProject(id, { localOnly: true });
  return removed;
}

/**
 * "Excluir meus dados do Simetrics": apaga da nuvem todos os projetos e PDFs da conta e
 * tira deste navegador os projetos dela. A conta Scientata em si continua (é compartilhada
 * com as outras aplicações Scientata). Os PDFs locais ficam, como em `releaseAccount`.
 */
export async function deleteAccountData(): Promise<void> {
  const uid = userId;
  if (!uid) return;
  for (const timer of pendingUploads.values()) clearTimeout(timer);
  pendingUploads.clear();

  const bucket = supabase.storage.from(CLOUD_BUCKET);
  for (const folder of ['projects', 'pdfs']) {
    // ponytail: até 1000 arquivos por pasta, como no `syncAll`; paginar se alguém passar disso.
    const { data, error } = await bucket.list(`${uid}/${folder}`, { limit: 1000 });
    if (error) throw error;
    const paths = (data ?? []).map((file) => `${uid}/${folder}/${file.name}`);
    if (paths.length > 0) {
      const { error: removeError } = await bucket.remove(paths);
      if (removeError) throw removeError;
    }
  }
  const { error } = await supabase.from('simetrics_projects').delete().eq('user_id', uid);
  if (error) throw error;

  for (const id of readSynced(uid)) await deleteProject(id, { localOnly: true });
  try {
    localStorage.removeItem(syncedKey(uid));
  } catch {
    // Sem armazenamento: nada a limpar.
  }
}

async function uploadPdf(uid: string, pdf: StoredPdf): Promise<void> {
  const bucket = supabase.storage.from(CLOUD_BUCKET);
  const sidecar = JSON.stringify({ name: pdf.name, pageTexts: pdf.pageTexts, storedAt: pdf.storedAt });
  const results = await Promise.all([
    bucket.upload(pdfPath(uid, pdf.hash, 'pdf'), pdf.data, { upsert: true, contentType: 'application/pdf' }),
    bucket.upload(pdfPath(uid, pdf.hash, 'json'), new Blob([sidecar]), { upsert: true, contentType: 'application/json' }),
  ]);
  const error = results.find((result) => result.error)?.error;
  if (error) throw error;
}

/** Depois de guardar um PDF localmente. PDFs acima do limite do bucket (50 MB) ficam só aqui. */
export function uploadPdfToCloud(pdf: StoredPdf): void {
  const uid = userId;
  if (!uid) return;
  void uploadPdf(uid, pdf).catch((cause) => console.error('Simetrics: falha ao enviar PDF à nuvem', cause));
}

/** PDF ausente neste navegador: busca a cópia da conta, se houver. */
export async function downloadPdfFromCloud(hash: string): Promise<StoredPdf | undefined> {
  const uid = userId;
  if (!uid) return undefined;
  const bucket = supabase.storage.from(CLOUD_BUCKET);
  const [file, meta] = await Promise.all([
    bucket.download(pdfPath(uid, hash, 'pdf')),
    bucket.download(pdfPath(uid, hash, 'json')),
  ]);
  if (file.error || meta.error || !file.data || !meta.data) return undefined;
  const sidecar = JSON.parse(await meta.data.text()) as Pick<StoredPdf, 'name' | 'pageTexts' | 'storedAt'>;
  return { hash, data: file.data, ...sidecar };
}

export function deletePdfFromCloud(hash: string): void {
  const uid = userId;
  if (!uid) return;
  void supabase.storage
    .from(CLOUD_BUCKET)
    .remove([pdfPath(uid, hash, 'pdf'), pdfPath(uid, hash, 'json')])
    .catch((cause) => console.error('Simetrics: falha ao excluir PDF da nuvem', cause));
}

export type SyncAction = 'upload' | 'download' | 'delete-local';

/**
 * Decide o destino de cada projeto: o lado com `updatedAt` mais novo vence; projeto só
 * local sobe (inclusive os criados antes do login), projeto só na nuvem desce, e projeto
 * que já esteve na nuvem (`synced`) e sumiu de lá foi excluído em outro aparelho.
 */
export function planSync(
  local: readonly { id: string; updatedAt: string }[],
  remote: ReadonlyMap<string, string>,
  synced: ReadonlySet<string>,
): { id: string; action: SyncAction }[] {
  const plan: { id: string; action: SyncAction }[] = [];
  for (const { id, updatedAt } of local) {
    const remoteAt = remote.get(id);
    if (remoteAt === undefined) plan.push({ id, action: synced.has(id) ? 'delete-local' : 'upload' });
    else if (Date.parse(updatedAt) > Date.parse(remoteAt)) plan.push({ id, action: 'upload' });
    else if (Date.parse(remoteAt) > Date.parse(updatedAt)) plan.push({ id, action: 'download' });
  }
  const localIds = new Set(local.map((meta) => meta.id));
  for (const id of remote.keys()) if (!localIds.has(id)) plan.push({ id, action: 'download' });
  return plan;
}

/**
 * Reconciliação ao entrar na conta (regras em `planSync`). o lado com `updatedAt` mais novo vence; projeto só
 * local sobe (inclusive os criados antes do login), projeto só na nuvem desce, e projeto
 * que já esteve na nuvem e sumiu de lá foi excluído em outro aparelho.
 * Devolve se algo mudou localmente (a lista de projetos precisa recarregar).
 */
export async function syncAll(): Promise<boolean> {
  const uid = userId;
  if (!uid) return false;
  useCloudSync.setState({ status: 'syncing' });
  let changedLocally = false;
  try {
    const { data: rows, error } = await supabase.from('simetrics_projects').select('id, updated_at');
    if (error) fail('lista de projetos', error);
    const remote = new Map(rows.map((row) => [row.id as string, row.updated_at as string]));
    const local = await getAllProjectMeta();

    for (const { id, action } of planSync(local, remote, readSynced(uid))) {
      if (action === 'upload') {
        const record = await getProject(id);
        if (record) await uploadProject(uid, record);
        continue;
      }
      if (action === 'delete-local') {
        await deleteProject(id, { localOnly: true });
        markSynced(uid, id, false);
      } else {
        await putProject(await downloadProject(uid, id), { localOnly: true });
        markSynced(uid, id, true);
      }
      changedLocally = true;
    }

    // PDFs: sobem os que só existem aqui; os da nuvem descem sob demanda (getPdf).
    // ponytail: lista até 1000 PDFs por conta; paginar `list` se alguém passar disso.
    const { data: files } = await supabase.storage.from(CLOUD_BUCKET).list(`${uid}/pdfs`, { limit: 1000 });
    const inCloud = new Set((files ?? []).filter((f) => f.name.endsWith('.pdf')).map((f) => f.name.slice(0, -4)));
    for (const hash of await getAllPdfHashes()) {
      if (inCloud.has(hash)) continue;
      const pdf = await getPdf(hash, { localOnly: true });
      if (pdf) await uploadPdf(uid, pdf).catch((cause) => console.error('Simetrics: PDF não enviado', cause));
    }

    useCloudSync.setState({ status: 'synced', lastSyncAt: new Date().toISOString() });
  } catch (cause) {
    useCloudSync.setState({ status: 'error' });
    console.error('Simetrics: falha na sincronização com a nuvem', cause);
  }
  return changedLocally;
}
