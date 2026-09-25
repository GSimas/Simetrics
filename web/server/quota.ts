import { createHash } from 'node:crypto';

/**
 * Cotas gratuitas por dispositivo.
 *
 * A unidade de cota é uma "pergunta" do Simi ou uma "execução" da classificação híbrida,
 * não uma requisição: uma pergunta pode gerar até quatro chamadas ao modelo (rodadas de
 * ferramentas), e todas descontam uma única vez. O cliente manda um id por unidade; o
 * servidor limita quantas requisições cada unidade pode fazer, para o mesmo id não virar
 * passe livre.
 *
 * O dispositivo é um identificador aleatório guardado no navegador. Na web não existe
 * identificação de dispositivo à prova de fraude — apagar o armazenamento gera um novo id.
 * Por isso há também um teto diário por IP, largo o bastante para laboratórios e
 * universidades que saem pelo mesmo IP. Os dois identificadores são gravados só como hash.
 *
 * O armazenamento não tem operação atômica (Netlify Blobs), então duas requisições
 * simultâneas do mesmo dispositivo podem, no limite, passar uma unidade além da cota. É
 * aceitável para uma cota de cortesia.
 */

export interface QuotaRecord {
  /** Id da unidade → requisições feitas por ela. */
  units: Record<string, number>;
}

export interface QuotaStore {
  read(key: string): Promise<QuotaRecord | null>;
  write(key: string, record: QuotaRecord): Promise<void>;
}

export interface QuotaPolicy {
  scope: string;
  /** Unidades por dispositivo (0 desativa a cota gratuita). */
  deviceLimit: number;
  /** Unidades por IP por dia (0 = sem teto). */
  ipDailyLimit: number;
  maxRequestsPerUnit: number;
}

export interface Identity {
  device: string | null;
  ip: string;
}

export type QuotaDecision =
  | { ok: true; used: number; limit: number; newUnit: boolean }
  | { ok: false; reason: 'device' | 'ip' | 'unit' | 'no-device'; used: number; limit: number };

export function hashId(value: string, salt: string): string {
  return createHash('sha256').update(`${salt}:${value}`).digest('hex').slice(0, 40);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function deviceKey(policy: QuotaPolicy, device: string, salt: string): string {
  return `${policy.scope}/device/${hashId(device, salt)}`;
}

function ipKey(policy: QuotaPolicy, ip: string, salt: string): string {
  return `${policy.scope}/ip/${today()}/${hashId(ip, salt)}`;
}

const count = (record: QuotaRecord | null) => Object.keys(record?.units ?? {}).length;

/** Id de dispositivo/unidade aceitável: curto e só com caracteres seguros. */
export function validId(value: string | null): string | null {
  const trimmed = value?.trim() ?? '';
  return /^[A-Za-z0-9-]{8,64}$/.test(trimmed) ? trimmed : null;
}

export async function readUsage(
  store: QuotaStore,
  policy: QuotaPolicy,
  identity: Identity,
  salt: string,
): Promise<{ used: number; limit: number }> {
  if (!identity.device) return { used: 0, limit: policy.deviceLimit };
  const record = await store.read(deviceKey(policy, identity.device, salt));
  return { used: Math.min(count(record), policy.deviceLimit), limit: policy.deviceLimit };
}

/** Registra uma requisição da unidade, decidindo se ela cabe na cota. */
export async function consume(
  store: QuotaStore,
  policy: QuotaPolicy,
  identity: Identity,
  unit: string,
  salt: string,
): Promise<QuotaDecision> {
  const limit = policy.deviceLimit;
  if (!identity.device) return { ok: false, reason: 'no-device', used: 0, limit };

  const dKey = deviceKey(policy, identity.device, salt);
  const device = (await store.read(dKey)) ?? { units: {} };
  const used = count(device);
  const existing = device.units[unit];

  if (existing !== undefined) {
    if (existing >= policy.maxRequestsPerUnit) return { ok: false, reason: 'unit', used, limit };
    device.units[unit] = existing + 1;
    await store.write(dKey, device);
    return { ok: true, used, limit, newUnit: false };
  }

  if (used >= limit) return { ok: false, reason: 'device', used, limit };

  device.units[unit] = 1;
  if (policy.ipDailyLimit > 0) {
    const iKey = ipKey(policy, identity.ip, salt);
    const ip = (await store.read(iKey)) ?? { units: {} };
    if (count(ip) >= policy.ipDailyLimit) return { ok: false, reason: 'ip', used, limit };
    ip.units[unit] = 1;
    await store.write(iKey, ip);
  }
  await store.write(dKey, device);
  return { ok: true, used: used + 1, limit, newUnit: true };
}

/** Devolve a unidade quando a primeira chamada dela falhou no provedor — não é culpa do usuário. */
export async function refund(
  store: QuotaStore,
  policy: QuotaPolicy,
  identity: Identity,
  unit: string,
  salt: string,
): Promise<void> {
  if (!identity.device) return;
  for (const key of [deviceKey(policy, identity.device, salt), ipKey(policy, identity.ip, salt)]) {
    const record = await store.read(key);
    if (record && unit in record.units) {
      delete record.units[unit];
      await store.write(key, record);
    }
  }
}

/** Armazenamento em memória, opcionalmente persistido num arquivo JSON (desenvolvimento). */
export class MemoryQuotaStore implements QuotaStore {
  private data = new Map<string, QuotaRecord>();

  constructor(
    private readonly persist?: {
      load: () => Record<string, QuotaRecord> | null;
      save: (data: Record<string, QuotaRecord>) => void;
    },
  ) {
    const loaded = persist?.load();
    if (loaded) this.data = new Map(Object.entries(loaded));
  }

  async read(key: string): Promise<QuotaRecord | null> {
    const record = this.data.get(key);
    return record ? { units: { ...record.units } } : null;
  }

  async write(key: string, record: QuotaRecord): Promise<void> {
    this.data.set(key, { units: { ...record.units } });
    this.persist?.save(Object.fromEntries(this.data));
  }
}
