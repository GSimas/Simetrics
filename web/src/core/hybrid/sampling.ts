import type { HybridDoc } from './documents';

/**
 * Amostragem estratificada para a descoberta de categorias.
 *
 * Uma amostra aleatória simples de ~120 documentos pode não conter nenhum exemplar de um
 * tema que ocupa 3% da base — e o que o DeepSeek não vê, ele não nomeia. Por isso os
 * estratos combinam o agrupamento do k-means (que já existe no app) com o período de
 * publicação (os temas mudam ao longo do tempo), e uma fração da amostra é reservada a
 * documentos atípicos do próprio agrupamento, que tendem a ser os temas minoritários.
 *
 * A semente é fixa e registrada na execução: a mesma base produz a mesma amostra.
 */

export interface SampleOptions {
  size: number;
  seed: number;
  /** Faixas de período (por quantis de ano). */
  periods?: number;
  /** Fração da amostra reservada a documentos atípicos. */
  outlierShare?: number;
}

export interface SampleResult {
  /** Posições dos documentos amostrados, em ordem crescente. */
  indices: number[];
  strata: number;
  outliers: number;
}

/** Texto mínimo para o documento ser útil ao modelo que descobre as categorias. */
const MIN_TEXT_LENGTH = 40;

/** Gerador pseudoaleatório com semente (mulberry32) — determinístico e rápido. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j] as T, out[i] as T];
  }
  return out;
}

function docText(doc: HybridDoc): string {
  return `${doc.title} ${doc.keywords} ${doc.abstract}`;
}

/** Faixa de período de cada documento, por quantis do ano. `-1` para sem ano. */
function periodOf(docs: readonly HybridDoc[], periods: number): number[] {
  const years = docs
    .map((doc) => doc.year)
    .filter((year): year is number => year !== null)
    .sort((a, b) => a - b);

  if (years.length === 0 || periods <= 1) return docs.map((doc) => (doc.year === null ? -1 : 0));

  const cuts: number[] = [];
  for (let p = 1; p < periods; p += 1) {
    cuts.push(years[Math.min(years.length - 1, Math.floor((years.length * p) / periods))] as number);
  }

  return docs.map((doc) => {
    if (doc.year === null) return -1;
    let period = 0;
    while (period < cuts.length && (doc.year as number) >= (cuts[period] as number)) period += 1;
    return period;
  });
}

/**
 * Quão típico o documento é do seu agrupamento: fração dos termos característicos do
 * agrupamento que aparecem no texto. Os menos típicos são os candidatos a tema minoritário.
 */
function typicality(doc: HybridDoc, terms: readonly string[]): number {
  if (terms.length === 0) return 1;
  const haystack = docText(doc).toLowerCase();
  let hits = 0;
  for (const term of terms) if (haystack.includes(term.toLowerCase())) hits += 1;
  return hits / terms.length;
}

/**
 * Distribui `budget` vagas entre estratos: uma para cada (os maiores primeiro, se não
 * couber), depois o restante proporcional ao tamanho, pelo método dos maiores restos.
 */
function allocate(sizes: readonly number[], budget: number): number[] {
  const allocation = sizes.map(() => 0);
  const order = sizes.map((_, i) => i).sort((a, b) => (sizes[b] as number) - (sizes[a] as number));

  let remaining = budget;
  for (const i of order) {
    if (remaining === 0) break;
    if ((sizes[i] as number) > 0) {
      allocation[i] = 1;
      remaining -= 1;
    }
  }

  const capacity = (i: number) => (sizes[i] as number) - (allocation[i] as number);
  while (remaining > 0) {
    const open = order.filter((i) => capacity(i) > 0);
    if (open.length === 0) break;
    const total = open.reduce((sum, i) => sum + capacity(i), 0);
    const quotas = open.map((i) => ({ i, exact: (capacity(i) / total) * remaining }));
    let given = 0;
    for (const quota of quotas) {
      const whole = Math.min(Math.floor(quota.exact), capacity(quota.i));
      allocation[quota.i] = (allocation[quota.i] as number) + whole;
      given += whole;
    }
    remaining -= given;
    const byRemainder = quotas
      .filter((quota) => capacity(quota.i) > 0)
      .sort((a, b) => (b.exact % 1) - (a.exact % 1));
    for (const quota of byRemainder) {
      if (remaining === 0) break;
      allocation[quota.i] = (allocation[quota.i] as number) + 1;
      remaining -= 1;
    }
    if (given === 0 && byRemainder.length === 0) break;
  }

  return allocation;
}

/**
 * Sorteia a amostra estratificada.
 *
 * @param assignments Agrupamento do k-means para cada documento (mesma ordem de `docs`),
 *   ou `null` quando não houver — nesse caso só o período estratifica.
 * @param clusterTerms Termos característicos de cada agrupamento, para achar os atípicos.
 */
export function stratifiedSample(
  docs: readonly HybridDoc[],
  assignments: readonly number[] | null,
  clusterTerms: ReadonlyMap<number, readonly string[]>,
  options: SampleOptions,
): SampleResult {
  const random = seededRandom(options.seed);
  const periods = periodOf(docs, options.periods ?? 3);
  const clusterOf = (i: number) => assignments?.[i] ?? 0;

  const strataKeys = new Map<string, number[]>();
  for (const doc of docs) {
    const key = `${clusterOf(doc.index)}|${periods[doc.index]}`;
    const members = strataKeys.get(key);
    if (members) members.push(doc.index);
    else strataKeys.set(key, [doc.index]);
  }

  // Base pequena: a amostra é a base inteira, e a validação já é a classificação final.
  if (docs.length <= options.size) {
    return { indices: docs.map((doc) => doc.index), strata: strataKeys.size, outliers: 0 };
  }

  const picked = new Set<number>();
  const usable = (i: number) => docText(docs[i] as HybridDoc).trim().length >= MIN_TEXT_LENGTH;

  // 1. Atípicos: os de menor tipicidade em cada agrupamento, alternando entre agrupamentos.
  const outlierBudget = assignments ? Math.round(options.size * (options.outlierShare ?? 0.1)) : 0;
  if (outlierBudget > 0) {
    const byCluster = new Map<number, number[]>();
    for (const doc of docs) {
      if (!usable(doc.index)) continue;
      const cluster = clusterOf(doc.index);
      const members = byCluster.get(cluster);
      if (members) members.push(doc.index);
      else byCluster.set(cluster, [doc.index]);
    }
    const queues = [...byCluster.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([cluster, members]) => {
        const terms = clusterTerms.get(cluster) ?? [];
        return shuffle(members, random).sort(
          (a, b) => typicality(docs[a] as HybridDoc, terms) - typicality(docs[b] as HybridDoc, terms),
        );
      });
    let round = 0;
    while (picked.size < outlierBudget && queues.some((queue) => queue.length > round)) {
      for (const queue of queues) {
        if (picked.size >= outlierBudget) break;
        const candidate = queue[round];
        if (candidate !== undefined) picked.add(candidate);
      }
      round += 1;
    }
  }
  const outliers = picked.size;

  // 2. Estratos: vagas proporcionais, preferindo documentos com texto suficiente.
  const strata = [...strataKeys.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  const available = strata.map(([, members]) => members.filter((i) => !picked.has(i)));
  const allocation = allocate(
    available.map((members) => members.length),
    Math.max(0, options.size - picked.size),
  );

  available.forEach((members, s) => {
    const ordered = shuffle(members, random).sort((a, b) => Number(usable(b)) - Number(usable(a)));
    for (const i of ordered.slice(0, allocation[s])) picked.add(i);
  });

  return { indices: [...picked].sort((a, b) => a - b), strata: strata.length, outliers };
}
