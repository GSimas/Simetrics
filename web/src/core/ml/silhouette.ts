/**
 * Silhouette score — ⇄ `sklearn.metrics.silhouette_score` (utils.py:1201).
 *
 * Mede quão bem cada ponto se encaixa no seu próprio agrupamento em relação ao vizinho
 * mais próximo:
 *
 *     s(i) = (b(i) − a(i)) / max(a(i), b(i))
 *
 * onde `a(i)` é a distância média aos pontos do mesmo cluster e `b(i)` a menor distância
 * média a um outro cluster. Varia de −1 a 1; próximo de 1 indica agrupamento coeso e bem
 * separado, próximo de 0 indica clusters sobrepostos.
 *
 * É isto que escolhe o número de temas no Simetrics, em vez de fixar um `k` arbitrário.
 */

function euclidean(a: readonly number[], b: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = (a[i] as number) - (b[i] as number);
    total += delta * delta;
  }
  return Math.sqrt(total);
}

/**
 * Silhouette médio de um particionamento.
 *
 * Custa O(n²) em distâncias. Para 10.000 documentos são 50 milhões de cálculos por valor
 * de `k` testado, o que inviabilizaria a varredura — daí o parâmetro `sampleSize`, que
 * estima o score sobre uma amostra determinística, como faz o `sample_size` do sklearn.
 *
 * @param points Coordenadas já reduzidas pelo LSA.
 * @param labels Cluster de cada ponto.
 * @param sampleSize Máximo de pontos considerados. `null` usa todos.
 */
export function silhouetteScore(
  points: readonly (readonly number[])[],
  labels: readonly number[],
  sampleSize: number | null = 2000,
): number {
  const total = points.length;
  if (total < 2) return 0;

  const distinctLabels = new Set(labels);
  // Com um único cluster não há separação a medir; com um cluster por ponto, idem.
  if (distinctLabels.size < 2 || distinctLabels.size >= total) return 0;

  // Amostragem por passo fixo: determinística e bem distribuída sobre a ordem original.
  const indices: number[] = [];
  if (sampleSize !== null && total > sampleSize) {
    const step = total / sampleSize;
    for (let i = 0; i < sampleSize; i += 1) indices.push(Math.floor(i * step));
  } else {
    for (let i = 0; i < total; i += 1) indices.push(i);
  }

  // Membros por cluster, para as médias de distância.
  const membersByLabel = new Map<number, number[]>();
  for (let i = 0; i < total; i += 1) {
    const label = labels[i] as number;
    let bucket = membersByLabel.get(label);
    if (!bucket) membersByLabel.set(label, (bucket = []));
    bucket.push(i);
  }

  let sum = 0;
  let counted = 0;

  for (const i of indices) {
    const point = points[i] as readonly number[];
    const ownLabel = labels[i] as number;
    const ownMembers = membersByLabel.get(ownLabel) as number[];

    // Um ponto sozinho no cluster tem silhouette 0 por convenção.
    if (ownMembers.length <= 1) {
      counted += 1;
      continue;
    }

    let ownDistance = 0;
    for (const j of ownMembers) {
      if (j !== i) ownDistance += euclidean(point, points[j] as readonly number[]);
    }
    ownDistance /= ownMembers.length - 1;

    let nearestOther = Number.POSITIVE_INFINITY;
    for (const [label, members] of membersByLabel) {
      if (label === ownLabel || members.length === 0) continue;

      let distance = 0;
      for (const j of members) distance += euclidean(point, points[j] as readonly number[]);
      distance /= members.length;

      if (distance < nearestOther) nearestOther = distance;
    }

    if (!Number.isFinite(nearestOther)) {
      counted += 1;
      continue;
    }

    const denominator = Math.max(ownDistance, nearestOther);
    sum += denominator === 0 ? 0 : (nearestOther - ownDistance) / denominator;
    counted += 1;
  }

  return counted === 0 ? 0 : sum / counted;
}
