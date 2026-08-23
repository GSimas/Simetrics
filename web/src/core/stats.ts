/**
 * Estatística com semântica idêntica à do pandas, usada pelas tabelas analíticas.
 */

const FLOAT_BUFFER = new DataView(new ArrayBuffer(8));

/**
 * Replica `round(x, digits)` do Python.
 *
 * Duas diferenças em relação ao `toFixed` do JavaScript tornam isso necessário:
 *
 * 1. O Python desempata para o dígito PAR; o `toFixed` desempata para longe do zero.
 * 2. Ambos operam sobre o valor binário exato do double, não sobre a representação
 *    decimal curta — por isso `round(2.675, 2)` é `2.67` nas duas linguagens.
 *
 * O empate exato não é hipotético: ele acontece sempre que o valor é um racional
 * diádico terminando no dígito seguinte, como `5 / 8 = 0.625`. Aí o Python devolve
 * `0.62` e o `toFixed` devolve `0.63` — divergência silenciosa numa média de citações
 * de um autor com 8 documentos.
 *
 * A implementação reconstrói o double como a fração exata `m * 2^e` e faz a divisão
 * em BigInt, então não há erro de arredondamento intermediário.
 */
export function pyRound(value: number, digits = 0): number {
  if (!Number.isFinite(value)) return value;
  if (Number.isInteger(value) && digits >= 0) return value;

  FLOAT_BUFFER.setFloat64(0, value);
  const bits = FLOAT_BUFFER.getBigUint64(0);

  const sign = bits >> 63n === 1n ? -1n : 1n;
  const rawExponent = Number((bits >> 52n) & 0x7ffn);
  const rawMantissa = bits & 0xf_ffff_ffff_ffffn;

  // Subnormais têm expoente implícito 1 e sem bit oculto.
  const mantissa = rawExponent === 0 ? rawMantissa : rawMantissa | 0x10_0000_0000_0000n;
  const exponent = (rawExponent === 0 ? 1 : rawExponent) - 1075;

  const scale = 10n ** BigInt(Math.abs(digits));

  // |value| * 10^digits  ==  numerator / denominator, exatamente.
  let numerator = digits >= 0 ? mantissa * scale : mantissa;
  let denominator = digits >= 0 ? 1n : scale;

  if (exponent >= 0) numerator <<= BigInt(exponent);
  else denominator <<= BigInt(-exponent);

  let quotient = numerator / denominator;
  const remainder = numerator % denominator;
  const twiceRemainder = remainder * 2n;

  if (twiceRemainder > denominator) {
    quotient += 1n;
  } else if (twiceRemainder === denominator && quotient % 2n === 1n) {
    // Empate exato: sobe apenas se o quociente for ímpar (ties-to-even).
    quotient += 1n;
  }

  const signed = Number(sign * quotient);
  return digits >= 0 ? signed / Number(scale) : signed * Number(scale);
}

/** Média aritmética. Devolve NaN para série vazia, como o pandas. */
export function mean(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

/** Mediana; com tamanho par, média dos dois centrais — igual ao pandas. */
export function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;

  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;

  if (sorted.length % 2 === 1) return sorted[middle] as number;
  return ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

/**
 * Desvio padrão AMOSTRAL (ddof=1) — o padrão do `Series.std()` do pandas.
 * O `np.std` usa ddof=0; trocar um pelo outro desloca todas as colunas de
 * "Desvio Padrão de Citações" das tabelas analíticas.
 */
export function std(values: readonly number[]): number {
  if (values.length < 2) return Number.NaN;

  const average = mean(values);
  let sumSquares = 0;
  for (const value of values) {
    const delta = value - average;
    sumSquares += delta * delta;
  }
  return Math.sqrt(sumSquares / (values.length - 1));
}

/** Desvio padrão populacional (ddof=0) — o `np.std`, usado nas métricas de rede. */
export function stdPopulation(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;

  const average = mean(values);
  let sumSquares = 0;
  for (const value of values) {
    const delta = value - average;
    sumSquares += delta * delta;
  }
  return Math.sqrt(sumSquares / values.length);
}

export function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

/**
 * Correlação de Spearman — ⇄ `Series.corr(method='spearman')`.
 * Usa postos médios em caso de empate, como o pandas/scipy.
 */
export function spearman(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length || a.length < 2) return Number.NaN;

  const rankA = averageRanks(a);
  const rankB = averageRanks(b);

  const meanA = mean(rankA);
  const meanB = mean(rankB);

  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;

  for (let i = 0; i < rankA.length; i += 1) {
    const deltaA = (rankA[i] as number) - meanA;
    const deltaB = (rankB[i] as number) - meanB;
    covariance += deltaA * deltaB;
    varianceA += deltaA * deltaA;
    varianceB += deltaB * deltaB;
  }

  const denominator = Math.sqrt(varianceA * varianceB);
  return denominator === 0 ? Number.NaN : covariance / denominator;
}

/** Postos 1..n com média nos empates (o "average" do `Series.rank()`). */
function averageRanks(values: readonly number[]): number[] {
  const order = values
    .map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value);

  const ranks = new Array<number>(values.length).fill(0);

  let position = 0;
  while (position < order.length) {
    let end = position;
    while (
      end + 1 < order.length &&
      (order[end + 1] as { value: number }).value === (order[position] as { value: number }).value
    ) {
      end += 1;
    }

    const sharedRank = (position + end) / 2 + 1;
    for (let i = position; i <= end; i += 1) {
      ranks[(order[i] as { index: number }).index] = sharedRank;
    }

    position = end + 1;
  }

  return ranks;
}

/** Regressão linear simples — ⇄ `np.polyfit(x, y, 1)`. Devolve [inclinação, intercepto]. */
export function linearFit(x: readonly number[], y: readonly number[]): [number, number] {
  const n = Math.min(x.length, y.length);
  if (n < 2) return [0, 0];

  const meanX = mean(x.slice(0, n));
  const meanY = mean(y.slice(0, n));

  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < n; i += 1) {
    const deltaX = (x[i] as number) - meanX;
    numerator += deltaX * ((y[i] as number) - meanY);
    denominator += deltaX * deltaX;
  }

  const slope = denominator === 0 ? 0 : numerator / denominator;
  return [slope, meanY - slope * meanX];
}
