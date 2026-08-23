import { ENGLISH_STOP_WORDS } from './stop-words';

/**
 * TF-IDF esparso compatível com o `TfidfVectorizer` do scikit-learn.
 *
 * Substitui o sklearn no navegador. Os padrões seguem os do sklearn — `smooth_idf=True`,
 * `norm='l2'`, `sublinear_tf=False` — porque é o que o pipeline Python usa.
 */

/** `token_pattern` padrão do sklearn: r"(?u)\b\w\w+\b" — dois ou mais caracteres de palavra. */
const TOKEN_PATTERN = /[\p{L}\p{N}_]{2,}/gu;

export interface TfidfOptions {
  /** Remove stop words em inglês, como `stop_words='english'`. */
  stopWords?: boolean;
  /** Faixa de n-gramas, inclusiva nas duas pontas. Padrão (1, 1). */
  ngramRange?: [number, number];
  /** Frequência mínima de documento para o termo entrar no vocabulário. */
  minDf?: number;
  /** Mantém apenas os N termos mais frequentes do corpus. */
  maxFeatures?: number;
}

/** Vetor esparso de um documento: índices de termo e pesos, já normalizados em L2. */
export interface SparseVector {
  indices: Int32Array;
  values: Float64Array;
}

export interface TfidfModel {
  vocabulary: Map<string, number>;
  idf: Float64Array;
  vectors: SparseVector[];
}

/** Tokeniza como o sklearn: minúsculas, `\b\w\w+\b`, stop words e depois n-gramas. */
export function tokenize(text: string, options: TfidfOptions = {}): string[] {
  const lowered = text.toLowerCase();
  const unigrams: string[] = [];

  TOKEN_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_PATTERN.exec(lowered)) !== null) {
    const token = match[0];
    // O sklearn descarta stop words ANTES de formar n-gramas.
    if (options.stopWords && ENGLISH_STOP_WORDS.has(token)) continue;
    unigrams.push(token);
  }

  const [minN, maxN] = options.ngramRange ?? [1, 1];
  if (minN === 1 && maxN === 1) return unigrams;

  const grams: string[] = [];
  for (let n = minN; n <= maxN; n += 1) {
    for (let i = 0; i + n <= unigrams.length; i += 1) {
      grams.push(unigrams.slice(i, i + n).join(' '));
    }
  }
  return grams;
}

/** Ajusta o vocabulário e transforma o corpus em vetores esparsos normalizados. */
export function fitTransform(
  documents: readonly string[],
  options: TfidfOptions = {},
): TfidfModel {
  const tokenized = documents.map((text) => tokenize(text, options));

  // Frequência de documento por termo.
  const documentFrequency = new Map<string, number>();
  for (const tokens of tokenized) {
    for (const token of new Set(tokens)) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
  }

  const minDf = options.minDf ?? 1;
  let terms = [...documentFrequency.entries()].filter(([, df]) => df >= minDf);

  if (options.maxFeatures !== undefined && terms.length > options.maxFeatures) {
    // O sklearn ordena por frequência total decrescente e desempata pelo termo.
    const totalFrequency = new Map<string, number>();
    for (const tokens of tokenized) {
      for (const token of tokens) {
        totalFrequency.set(token, (totalFrequency.get(token) ?? 0) + 1);
      }
    }
    // A métrica é idêntica à do sklearn (frequência total no corpus), mas o desempate não:
    // o sklearn usa `np.argsort`, um introsort instável, e com muitos termos empatados no
    // valor de corte a escolha dele é arbitrária. Desempatamos alfabeticamente — determinístico
    // e reproduzível. Só afeta `maxFeatures`, usado apenas no clustering temático.
    terms.sort((left, right) => {
      const delta = (totalFrequency.get(right[0]) ?? 0) - (totalFrequency.get(left[0]) ?? 0);
      return delta !== 0 ? delta : left[0] < right[0] ? -1 : left[0] > right[0] ? 1 : 0;
    });
    terms = terms.slice(0, options.maxFeatures);
  }

  // O vocabulário do sklearn é indexado em ordem alfabética dos termos.
  terms.sort((left, right) => (left[0] < right[0] ? -1 : left[0] > right[0] ? 1 : 0));

  const vocabulary = new Map<string, number>();
  const idf = new Float64Array(terms.length);
  const documentCount = documents.length;

  terms.forEach(([term, df], index) => {
    vocabulary.set(term, index);
    // smooth_idf=True: ln((1 + n) / (1 + df)) + 1
    idf[index] = Math.log((1 + documentCount) / (1 + df)) + 1;
  });

  const vectors = tokenized.map((tokens) => vectorize(tokens, vocabulary, idf));
  return { vocabulary, idf, vectors };
}

/** Constrói o vetor esparso L2-normalizado de um documento já tokenizado. */
function vectorize(
  tokens: readonly string[],
  vocabulary: ReadonlyMap<string, number>,
  idf: Float64Array,
): SparseVector {
  const counts = new Map<number, number>();
  for (const token of tokens) {
    const index = vocabulary.get(token);
    if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
  }

  const indices = new Int32Array(counts.size);
  const values = new Float64Array(counts.size);

  let position = 0;
  let squaredNorm = 0;
  // Índices em ordem crescente, como a matriz CSR do scipy.
  for (const index of [...counts.keys()].sort((a, b) => a - b)) {
    const weight = (counts.get(index) as number) * (idf[index] as number);
    indices[position] = index;
    values[position] = weight;
    squaredNorm += weight * weight;
    position += 1;
  }

  const norm = Math.sqrt(squaredNorm);
  if (norm > 0) {
    for (let i = 0; i < values.length; i += 1) values[i] = (values[i] as number) / norm;
  }

  return { indices, values };
}

/** Transforma um texto novo usando um vocabulário já ajustado. */
export function transformOne(
  text: string,
  model: TfidfModel,
  options: TfidfOptions = {},
): SparseVector {
  return vectorize(tokenize(text, options), model.vocabulary, model.idf);
}

export interface SimilarPair {
  a: number;
  b: number;
  score: number;
}

/**
 * Pares de documentos com similaridade de cosseno acima do limiar, com `a < b`.
 *
 * Usa índice invertido em vez do produto matricial completo. Essa é a diferença entre
 * viável e inviável: em 10.000 documentos, comparar todos os pares são 50 milhões de
 * operações que travam a aba. O índice invertido só visita pares que compartilham ao
 * menos um termo — que é exatamente o que a multiplicação esparsa do scipy faz por baixo
 * dos panos no Python.
 *
 * Como os vetores estão normalizados em L2, o produto escalar já é o cosseno.
 */
export function findSimilarPairs(
  vectors: readonly SparseVector[],
  threshold: number,
  onProgress?: (ratio: number) => void,
): SimilarPair[] {
  // Termo -> documentos que o contêm, com o peso correspondente.
  const postings = new Map<number, { doc: number; weight: number }[]>();
  for (let doc = 0; doc < vectors.length; doc += 1) {
    const vector = vectors[doc] as SparseVector;
    for (let i = 0; i < vector.indices.length; i += 1) {
      const term = vector.indices[i] as number;
      let list = postings.get(term);
      if (!list) postings.set(term, (list = []));
      list.push({ doc, weight: vector.values[i] as number });
    }
  }

  const pairs: SimilarPair[] = [];
  const accumulator = new Float64Array(vectors.length);
  const touched: number[] = [];

  for (let doc = 0; doc < vectors.length; doc += 1) {
    const vector = vectors[doc] as SparseVector;
    touched.length = 0;

    for (let i = 0; i < vector.indices.length; i += 1) {
      const term = vector.indices[i] as number;
      const weight = vector.values[i] as number;
      const list = postings.get(term);
      if (!list) continue;

      for (const posting of list) {
        // Só a metade superior: evita computar cada par duas vezes.
        if (posting.doc <= doc) continue;
        if (accumulator[posting.doc] === 0) touched.push(posting.doc);
        accumulator[posting.doc] = (accumulator[posting.doc] as number) + weight * posting.weight;
      }
    }

    for (const other of touched) {
      const score = accumulator[other] as number;
      accumulator[other] = 0;
      if (score >= threshold) pairs.push({ a: doc, b: other, score });
    }

    if (onProgress && doc % 250 === 0) onProgress(doc / vectors.length);
  }

  return pairs;
}
