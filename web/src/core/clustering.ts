import { kmeans } from 'ml-kmeans';

import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { fitTransform } from './ml/tfidf';
import { silhouetteScore } from './ml/silhouette';
import { randomizedSvd } from './ml/svd';
import { collectColumns, pickColumn } from './text';

/**
 * Agrupamento temático de documentos — ⇄ `categorizar_temas_por_cluster` (utils.py:1146).
 *
 * O pipeline reproduz o do Python: TF-IDF → LSA → varredura de `k` pelo Silhouette →
 * K-Means definitivo. Só a nomeação dos grupos usa o Gemini, e ela acontece fora daqui.
 *
 * ATENÇÃO — a versão Python não funciona. Ela constrói o `TfidfVectorizer` com
 * `token_pattern=None` e sem `tokenizer`, o que levanta `TypeError` no scikit-learn. Ao
 * contrário da deduplicação, aqui não há `try/except`: a categorização temática levanta
 * exceção no app atual. Esta implementação é, portanto, a primeira que de fato roda.
 *
 * A atribuição dos clusters NÃO é reproduzível a partir do Python: o `random_state=42` do
 * scikit-learn depende do gerador do NumPy, que não existe em JavaScript. Aqui a semente é
 * própria e fixa, então execuções repetidas sobre a mesma base dão o mesmo resultado.
 */

/** Amostra mínima para que o agrupamento tenha algum significado. */
const MIN_DOCUMENTS = 5;

/** Dimensões do LSA — o `n_components=50` do Python. */
const LSA_COMPONENTS = 50;

/** Vocabulário máximo do TF-IDF — o `max_features=1500` do Python. */
const MAX_FEATURES = 1500;

/** Semente única para LSA e K-Means, garantindo reprodutibilidade. */
const SEED = 42;

export interface ClusteringOptions {
  /** Teto de agrupamentos testados. */
  maxClusters?: number;
  onProgress?: (ratio: number, phase: string) => void;
}

export interface ClusterSample {
  clusterId: number;
  size: number;
  /** Documentos representativos, para o modelo nomear o tema. */
  samples: { title: string; abstract: string }[];
  /** Termos mais característicos do agrupamento, por peso médio no TF-IDF. */
  topTerms: string[];
}

export interface ClusteringResult {
  /** Cluster de cada documento, na ordem do dataset. */
  assignments: number[];
  clusterCount: number;
  silhouette: number;
  clusters: ClusterSample[];
}

/** Quantos documentos por cluster são enviados ao modelo para nomeação. */
const SAMPLES_PER_CLUSTER = 5;
const TOP_TERMS_PER_CLUSTER = 8;
const SAMPLE_ABSTRACT_LIMIT = 600;

/**
 * Agrupa os documentos por similaridade textual e devolve amostras de cada grupo.
 *
 * Devolve `null` quando a base é pequena demais para que o agrupamento signifique algo.
 */
export function clusterDocuments(
  rows: Dataset,
  options: ClusteringOptions = {},
): ClusteringResult | null {
  if (rows.length < MIN_DOCUMENTS) return null;

  const allColumns = collectColumns(rows);
  const titleColumn = pickColumn(allColumns, FIELD_CANDIDATES.title);
  const keywordsColumn = pickColumn(allColumns, FIELD_CANDIDATES.keywords);

  // Texto combinado: título, palavras-chave e resumo — ⇄ `TEXTO_COMBINADO` do Python.
  const texts = rows.map((doc) =>
    [
      titleColumn ? String(doc[titleColumn] ?? '') : '',
      keywordsColumn ? String(doc[keywordsColumn] ?? '') : '',
      String(doc[FIELD.ABSTRACT] ?? ''),
    ]
      .filter(Boolean)
      .join(' '),
  );

  options.onProgress?.(0.1, 'Vetorizando textos');
  const model = fitTransform(texts, { stopWords: true, maxFeatures: MAX_FEATURES });
  if (model.vocabulary.size < 2) return null;

  options.onProgress?.(0.3, 'Comprimindo semântica (LSA)');
  const embedding = randomizedSvd(model.vectors, model.vocabulary.size, {
    components: LSA_COMPONENTS,
    seed: SEED,
  });

  // Varredura do k ótimo pelo Silhouette — ⇄ utils.py:1197.
  const maxClusters = Math.min(options.maxClusters ?? 10, rows.length - 1);
  let bestK = 2;
  let bestScore = -1;
  let bestLabels: number[] = [];

  for (let k = 2; k <= maxClusters; k += 1) {
    options.onProgress?.(
      0.35 + (0.5 * (k - 1)) / Math.max(maxClusters - 1, 1),
      `Testando ${k} agrupamentos`,
    );

    const attempt = kmeans(embedding, k, {
      seed: SEED,
      initialization: 'kmeans++',
      maxIterations: 100,
    });

    const score = silhouetteScore(embedding, attempt.clusters);
    if (score > bestScore) {
      bestScore = score;
      bestK = k;
      bestLabels = attempt.clusters;
    }
  }

  // Com apenas dois agrupamentos possíveis a varredura não roda; calcula direto.
  if (bestLabels.length === 0) {
    const attempt = kmeans(embedding, 2, {
      seed: SEED,
      initialization: 'kmeans++',
      maxIterations: 100,
    });
    bestLabels = attempt.clusters;
    bestScore = silhouetteScore(embedding, bestLabels);
    bestK = 2;
  }

  options.onProgress?.(0.9, 'Selecionando documentos representativos');

  return {
    assignments: bestLabels,
    clusterCount: bestK,
    silhouette: bestScore,
    clusters: describeClusters(rows, texts, bestLabels, bestK, model, titleColumn),
  };
}

/** Monta a descrição de cada agrupamento: tamanho, amostras e termos característicos. */
function describeClusters(
  rows: Dataset,
  texts: readonly string[],
  labels: readonly number[],
  clusterCount: number,
  model: ReturnType<typeof fitTransform>,
  titleColumn: string | null,
): ClusterSample[] {
  const inverseVocabulary = new Map([...model.vocabulary].map(([term, index]) => [index, term]));
  const clusters: ClusterSample[] = [];

  for (let clusterId = 0; clusterId < clusterCount; clusterId += 1) {
    const members: number[] = [];
    for (let i = 0; i < labels.length; i += 1) {
      if (labels[i] === clusterId) members.push(i);
    }
    if (members.length === 0) continue;

    // Termos característicos: maior peso TF-IDF médio dentro do agrupamento.
    const weightByTerm = new Map<number, number>();
    for (const index of members) {
      const vector = model.vectors[index];
      if (!vector) continue;
      for (let position = 0; position < vector.indices.length; position += 1) {
        const term = vector.indices[position] as number;
        weightByTerm.set(term, (weightByTerm.get(term) ?? 0) + (vector.values[position] as number));
      }
    }

    const topTerms = [...weightByTerm.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, TOP_TERMS_PER_CLUSTER)
      .map(([term]) => inverseVocabulary.get(term) ?? '')
      .filter(Boolean);

    // Amostras: os documentos com mais texto, que é o critério do Python — mais texto
    // significa mais contexto para o modelo entender do que o agrupamento trata.
    const samples = [...members]
      .sort((left, right) => (texts[right]?.length ?? 0) - (texts[left]?.length ?? 0))
      .slice(0, SAMPLES_PER_CLUSTER)
      .map((index) => {
        const doc = rows[index] as SimetricsDoc;
        const abstract = String(doc[FIELD.ABSTRACT] ?? '');
        return {
          title: titleColumn ? String(doc[titleColumn] ?? '') : '',
          abstract:
            abstract.length > SAMPLE_ABSTRACT_LIMIT
              ? `${abstract.slice(0, SAMPLE_ABSTRACT_LIMIT)}…`
              : abstract || 'Sem resumo',
        };
      });

    clusters.push({ clusterId, size: members.length, samples, topTerms });
  }

  return clusters;
}

/** Rótulo usado quando a nomeação por IA falha para um agrupamento. */
export function fallbackThemeName(cluster: ClusterSample): string {
  if (cluster.topTerms.length === 0) return `Tema ${cluster.clusterId + 1}`;
  // Sem o modelo, os próprios termos característicos já descrevem o grupo.
  return cluster.topTerms
    .slice(0, 3)
    .map((term) => term.charAt(0).toUpperCase() + term.slice(1))
    .join(', ');
}

/** Aplica os nomes de tema aos documentos, devolvendo uma nova base. */
export function applyThemes(
  rows: Dataset,
  assignments: readonly number[],
  names: ReadonlyMap<number, string>,
): Dataset {
  return rows.map((doc, index) => ({
    ...doc,
    [FIELD.THEME]: names.get(assignments[index] ?? -1) ?? 'Outros/Não Categorizado',
  }));
}
