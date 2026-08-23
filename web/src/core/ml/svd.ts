import { Matrix, QrDecomposition, SingularValueDecomposition } from 'ml-matrix';

import { seededRandom } from '../graph/compact';
import type { SparseVector } from './tfidf';

/**
 * Redução de dimensionalidade por SVD truncado — ⇄ `TruncatedSVD` do scikit-learn,
 * usado como LSA (Latent Semantic Analysis) antes do K-Means (utils.py:1186).
 *
 * Por que isso existe no pipeline: a matriz TF-IDF tem milhares de dimensões e é quase
 * toda zeros. Nessa esparsidade, distâncias euclidianas perdem o significado — todos os
 * pontos ficam praticamente equidistantes — e tanto o K-Means quanto o Silhouette
 * deixam de discriminar. Comprimir para ~50 dimensões densas restaura a geometria.
 *
 * O algoritmo é o SVD randomizado (Halko, Martinsson & Tropp, 2011), o mesmo que o
 * scikit-learn usa por padrão: projeta a matriz num subespaço aleatório de dimensão
 * reduzida, onde o SVD exato fica barato.
 */

export interface RandomizedSvdOptions {
  /** Dimensões de saída. */
  components: number;
  /**
   * Colunas extras na projeção aleatória. Amostrar um pouco além do necessário melhora
   * bastante a aproximação dos últimos componentes, ao custo de quase nada.
   */
  oversampling?: number;
  /**
   * Iterações de potência. Cada uma aproxima o subespaço projetado do subespaço dominante
   * real — necessário quando o espectro decai devagar, que é o caso de TF-IDF.
   */
  powerIterations?: number;
  seed?: number;
}

/** Multiplica a matriz esparsa (documentos × termos) por uma densa (termos × k). */
function sparseTimesDense(
  rows: readonly SparseVector[],
  dense: Matrix,
): Matrix {
  const result = Matrix.zeros(rows.length, dense.columns);

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i] as SparseVector;
    for (let position = 0; position < row.indices.length; position += 1) {
      const term = row.indices[position] as number;
      const weight = row.values[position] as number;

      for (let column = 0; column < dense.columns; column += 1) {
        result.set(i, column, result.get(i, column) + weight * dense.get(term, column));
      }
    }
  }

  return result;
}

/** Multiplica a transposta da matriz esparsa (termos × documentos) por uma densa. */
function sparseTransposeTimesDense(
  rows: readonly SparseVector[],
  dense: Matrix,
  featureCount: number,
): Matrix {
  const result = Matrix.zeros(featureCount, dense.columns);

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i] as SparseVector;
    for (let position = 0; position < row.indices.length; position += 1) {
      const term = row.indices[position] as number;
      const weight = row.values[position] as number;

      for (let column = 0; column < dense.columns; column += 1) {
        result.set(term, column, result.get(term, column) + weight * dense.get(i, column));
      }
    }
  }

  return result;
}

/** Amostra normal padrão pelo método de Box-Muller, com PRNG semeado. */
function randomGaussianMatrix(rows: number, columns: number, seed: number): Matrix {
  const random = seededRandom(seed);
  const matrix = Matrix.zeros(rows, columns);

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < columns; j += 1) {
      // `1 - random()` evita log(0), que produziria infinito.
      const u = 1 - random();
      const v = random();
      matrix.set(i, j, Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
    }
  }

  return matrix;
}

/**
 * Projeta documentos esparsos em `components` dimensões densas.
 *
 * @returns Matriz densa (documentos × components), equivalente ao
 *   `TruncatedSVD.fit_transform` do scikit-learn.
 */
export function randomizedSvd(
  rows: readonly SparseVector[],
  featureCount: number,
  options: RandomizedSvdOptions,
): number[][] {
  const documentCount = rows.length;
  const components = Math.min(
    options.components,
    Math.max(1, documentCount - 1),
    Math.max(1, featureCount - 1),
  );

  const oversampling = options.oversampling ?? 10;
  const powerIterations = options.powerIterations ?? 4;
  const projected = Math.min(components + oversampling, documentCount, featureCount);

  // 1. Projeção aleatória: Y = A Ω, com Ω gaussiana (termos × projected).
  const omega = randomGaussianMatrix(featureCount, projected, options.seed ?? 42);
  let y = sparseTimesDense(rows, omega);

  // 2. Iterações de potência: Y ← A (Aᵀ Y), reortogonalizando a cada passo para não
  //    perder posto por erro numérico.
  for (let iteration = 0; iteration < powerIterations; iteration += 1) {
    y = orthonormalize(y);
    const z = orthonormalize(sparseTransposeTimesDense(rows, y, featureCount));
    y = sparseTimesDense(rows, z);
  }

  // 3. Base ortonormal Q do subespaço projetado.
  const q = orthonormalize(y);

  // 4. B = Qᵀ A, uma matriz pequena onde o SVD exato é barato.
  //    Calculamos Bᵀ = Aᵀ Q para reaproveitar o produto esparso já implementado.
  const bTranspose = sparseTransposeTimesDense(rows, q, featureCount);

  // 5. SVD de B. Como temos Bᵀ (termos × projected), os vetores singulares à esquerda de
  //    B são os da direita de Bᵀ.
  const svd = new SingularValueDecomposition(bTranspose, { autoTranspose: true });
  const rightSingular = svd.rightSingularVectors; // (projected × projected)
  const singularValues = svd.diagonal;

  // 6. U Σ = Q · Ṽ · Σ, truncado nos `components` primeiros.
  const embedding: number[][] = [];
  for (let i = 0; i < documentCount; i += 1) {
    const vector: number[] = [];
    for (let k = 0; k < components; k += 1) {
      let value = 0;
      for (let j = 0; j < q.columns; j += 1) {
        value += q.get(i, j) * rightSingular.get(j, k);
      }
      vector.push(value * (singularValues[k] ?? 0));
    }
    embedding.push(vector);
  }

  return embedding;
}

/** Base ortonormal das colunas, via decomposição QR. */
function orthonormalize(matrix: Matrix): Matrix {
  return new QrDecomposition(matrix).orthogonalMatrix;
}
