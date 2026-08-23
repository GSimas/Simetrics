import { currentYear } from '@/lib/schema';
import type { ScientometricIndices } from '@/lib/types';
import { pyRound } from './stats';
import { toNumeric } from './text';

/**
 * Índices h, g, i10 e m — ⇄ `extrair_indices_cientometricos` (utils.py:22).
 *
 * Fonte única de verdade da matemática cientométrica, exatamente como no Python:
 * toda tabela de entidade (autores, países, venues, keywords) chama esta função em vez
 * de recalcular por conta própria.
 *
 * @param citations  Citações brutas dos documentos da entidade; não-números viram 0.
 * @param years      Anos de publicação, para o índice m. Sem anos, m = 0.
 * @param baseYear   Ano de referência do índice m.
 */
export function computeIndices(
  citations: readonly unknown[],
  years?: readonly unknown[] | null,
  baseYear: number = currentYear(),
): ScientometricIndices {
  // `pd.to_numeric(errors='coerce').fillna(0).astype(int)` — truncando para inteiro.
  const sorted = citations
    .map((value) => {
      const numeric = toNumeric(value);
      return numeric === null ? 0 : Math.trunc(numeric);
    })
    .sort((a, b) => b - a);

  // h: maior posição i em que o i-ésimo documento tem ao menos i citações.
  let h = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    if ((sorted[i] as number) >= i + 1) h = i + 1;
    else break;
  }

  // g: maior posição em que a soma acumulada alcança o quadrado da posição.
  // O Python encerra no primeiro valor que falha, e não no maior g global — mantido.
  let g = 0;
  let runningTotal = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    runningTotal += sorted[i] as number;
    if (runningTotal >= (i + 1) ** 2) g = i + 1;
    else break;
  }

  let i10 = 0;
  for (const value of sorted) {
    if (value >= 10) i10 += 1;
  }

  // m: h dividido pelos anos de atuação, contados de forma inclusiva.
  let m = 0;
  if (years && years.length > 0) {
    let firstYear = Number.POSITIVE_INFINITY;
    for (const value of years) {
      const numeric = toNumeric(value);
      if (numeric !== null) firstYear = Math.min(firstYear, Math.trunc(numeric));
    }

    if (Number.isFinite(firstYear)) {
      const activeYears = baseYear - firstYear + 1;
      if (activeYears > 0) m = pyRound(h / activeYears, 3);
    }
  }

  return { h, g, i10, m };
}

/**
 * Lei de Lotka — ⇄ `plot_lotkas_law` (utils.py:1763).
 * Devolve a distribuição observada de produtividade e a curva teórica `c / x²`.
 */
export interface LotkaDistribution {
  observed: { articles: number; frequency: number }[];
  theoretical: { articles: number; frequency: number }[];
}

export function lotkaDistribution(authorDocCounts: readonly number[]): LotkaDistribution | null {
  if (authorDocCounts.length === 0) return null;

  // Quantos autores escreveram exatamente X artigos.
  const authorsPerArticleCount = new Map<number, number>();
  for (const count of authorDocCounts) {
    authorsPerArticleCount.set(count, (authorsPerArticleCount.get(count) ?? 0) + 1);
  }

  const articleCounts = [...authorsPerArticleCount.keys()].sort((a, b) => a - b);
  let totalAuthors = 0;
  for (const count of authorsPerArticleCount.values()) totalAuthors += count;

  const observed = articleCounts.map((articles) => ({
    articles,
    frequency: (authorsPerArticleCount.get(articles) as number) / totalAuthors,
  }));

  // Lotka assume expoente 2; `c` normaliza a soma das probabilidades no domínio observado.
  const maxArticles = articleCounts[articleCounts.length - 1] as number;
  let inverseSquareSum = 0;
  for (let x = 1; x <= maxArticles; x += 1) inverseSquareSum += 1 / x ** 2;
  const c = 1 / inverseSquareSum;

  const theoretical = Array.from({ length: maxArticles }, (_, index) => {
    const articles = index + 1;
    return { articles, frequency: c / articles ** 2 };
  });

  return { observed, theoretical };
}
