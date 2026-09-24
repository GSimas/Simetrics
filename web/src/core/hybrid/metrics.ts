import {
  OTHER_CATEGORY_ID,
  type CategoryCount,
  type ConfidenceThresholds,
  type ConfidenceTier,
  type DocumentDecision,
  type HybridCategory,
  type HybridRun,
  type ValidationRound,
} from './types';

/**
 * Métricas da classificação híbrida: concordância entre o rótulo do DeepSeek e a decisão
 * do Jev, roteamento por confiança e cobertura.
 */

/** Faixa do documento. "Outro" com qualquer confiança nunca é aceito automaticamente. */
export function tierOf(decision: DocumentDecision, thresholds: ConfidenceThresholds): ConfidenceTier {
  if (decision.confidence < thresholds.review) return 'unclassified';
  if (decision.confidence < thresholds.accept) return 'review';
  return 'auto';
}

/**
 * Compara rótulos de referência (DeepSeek) com as decisões do Jev.
 *
 * Além da concordância bruta, calcula o kappa de Cohen, que desconta a concordância
 * esperada pelo acaso — com uma categoria dominante, 80% de concordância bruta pode
 * significar muito pouco. É a métrica que um revisor de artigo espera ver.
 */
export function compareLabels(
  reference: ReadonlyMap<number, string>,
  decisions: ReadonlyMap<number, DocumentDecision>,
  categories: readonly HybridCategory[],
  otherName: string,
  taxonomyVersion: number,
): ValidationRound {
  const validIds = new Set([...categories.map((category) => category.id), OTHER_CATEGORY_ID]);
  const pairs: [string, string][] = [];
  for (const [index, label] of reference) {
    const decision = decisions.get(index);
    // Rótulo de uma categoria que o usuário removeu não tem com o que ser comparado.
    if (!decision || !validIds.has(label)) continue;
    pairs.push([label, decision.categoryId]);
  }

  const n = pairs.length;
  const agree = pairs.filter(([a, b]) => a === b).length;
  const observed = n > 0 ? agree / n : 0;

  const refCounts = new Map<string, number>();
  const clsCounts = new Map<string, number>();
  for (const [a, b] of pairs) {
    refCounts.set(a, (refCounts.get(a) ?? 0) + 1);
    clsCounts.set(b, (clsCounts.get(b) ?? 0) + 1);
  }
  let expected = 0;
  for (const [label, count] of refCounts) expected += (count / (n || 1)) * ((clsCounts.get(label) ?? 0) / (n || 1));
  const kappa = n === 0 ? 0 : expected >= 1 ? 1 : (observed - expected) / (1 - expected);

  const nameOf = (id: string) =>
    id === OTHER_CATEGORY_ID ? otherName : (categories.find((category) => category.id === id)?.name ?? id);

  const perCategory = [...refCounts.entries()]
    .map(([categoryId, support]) => ({
      categoryId,
      name: nameOf(categoryId),
      support,
      agreement: pairs.filter(([a, b]) => a === categoryId && b === categoryId).length / support,
    }))
    .sort((a, b) => a.agreement - b.agreement);

  return { taxonomyVersion, compared: n, agreement: observed, kappa, perCategory };
}

/** Categorias cuja descrição provavelmente está ambígua: suporte mínimo e concordância baixa. */
export function weakCategories(round: ValidationRound, minAgreement = 0.6, minSupport = 3): string[] {
  return round.perCategory
    .filter((item) => item.categoryId !== OTHER_CATEGORY_ID)
    .filter((item) => item.support >= minSupport && item.agreement < minAgreement)
    .map((item) => item.categoryId);
}

/** Documentos que sobram: "outro" ou abaixo do limiar de revisão. */
export function leftoverIndices(
  decisions: ReadonlyMap<number, DocumentDecision>,
  thresholds: ConfidenceThresholds,
): number[] {
  const out: number[] = [];
  for (const [index, decision] of decisions) {
    if (decision.categoryId === OTHER_CATEGORY_ID || tierOf(decision, thresholds) === 'unclassified') out.push(index);
  }
  return out;
}

export function coverageOf(
  decisions: ReadonlyMap<number, DocumentDecision>,
  thresholds: ConfidenceThresholds,
): HybridRun['coverage'] {
  const coverage = { total: decisions.size, auto: 0, review: 0, unclassified: 0, other: 0 };
  for (const decision of decisions.values()) {
    if (decision.categoryId === OTHER_CATEGORY_ID) coverage.other += 1;
    coverage[tierOf(decision, thresholds)] += 1;
  }
  return coverage;
}

export function categoryCountsOf(
  decisions: ReadonlyMap<number, DocumentDecision>,
  categories: readonly HybridCategory[],
  otherName: string,
): CategoryCount[] {
  const totals = new Map<string, { documents: number; confidence: number }>();
  for (const decision of decisions.values()) {
    const entry = totals.get(decision.categoryId) ?? { documents: 0, confidence: 0 };
    entry.documents += 1;
    entry.confidence += decision.confidence;
    totals.set(decision.categoryId, entry);
  }
  return [...categories.map((category) => ({ id: category.id, name: category.name })), { id: OTHER_CATEGORY_ID, name: otherName }]
    .map(({ id, name }) => {
      const entry = totals.get(id) ?? { documents: 0, confidence: 0 };
      return {
        categoryId: id,
        name,
        documents: entry.documents,
        meanConfidence: entry.documents > 0 ? entry.confidence / entry.documents : 0,
      };
    })
    .filter((item) => item.documents > 0 || item.categoryId !== OTHER_CATEGORY_ID)
    .sort((a, b) => b.documents - a.documents);
}
