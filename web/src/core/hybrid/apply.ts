import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { tierOf } from './metrics';
import {
  OTHER_CATEGORY_ID,
  type ConfidenceThresholds,
  type ConfidenceTier,
  type DocumentDecision,
  type HybridCategory,
} from './types';

/** Valor gravado em `TEMA_STATUS` para cada faixa. */
export const TIER_STATUS: Record<ConfidenceTier, string> = {
  auto: 'auto',
  review: 'revisar',
  unclassified: 'nao_classificado',
};

export function unclassifiedName(locale: 'pt' | 'en'): string {
  return locale === 'pt' ? 'Não classificado' : 'Unclassified';
}

/**
 * Nome da opção "nenhuma categoria serve". Nunca vira um tema "Outros": o documento que
 * cai nela fica não classificado, igual ao de baixa confiança.
 */
export function otherCategoryName(locale: 'pt' | 'en'): string {
  return unclassifiedName(locale);
}

/**
 * Grava o resultado na base: o tema no mesmo campo que o k-means usa (`TEMA_GEMINI`), para
 * que todas as análises por tema — produção, boxplot, QL, busca, chat — funcionem sem
 * mudança, mais a confiança e a faixa em campos próprios.
 */
export function applyHybridThemes(
  rows: Dataset,
  decisions: ReadonlyMap<number, DocumentDecision>,
  categories: readonly HybridCategory[],
  thresholds: ConfidenceThresholds,
  locale: 'pt' | 'en',
): Dataset {
  const names = new Map(categories.map((category) => [category.id, category.name]));
  const unclassified = unclassifiedName(locale);

  return rows.map((doc, index) => {
    const decision = decisions.get(index);
    if (!decision) {
      return { ...doc, [FIELD.THEME]: unclassified, [FIELD.THEME_CONFIDENCE]: null, [FIELD.THEME_STATUS]: TIER_STATUS.unclassified };
    }
    const tier = decision.categoryId === OTHER_CATEGORY_ID ? 'unclassified' : tierOf(decision, thresholds);
    return {
      ...doc,
      [FIELD.THEME]: tier === 'unclassified' ? unclassified : (names.get(decision.categoryId) ?? unclassified),
      [FIELD.THEME_CONFIDENCE]: Math.round(decision.confidence * 1000) / 1000,
      [FIELD.THEME_STATUS]: TIER_STATUS[tier],
    };
  });
}
