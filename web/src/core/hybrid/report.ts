import type { HybridRun } from './types';

/**
 * Texto de métodos da classificação híbrida, pronto para a seção de metodologia de um
 * artigo. Tudo o que é necessário para reproduzir ou auditar o resultado sai do registro
 * da execução: modelos, amostra, semente, limiares, concordância e rodadas de ajuste.
 */

function percent(value: number, locale: 'pt' | 'en'): string {
  return `${(value * 100).toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US', { maximumFractionDigits: 1 })}%`;
}

function decimal(value: number, locale: 'pt' | 'en', digits = 2): string {
  return value.toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function count(value: number, locale: 'pt' | 'en'): string {
  return value.toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US');
}

export function hybridMethodsText(run: HybridRun, locale: 'pt' | 'en'): string {
  const pt = locale === 'pt';
  const firstOrigin = run.taxonomyVersions[0]?.origin;
  const manual = firstOrigin === 'manual';
  const initialCount = run.taxonomyVersions[0]?.categories.length ?? run.finalTaxonomy.length;
  const clarifyRounds = run.taxonomyVersions.filter((version) => version.origin === 'clarify').length;
  const firstRound = run.validation[0];
  const lastRound = run.validation[run.validation.length - 1];
  const { coverage, thresholds, sample } = run;
  const parts: string[] = [];

  if (manual) {
    parts.push(
      pt
        ? `As ${initialCount} categorias temáticas foram definidas pelo pesquisador.`
        : `The ${initialCount} thematic categories were defined by the researcher.`,
    );
  } else {
    const strata = sample.clusterCount > 0
      ? pt
        ? `estratificada por ${sample.clusterCount} agrupamentos K-Means (TF-IDF + LSA) e por período de publicação, com ${sample.outliers} documentos atípicos`
        : `stratified by ${sample.clusterCount} K-Means clusters (TF-IDF + LSA) and publication period, including ${sample.outliers} atypical documents`
      : pt
        ? 'estratificada por período de publicação'
        : 'stratified by publication period';
    parts.push(
      pt
        ? `Uma amostra de ${count(sample.size, locale)} documentos, ${strata} (semente ${sample.seed}), foi analisada pelo modelo de linguagem ${run.models.discovery}, que propôs ${initialCount} categorias temáticas mutuamente exclusivas${run.userEdited ? ', revisadas em seguida pelo pesquisador' : ''}.`
        : `A sample of ${count(sample.size, locale)} documents, ${strata} (seed ${sample.seed}), was analysed by the language model ${run.models.discovery}, which proposed ${initialCount} mutually exclusive thematic categories${run.userEdited ? ', subsequently reviewed by the researcher' : ''}.`,
    );
  }

  parts.push(
    pt
      ? `O modelo de decisão ${run.models.classifier} (TypeSafe) classificou os ${count(coverage.total, locale)} documentos da base a partir do título, das palavras-chave e do resumo, com uma pergunta de escolha única por documento e probabilidade calibrada.`
      : `The decision model ${run.models.classifier} (TypeSafe) classified all ${count(coverage.total, locale)} documents from title, keywords and abstract, using one single-choice question per document with calibrated probability.`,
  );

  if (firstRound && lastRound && firstRound.compared > 0) {
    const rounds = clarifyRounds > 0
      ? pt
        ? ` Após ${clarifyRounds} rodada(s) de esclarecimento das descrições das categorias, a concordância passou de ${percent(firstRound.agreement, locale)} (κ = ${decimal(firstRound.kappa, locale)}) para ${percent(lastRound.agreement, locale)} (κ = ${decimal(lastRound.kappa, locale)}).`
        : ` After ${clarifyRounds} round(s) clarifying the category descriptions, agreement went from ${percent(firstRound.agreement, locale)} (κ = ${decimal(firstRound.kappa, locale)}) to ${percent(lastRound.agreement, locale)} (κ = ${decimal(lastRound.kappa, locale)}).`
      : '';
    parts.push(
      pt
        ? `Na amostra (n = ${count(firstRound.compared, locale)}), a concordância entre os rótulos dos dois modelos foi de ${percent(firstRound.agreement, locale)} (κ de Cohen = ${decimal(firstRound.kappa, locale)}).${rounds}`
        : `On the sample (n = ${count(firstRound.compared, locale)}), agreement between the two models' labels was ${percent(firstRound.agreement, locale)} (Cohen's κ = ${decimal(firstRound.kappa, locale)}).${rounds}`,
    );
  }

  if (run.expansionRounds > 0) {
    const added = run.finalTaxonomy.length - (run.taxonomyVersions.filter((v) => v.origin !== 'expand').at(-1)?.categories.length ?? run.finalTaxonomy.length);
    parts.push(
      pt
        ? `Uma rodada de expansão sobre os documentos remanescentes (classificados como "outros" ou com baixa confiança) acrescentou ${added} categoria(s), e a base foi reclassificada integralmente com a taxonomia final de ${run.finalTaxonomy.length} categorias.`
        : `An expansion round over the remaining documents (classified as "other" or with low confidence) added ${added} category(ies), and the whole corpus was reclassified with the final taxonomy of ${run.finalTaxonomy.length} categories.`,
    );
  }

  if (run.finalAgreement && run.finalAgreement.compared > 0 && run.expansionRounds > 0) {
    parts.push(
      pt
        ? `Considerando todos os documentos rotulados pelo modelo de linguagem (n = ${count(run.finalAgreement.compared, locale)}), a concordância final foi de ${percent(run.finalAgreement.agreement, locale)} (κ = ${decimal(run.finalAgreement.kappa, locale)}).`
        : `Across all documents labelled by the language model (n = ${count(run.finalAgreement.compared, locale)}), final agreement was ${percent(run.finalAgreement.agreement, locale)} (κ = ${decimal(run.finalAgreement.kappa, locale)}).`,
    );
  }

  parts.push(
    pt
      ? `Classificações com confiança ≥ ${decimal(thresholds.accept, locale)} foram aceitas automaticamente (${count(coverage.auto, locale)} documentos); entre ${decimal(thresholds.review, locale)} e ${decimal(thresholds.accept, locale)}, aceitas e sinalizadas para revisão (${count(coverage.review, locale)}); abaixo de ${decimal(thresholds.review, locale)}, mantidas como não classificadas (${count(coverage.unclassified, locale)}).`
      : `Classifications with confidence ≥ ${decimal(thresholds.accept, locale)} were accepted automatically (${count(coverage.auto, locale)} documents); between ${decimal(thresholds.review, locale)} and ${decimal(thresholds.accept, locale)}, accepted and flagged for review (${count(coverage.review, locale)}); below ${decimal(thresholds.review, locale)}, left unclassified (${count(coverage.unclassified, locale)}).`,
  );

  return parts.join(' ');
}
