import type { HybridRun } from './types';

/**
 * Linhas da tabela de temas da classificação híbrida, compartilhadas pela prévia do
 * relatório, pelo PDF e pelo DOCX — os três mostram exatamente os mesmos números.
 */
export interface HybridReportRow {
  name: string;
  documents: number;
  share: number;
  meanConfidence: number;
}

export function hybridReportRows(run: HybridRun): HybridReportRow[] {
  const total = run.coverage.total;
  return run.categoryCounts
    .filter((item) => item.documents > 0)
    .map((item) => ({
      name: item.name,
      documents: item.documents,
      share: total > 0 ? (item.documents / total) * 100 : 0,
      meanConfidence: item.meanConfidence,
    }));
}

export function hybridReportTitle(run: HybridRun, locale: 'pt' | 'en'): string {
  const round = run.validation[run.validation.length - 1];
  const kappa = round && round.compared > 0 ? ` · κ ${round.kappa.toFixed(2)}` : '';
  return locale === 'en'
    ? `Hybrid Thematic Classification (${run.finalTaxonomy.length} categories${kappa})`
    : `Classificação Temática Híbrida (${run.finalTaxonomy.length} categorias${kappa})`;
}
