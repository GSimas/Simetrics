import type { DatabaseName } from './schema';

/**
 * Um documento após normalização. Índice aberto porque as bases trazem dezenas de campos
 * extras (FU, PU, C3…) que preservamos para exportação, mas os campos abaixo são garantidos
 * por `normalize.ts`.
 */
export interface SimetricsDoc extends Record<string, unknown> {
  TITLE: string;
  AUTHORS: string;
  'YEAR CLEAN': number | null;
  'TOTAL CITATIONS': number;
  'SECONDARY TITLE': string;
  ABSTRACT: string;
  KEYWORDS: string;
  COUNTRY: string;
  DOI: string;
  REFERENCES_UNIFIED: string;
  'BASE DE DADOS': DatabaseName | string;
  TEMA_GEMINI?: string;
}

export type Dataset = SimetricsDoc[];

/** Um documento descartado na deduplicação, com o documento que o substituiu. */
export interface DuplicateRecord extends SimetricsDoc {
  'DOCUMENTO DE REFERÊNCIA (MANTIDO)': string;
}

export interface DedupResult {
  kept: Dataset;
  removed: DuplicateRecord[];
}

/** Índices cientométricos de uma entidade — ⇄ extrair_indices_cientometricos. */
export interface ScientometricIndices {
  h: number;
  g: number;
  i10: number;
  m: number;
}

/** Linha das tabelas de Autores / Países / Venues / Keywords. */
export interface EntityMetrics extends ScientometricIndices {
  entity: string;
  docs: number;
  citations: number;
}

/** ⇄ analisar_completude_metadados */
export interface MetadataCompleteness {
  field: string;
  missing: number;
  missingPct: number;
  status: 'Excelente' | 'Bom' | 'Aceitável' | 'Ruim';
}

/** ⇄ calcular_metricas_bibliometrix */
export interface BibliometrixMetrics {
  growthRate: number;
  mcp: number;
  scp: number;
  coauthIndex: number;
  singleAuthorDocs: number;
  avgCitPerYear: number;
}

/** ⇄ resumir_base_bibliometrica */
export interface DatasetSummary {
  totalDocs: number;
  timespan: string;
  avgAge: number | null;
  authorsCount: number;
  countriesCount: number;
  keywordsCount: number;
  venuesCount: number;
  bibliometrix: BibliometrixMetrics;
}

export type NodeKind = 'Documento' | 'Autor' | 'País' | 'Local de Publicação (Venue)';

/** Uma linha da tabela de nós SNA — ⇄ _engine_calculo_sna. */
export interface SnaNodeMetrics {
  item: string;
  kind: NodeKind | 'Outro';
  degreeAbsolute: number;
  degreeCentrality: number;
  eigenvector: number;
  betweenness: number;
  closeness: number;
}

/**
 * Métricas globais da rede — ⇄ _calcular_metricas_globais_sna.
 * Campos podem vir como string quando o cálculo é suprimido por custo
 * (ex.: eficiência global em grafos com ≥1500 nós).
 */
export interface SnaGlobalMetrics {
  density: number;
  clustering: number;
  entropy: number;
  efficiency: number | string;
  meanDegree: number;
  stdDegree: number;
  minDegree: number;
  maxDegree: number;
  meanPageRank: number;
  meanEigenvector: number;
  powerLawExponent: number;
  assortativity: number;
  spearmanDegreeBetweenness: number;
}

/** Resultado de similaridade Jaccard — ⇄ calcular_similares_biblio. */
export interface SimilarityHit {
  item: string;
  similarity: number;
  sharedTraits: string;
}

export type SearchEntityType =
  | 'Documento'
  | 'Autor'
  | 'País'
  | 'Local de Publicação (Venue)'
  | 'Tema';

/** Progresso emitido pelos workers, substituindo st.progress. */
export interface WorkerProgress {
  phase: string;
  ratio: number;
  detail?: string;
}
