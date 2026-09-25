/**
 * Tipos da classificação híbrida: um LLM gerativo (DeepSeek) descobre as categorias a
 * partir de uma amostra, e um modelo de decisão (Jev, da TypeSafe) aplica essas categorias
 * à base inteira, devolvendo probabilidade calibrada por documento.
 *
 * Tudo aqui é serializável: o registro da execução (`HybridRun`) é salvo junto com o
 * projeto e alimenta a seção de métodos do relatório — é o que torna o resultado
 * reproduzível e citável.
 */

/** Uma categoria da taxonomia. O `id` é estável entre versões; o `name` pode mudar. */
export interface HybridCategory {
  id: string;
  /** Nome exibido, no idioma da interface. Também é a chave da opção no Choice do Jev. */
  name: string;
  /** O que a categoria cobre — em inglês, idioma em que o Jev tem melhor acurácia. */
  what: string;
  /** O que a categoria NÃO cobre. É o campo que mais reduz a leitura literal do Jev. */
  notFor: string;
  /** Títulos de documentos da amostra que exemplificam a categoria. */
  examples: string[];
}

/** Id reservado da categoria "outro/nenhuma", sempre presente e sempre a última opção. */
export const OTHER_CATEGORY_ID = '__other__';

export type TaxonomyOrigin = 'discovery' | 'clarify' | 'expand' | 'manual' | 'user-edit';

export interface TaxonomyVersion {
  version: number;
  origin: TaxonomyOrigin;
  createdAt: string;
  categories: HybridCategory[];
}

/** Faixa de confiança de cada documento, decidida pelos limiares configurados. */
export type ConfidenceTier = 'auto' | 'review' | 'unclassified';

export interface ConfidenceThresholds {
  /** Acima deste valor, a classificação é aceita automaticamente. */
  accept: number;
  /** Abaixo deste valor, o documento fica "não classificado". Entre os dois, "revisar". */
  review: number;
}

/** Resultado do Jev para um documento. */
export interface DocumentDecision {
  categoryId: string;
  confidence: number;
  /** Probabilidade da opção escolhida — difere da confiança, que mede a forma da distribuição. */
  probability: number;
}

export interface CategoryAgreement {
  categoryId: string;
  name: string;
  /** Documentos que o DeepSeek pôs nesta categoria. */
  support: number;
  /** Fração desses documentos em que o Jev concordou. */
  agreement: number;
}

export interface ValidationRound {
  taxonomyVersion: number;
  /** Documentos com rótulo do DeepSeek comparados ao Jev. */
  compared: number;
  agreement: number;
  /** Kappa de Cohen — concordância descontado o acaso. */
  kappa: number;
  perCategory: CategoryAgreement[];
}

export interface CategoryCount {
  categoryId: string;
  name: string;
  documents: number;
  meanConfidence: number;
}

/** Registro completo de uma execução — o que vai para o projeto e para o relatório. */
export interface HybridRun {
  id: string;
  startedAt: string;
  finishedAt: string;
  locale: 'pt' | 'en';
  models: {
    discovery: string | null;
    /** Modelo que o Jev informou na resposta (ex.: jev-1.13.0), não o alias pedido. */
    classifier: string;
  };
  sample: {
    size: number;
    seed: number;
    strata: number;
    outliers: number;
    clusterCount: number;
    silhouette: number | null;
  };
  thresholds: ConfidenceThresholds;
  taxonomyVersions: TaxonomyVersion[];
  finalTaxonomy: HybridCategory[];
  validation: ValidationRound[];
  /** Concordância final: todos os rótulos do DeepSeek (amostra + sobras) contra o Jev final. */
  finalAgreement: ValidationRound | null;
  expansionRounds: number;
  coverage: {
    total: number;
    auto: number;
    review: number;
    unclassified: number;
    other: number;
  };
  categoryCounts: CategoryCount[];
  usage: {
    discoveryInputTokens: number;
    discoveryOutputTokens: number;
    classifierInputTokens: number;
    classifierRequests: number;
  };
  userEdited: boolean;
}
