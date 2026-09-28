/**
 * Modelo de dados da revisão sistematizada (revisão sistemática, de escopo, integrativa…).
 *
 * O desenho segue o fluxo do Parsifal (planejamento → condução → relato), mas o estado
 * vive dentro do projeto do Simetrics, ao lado da base bibliométrica: a triagem trabalha
 * sobre a base ativa (já deduplicada), e as decisões ficam indexadas por uma chave
 * derivada do conteúdo do registro (ver `record-key.ts`), não pela posição na base — assim
 * sobrevivem a uma nova deduplicação, a arquivos acrescentados e, na fase de dois
 * revisores, à troca de arquivos entre pessoas que importaram a mesma busca.
 */

export const REVIEW_SCHEMA_VERSION = 1;

export const REVIEW_TYPES = [
  'systematic',
  'scoping',
  'integrative',
  'rapid',
  'umbrella',
  'mapping',
  'other',
] as const;
export type ReviewType = (typeof REVIEW_TYPES)[number];

/** Estruturas de pergunta de pesquisa. Cada uma define os campos do protocolo. */
export const FRAMEWORKS = ['PICO', 'PICOC', 'PCC', 'SPIDER', 'none'] as const;
export type Framework = (typeof FRAMEWORKS)[number];

export const FRAMEWORK_FIELDS: Record<Framework, readonly string[]> = {
  PICO: ['population', 'intervention', 'comparison', 'outcome'],
  PICOC: ['population', 'intervention', 'comparison', 'outcome', 'context'],
  PCC: ['population', 'concept', 'context'],
  SPIDER: ['sample', 'phenomenon', 'design', 'evaluation', 'researchType'],
  none: [],
};

/** Estrutura e diretriz de relato sugeridas para cada tipo — o usuário pode trocar a estrutura. */
export const REVIEW_DEFAULTS: Record<ReviewType, { framework: Framework; guideline: string }> = {
  systematic: { framework: 'PICO', guideline: 'PRISMA 2020' },
  scoping: { framework: 'PCC', guideline: 'PRISMA-ScR' },
  integrative: { framework: 'PICO', guideline: 'PRISMA 2020' },
  rapid: { framework: 'PICO', guideline: 'PRISMA 2020' },
  umbrella: { framework: 'PICO', guideline: 'PRIOR' },
  mapping: { framework: 'PICOC', guideline: 'PRISMA 2020' },
  other: { framework: 'none', guideline: 'PRISMA 2020' },
};

export interface ReviewQuestion {
  id: string;
  text: string;
}

/**
 * Um conceito da busca e seus sinônimos. Termos de um conceito se unem por OR; conceitos
 * entre si, por AND — a estrutura clássica de uma string de busca.
 */
export interface SearchConcept {
  id: string;
  label: string;
  terms: string[];
}

export type CriterionKind = 'inclusion' | 'exclusion';

export interface Criterion {
  id: string;
  kind: CriterionKind;
  text: string;
}

export type ScreeningStage = 'title-abstract' | 'full-text';

/** "Talvez" segue para o texto completo, como no Covidence: é uma inclusão com dúvida. */
export type TitleAbstractDecision = 'include' | 'exclude' | 'maybe';
export type FullTextDecision = 'include' | 'exclude' | 'not-retrieved';

export interface RecordScreening {
  ta?: TitleAbstractDecision;
  /** Critério de exclusão que motivou a exclusão na triagem por título e resumo (opcional). */
  taReason?: string;
  ft?: FullTextDecision;
  /** Critério de exclusão do texto completo — o PRISMA 2020 pede o motivo nesta etapa. */
  ftReason?: string;
  note?: string;
  updatedAt: string;
}

export interface ReviewState {
  schemaVersion: typeof REVIEW_SCHEMA_VERSION;
  type: ReviewType;
  title: string;
  objective: string;
  framework: Framework;
  frameworkValues: Record<string, string>;
  questions: ReviewQuestion[];
  concepts: SearchConcept[];
  /** Bases para as quais gerar a string de busca (ids de `SEARCH_TARGETS`). */
  searchTargets: string[];
  criteria: Criterion[];
  /** Decisões por chave de registro. */
  decisions: Record<string, RecordScreening>;
  /**
   * Quem tomou as decisões deste arquivo: o id do dispositivo. Serve à fase de dois
   * revisores (juntar arquivos de pessoas diferentes) e, depois, à conta Scientata.
   */
  reviewerId: string;
  createdAt: string;
  updatedAt: string;
}
