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

/** 2: texto completo em PDF e evidências ligadas às respostas (`documents`, `evidence`). */
export const REVIEW_SCHEMA_VERSION = 2;

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

/**
 * Avaliação de qualidade no modelo do Parsifal: perguntas com as mesmas respostas para
 * todas, cada resposta com um peso; a nota do estudo é a soma dos pesos respondidos.
 */
export interface QualityQuestion {
  id: string;
  text: string;
}

export interface QualityAnswer {
  id: string;
  label: string;
  weight: number;
}

export const EXTRACTION_FIELD_TYPES = ['text', 'number', 'boolean', 'date', 'select', 'multiselect'] as const;
export type ExtractionFieldType = (typeof EXTRACTION_FIELD_TYPES)[number];

export interface ExtractionField {
  id: string;
  label: string;
  type: ExtractionFieldType;
  /** Opções de `select` e `multiselect`. */
  options: string[];
}

export type ExtractionValue = string | number | boolean | string[];

export interface StudyExtraction {
  values: Record<string, ExtractionValue>;
  done: boolean;
}

/**
 * O PDF do texto completo de um estudo. O arquivo em si fica fora do projeto, num banco
 * próprio do navegador (`pdf-store.ts`), endereçado pelo hash: o projeto e o arquivo de
 * decisões trocado entre revisores continuam leves e não levam o artigo por acidente.
 */
export interface StudyDocument {
  /** SHA-256 do arquivo — quem recebe o projeto sem o PDF anexa a própria cópia e ela é reconhecida. */
  hash: string;
  name: string;
  size: number;
  pages: number;
  /**
   * Camada de texto: `none` é PDF escaneado (sem texto para selecionar nem para a IA ler);
   * `partial`, algumas páginas sem texto — em geral figuras ou tabelas em imagem.
   */
  textLayer: 'ok' | 'partial' | 'none';
  addedAt: string;
}

/**
 * Onde uma resposta mora: um campo da extração, uma pergunta da avaliação de qualidade ou
 * o motivo de exclusão do texto completo. Em texto (`extraction:<id>`, `quality:<id>`,
 * `ft-exclusion`) para servir de chave de mapa.
 */
export type EvidenceTargetKey = `extraction:${string}` | `quality:${string}` | 'ft-exclusion';

/** Retângulo numa página, em frações do tamanho dela (0–1) — independe do zoom. */
export interface PageRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Onde o trecho citado pela IA foi achado no texto do PDF: `exact` (o mesmo texto, fora
 * espaços e pontuação), `approximate` (começo e fim batem, o meio difere um pouco) ou
 * `not-found` — citação que o modelo pode ter inventado e que não pode ser confirmada direto.
 */
export type QuoteLocation = 'exact' | 'approximate' | 'not-found';

/**
 * Um trecho do artigo que sustenta uma resposta. Guarda o texto citado e um pouco do
 * texto em volta (como o seletor de citação da W3C Web Annotation), não só coordenadas:
 * assim o destaque é reencontrado mesmo em outra cópia do PDF, e o relatório mostra a
 * citação sem precisar do arquivo.
 */
export interface Evidence {
  id: string;
  /** 1 em diante. */
  page: number;
  /** Vazio numa marcação de área (tabela, figura). */
  quote: string;
  prefix: string;
  suffix: string;
  /** Contorno do destaque — o que resta quando o texto não é reencontrado, e toda a marcação de área. */
  rects: PageRect[];
  origin: 'manual' | 'ai';
  location: QuoteLocation;
  reviewerId: string;
  createdAt: string;
}

/** Resposta proposta pela IA: o valor do campo, o id da resposta de qualidade ou o id do critério de exclusão. */
export interface AiSuggestion {
  value: ExtractionValue;
  /** Por que o modelo respondeu assim, em uma frase. */
  rationale: string;
  model: string;
  createdAt: string;
}

/**
 * Conferência humana de uma resposta:
 * - `suggested`: a IA propôs, ninguém conferiu ainda;
 * - `confirmed`: o revisor aceitou a resposta da IA como veio;
 * - `edited`: o revisor corrigiu a resposta da IA;
 * - `rejected`: o revisor descartou a proposta;
 * - `manual`: o revisor respondeu sem a IA.
 */
export type VerificationStatus = 'suggested' | 'confirmed' | 'edited' | 'rejected' | 'manual';

export interface TargetEvidence {
  evidence: Evidence[];
  suggestion?: AiSuggestion;
  status: VerificationStatus;
  verifiedBy?: string;
  verifiedAt?: string;
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
  qualityQuestions: QualityQuestion[];
  qualityAnswers: QualityAnswer[];
  /** Nota mínima para o estudo passar; `null` = sem nota de corte. */
  qualityCutoff: number | null;
  /** Estudos abaixo da nota de corte saem da seleção final (e entram como excluídos no PRISMA). */
  excludeBelowCutoff: boolean;
  /** Respostas por estudo: chave do registro → pergunta → resposta. */
  quality: Record<string, Record<string, string>>;
  extractionFields: ExtractionField[];
  extraction: Record<string, StudyExtraction>;
  /** PDF do texto completo por chave de registro. */
  documents: Record<string, StudyDocument>;
  /** Evidências e conferência por estudo → resposta (`EvidenceTargetKey`). */
  evidence: Record<string, Partial<Record<EvidenceTargetKey, TargetEvidence>>>;
  /**
   * Quem tomou as decisões deste arquivo: o id do dispositivo. Serve à fase de dois
   * revisores (juntar arquivos de pessoas diferentes) e, depois, à conta Scientata.
   */
  reviewerId: string;
  createdAt: string;
  updatedAt: string;
}
