/**
 * Nomes canônicos de campo, idênticos aos do pipeline Python (`padronizar_base_bibliometrica`
 * em utils.py). Manter estas strings iguais às do Streamlit é o que permite comparar
 * campo a campo com o oráculo de paridade em tests/parity.
 *
 * NÃO renomear para camelCase: os testes de paridade leem o golden.json exportado pelo
 * pandas, cujas chaves são exatamente estes rótulos.
 */
export const FIELD = {
  TITLE: 'TITLE',
  AUTHORS: 'AUTHORS',
  YEAR: 'YEAR',
  YEAR_CLEAN: 'YEAR CLEAN',
  TOTAL_CITATIONS: 'TOTAL CITATIONS',
  SECONDARY_TITLE: 'SECONDARY TITLE',
  ABSTRACT: 'ABSTRACT',
  KEYWORDS: 'KEYWORDS',
  COUNTRY: 'COUNTRY',
  DOI: 'DOI',
  DOCUMENT_TYPE: 'DOCUMENT TYPE',
  TYPE_OF_REFERENCE: 'TYPE OF REFERENCE',
  REFERENCES_UNIFIED: 'REFERENCES_UNIFIED',
  CITED_REFERENCES: 'CITED REFERENCES',
  DATABASE: 'BASE DE DADOS',
  THEME: 'TEMA_GEMINI',
} as const;

/**
 * Candidatos por campo, na ordem de preferência — replica os `_pick_column(df, [...])`
 * espalhados pelo utils.py. Bases diferentes exportam rótulos diferentes para o mesmo dado.
 */
export const FIELD_CANDIDATES = {
  title: [FIELD.TITLE, 'TI'],
  authors: [FIELD.AUTHORS, 'AU'],
  countries: [FIELD.COUNTRY],
  venue: [FIELD.SECONDARY_TITLE, 'SO', 'JO'],
  year: [FIELD.YEAR_CLEAN, FIELD.YEAR, 'PY'],
  keywords: [FIELD.KEYWORDS, 'KW', 'DE'],
  doi: [FIELD.DOI, 'DO'],
  references: [FIELD.REFERENCES_UNIFIED, 'REFERENCES', FIELD.CITED_REFERENCES, 'CR'],
  affiliation: ['AUTHOR ADDRESS', 'AD', 'C1', 'AFFILIATIONS'],
  citations: ['TC', 'Z9', 'TIMES CITED', 'CITED BY'],
} as const satisfies Record<string, readonly string[]>;

/** Separador de valores múltiplos em um único campo (autores, países, keywords). */
export const MULTI_VALUE_SEPARATOR = ';';

/** Bases suportadas no seletor de origem do upload. Ordem = ordem do dropdown. */
export const DATABASES = [
  'Scopus',
  'Web of Science',
  'SciELO',
  'PubMed',
  'Cochrane',
  'Outra',
] as const;

export type DatabaseName = (typeof DATABASES)[number];

/** Teto de processamento anunciado na UI do Streamlit. */
export const MAX_DOCUMENTS = 10_000;

/**
 * Ano-base dos índices dependentes de tempo (m-index, idade média, citações/ano).
 * O Python usa `date.today().year`; replicado aqui para que o cálculo acompanhe o relógio.
 */
export function currentYear(): number {
  return new Date().getFullYear();
}
