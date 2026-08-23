import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { tokenize } from './ml/tfidf';
import { collectColumns, pickColumn, toNumeric } from './text';

/**
 * Recuperação de documentos por BM25.
 *
 * É o que viabiliza o assistente sem enviar a base inteira. O app Streamlit injetava
 * todo o dataset em JSON dentro do `system_instruction` (Geral.py:2227) — com 10.000
 * documentos e resumos, isso passa das dezenas de megabytes, estourando tanto o limite
 * de 6 MB de payload da Netlify quanto qualquer janela de contexto razoável.
 *
 * Aqui o navegador indexa a base uma vez e, a cada pergunta, seleciona as poucas dezenas
 * de documentos mais relevantes. O que viaja pela rede são esses documentos, não a base.
 *
 * BM25 em vez de TF-IDF puro porque ele satura a frequência do termo: um documento que
 * repete "memética" cinquenta vezes não é cinquenta vezes mais relevante que um que a
 * menciona uma vez, e o BM25 modela exatamente isso.
 */

/** Saturação da frequência do termo. 1.2 é o valor canônico da literatura. */
const K1 = 1.2;
/** Peso da normalização por comprimento do documento. */
const B = 0.75;

export interface RetrievalIndex {
  /** Termo -> lista de (documento, frequência). */
  postings: Map<string, { doc: number; frequency: number }[]>;
  /** Frequência de documento por termo. */
  documentFrequency: Map<string, number>;
  documentLengths: Int32Array;
  averageLength: number;
  documentCount: number;
}

/** Texto indexável de um documento: título, resumo e palavras-chave. */
function documentText(
  doc: SimetricsDoc,
  columns: { title: string | null; keywords: string | null },
): string {
  const parts = [
    columns.title ? String(doc[columns.title] ?? '') : '',
    columns.keywords ? String(doc[columns.keywords] ?? '') : '',
    String(doc[FIELD.ABSTRACT] ?? ''),
  ];
  return parts.filter(Boolean).join(' ');
}

/** Indexa a base. Roda uma vez por dataset, no worker. */
export function buildIndex(rows: Dataset): RetrievalIndex {
  const allColumns = collectColumns(rows);
  const columns = {
    title: pickColumn(allColumns, FIELD_CANDIDATES.title),
    keywords: pickColumn(allColumns, FIELD_CANDIDATES.keywords),
  };

  const postings = new Map<string, { doc: number; frequency: number }[]>();
  const documentFrequency = new Map<string, number>();
  const documentLengths = new Int32Array(rows.length);

  let totalLength = 0;

  rows.forEach((doc, index) => {
    const tokens = tokenize(documentText(doc, columns), { stopWords: true });
    documentLengths[index] = tokens.length;
    totalLength += tokens.length;

    const frequencies = new Map<string, number>();
    for (const token of tokens) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);

    for (const [term, frequency] of frequencies) {
      let list = postings.get(term);
      if (!list) postings.set(term, (list = []));
      list.push({ doc: index, frequency });
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
  });

  return {
    postings,
    documentFrequency,
    documentLengths,
    averageLength: rows.length > 0 ? totalLength / rows.length : 0,
    documentCount: rows.length,
  };
}

export interface RetrievalHit {
  doc: number;
  score: number;
}

/**
 * Documentos mais relevantes para uma consulta, do mais ao menos relevante.
 *
 * Percorre apenas as listas invertidas dos termos da pergunta — nunca a base inteira.
 */
export function search(index: RetrievalIndex, query: string, topN = 40): RetrievalHit[] {
  const terms = tokenize(query, { stopWords: true });
  if (terms.length === 0) return [];

  const scores = new Map<number, number>();

  for (const term of new Set(terms)) {
    const list = index.postings.get(term);
    if (!list) continue;

    const df = index.documentFrequency.get(term) ?? 0;
    // IDF do BM25 (Robertson), com o +0.5 que suaviza termos muito comuns.
    const idf = Math.log(1 + (index.documentCount - df + 0.5) / (df + 0.5));

    for (const posting of list) {
      const length = index.documentLengths[posting.doc] as number;
      const normalization = 1 - B + (B * length) / (index.averageLength || 1);
      const saturated =
        (posting.frequency * (K1 + 1)) / (posting.frequency + K1 * normalization);

      scores.set(posting.doc, (scores.get(posting.doc) ?? 0) + idf * saturated);
    }
  }

  return [...scores.entries()]
    .map(([doc, score]) => ({ doc, score }))
    .sort((left, right) => right.score - left.score)
    .slice(0, topN);
}

/** Um documento compactado para caber no contexto do modelo. */
export interface ContextDocument {
  title: string;
  authors: string;
  year: number | null;
  venue: string;
  citations: number;
  keywords: string;
  abstract: string;
  theme?: string;
}

/** Tamanho máximo do resumo enviado, em caracteres. */
const ABSTRACT_LIMIT = 700;

/**
 * Converte documentos selecionados no formato enxuto que vai para o modelo.
 *
 * Os resumos são truncados: alguns passam de 3.000 caracteres, e quarenta deles inteiros
 * consumiriam a maior parte da janela de contexto sem acrescentar proporcionalmente.
 */
export function toContextDocuments(rows: Dataset, hits: readonly RetrievalHit[]): ContextDocument[] {
  const allColumns = collectColumns(rows);
  const titleColumn = pickColumn(allColumns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(allColumns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(allColumns, FIELD_CANDIDATES.venue);
  const keywordsColumn = pickColumn(allColumns, FIELD_CANDIDATES.keywords);
  const hasTheme = allColumns.has(FIELD.THEME);

  return hits.map(({ doc: index }) => {
    const doc = rows[index] as SimetricsDoc;
    const abstract = String(doc[FIELD.ABSTRACT] ?? '');

    const context: ContextDocument = {
      title: titleColumn ? String(doc[titleColumn] ?? '') : '',
      authors: authorsColumn ? String(doc[authorsColumn] ?? '') : '',
      year: toNumeric(doc[FIELD.YEAR_CLEAN]),
      venue: venueColumn ? String(doc[venueColumn] ?? '') : '',
      citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
      keywords: keywordsColumn ? String(doc[keywordsColumn] ?? '') : '',
      abstract:
        abstract.length > ABSTRACT_LIMIT ? `${abstract.slice(0, ABSTRACT_LIMIT)}…` : abstract,
    };

    if (hasTheme) {
      const theme = String(doc[FIELD.THEME] ?? '');
      if (theme) context.theme = theme;
    }

    return context;
  });
}
