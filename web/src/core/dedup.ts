import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, DedupResult, DuplicateRecord, SimetricsDoc } from '@/lib/types';
import { fitTransform, findSimilarPairs } from './ml/tfidf';
import { collectColumns, pickColumn, toNumeric } from './text';

/** Coluna que o relatório de excluídos usa para apontar o documento mantido. */
const KEPT_REFERENCE = 'DOCUMENTO DE REFERÊNCIA (MANTIDO)';

/**
 * Ordena por citações decrescentes — ⇄ o `sort_values(..., na_position='last')` que abre
 * as duas funções de dedup no Python. É o que decide qual cópia sobrevive: em caso de
 * duplicata, fica a versão mais citada.
 *
 * A ordenação precisa ser estável para empates, como a do pandas (mergesort no caminho
 * de `na_position`); `Array.prototype.sort` é estável por especificação desde o ES2019.
 */
function sortByCitationsDesc(rows: Dataset): { doc: SimetricsDoc; original: number }[] {
  return rows
    .map((doc, original) => ({ doc, original }))
    .sort((left, right) => {
      const a = toNumeric(left.doc[FIELD.TOTAL_CITATIONS]) ?? 0;
      const b = toNumeric(right.doc[FIELD.TOTAL_CITATIONS]) ?? 0;
      return b - a;
    });
}

/**
 * Deduplicação por DOI — ⇄ `deduplicar_por_doi` (utils.py:2864).
 * DOI normalizado (trim + minúsculas); a primeira ocorrência na ordem por citações vence.
 */
export function dedupByDoi(rows: Dataset): DedupResult {
  const columns = collectColumns(rows);
  const doiColumn = pickColumn(columns, FIELD_CANDIDATES.doi);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);

  if (!doiColumn) return { kept: [...rows], removed: [] };

  const ordered = sortByCitationsDesc(rows);
  const firstByDoi = new Map<string, SimetricsDoc>();
  const kept: Dataset = [];
  const removed: DuplicateRecord[] = [];

  for (const { doc } of ordered) {
    const doi = String(doc[doiColumn] ?? '').trim().toLowerCase();

    // Documento sem DOI nunca é tratado como duplicata: não há evidência de identidade.
    if (!doi) {
      kept.push(doc);
      continue;
    }

    const existing = firstByDoi.get(doi);
    if (existing === undefined) {
      firstByDoi.set(doi, doc);
      kept.push(doc);
      continue;
    }

    removed.push({
      ...doc,
      [KEPT_REFERENCE]: titleColumn ? String(existing[titleColumn] ?? '') : '',
    } as DuplicateRecord);
  }

  return { kept, removed };
}

export interface SimilarityDedupOptions {
  /** Similaridade de cosseno mínima para considerar duplicata. Padrão 0.90. */
  threshold?: number;
  onProgress?: (ratio: number) => void;
}

/**
 * Deduplicação por similaridade de título — ⇄ `deduplicar_por_similaridade` (utils.py:2899).
 *
 * Duas etapas, como no Python: primeiro títulos normalizados idênticos, depois TF-IDF com
 * bigramas e cosseno acima do limiar.
 *
 * ATENÇÃO — divergência deliberada em relação ao Python: lá o `TfidfVectorizer` é
 * construído com `token_pattern=None` e SEM `tokenizer`, o que levanta `TypeError` no
 * scikit-learn. Como a chamada está dentro de um `except Exception: pass`, a falha é
 * silenciosa e a etapa de similaridade nunca roda — na prática o app só faz dedup por
 * título exato. Aqui a etapa funciona de fato, então esta função remove MAIS duplicatas
 * que o Streamlit. É correção de bug, não regressão de paridade.
 */
export function dedupBySimilarity(
  rows: Dataset,
  options: SimilarityDedupOptions = {},
): DedupResult {
  const threshold = options.threshold ?? 0.9;
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);

  if (!titleColumn || rows.length < 2) return { kept: [...rows], removed: [] };

  const ordered = sortByCitationsDesc(rows);
  const normalized = ordered.map(({ doc }) =>
    String(doc[titleColumn] ?? '')
      .toLowerCase()
      .replace(/\s+/g, ' ')
      .trim(),
  );

  const excluded = new Set<number>();
  const keptReference = new Map<number, string>();

  // Etapa 1 — títulos normalizados idênticos.
  const firstByTitle = new Map<string, number>();
  for (let i = 0; i < ordered.length; i += 1) {
    const title = normalized[i] as string;
    if (!title) continue;

    const first = firstByTitle.get(title);
    if (first === undefined) {
      firstByTitle.set(title, i);
      continue;
    }

    excluded.add(i);
    keptReference.set(i, String((ordered[first] as { doc: SimetricsDoc }).doc[titleColumn] ?? ''));
  }

  // Etapa 2 — similaridade de cosseno sobre o que sobrou.
  const candidates: number[] = [];
  for (let i = 0; i < ordered.length; i += 1) {
    if (!excluded.has(i) && normalized[i]) candidates.push(i);
  }

  if (candidates.length > 1) {
    const model = fitTransform(
      candidates.map((index) => normalized[index] as string),
      { stopWords: true, ngramRange: [1, 2], minDf: 1 },
    );

    const pairs = findSimilarPairs(model.vectors, threshold, options.onProgress);
    // Pares em ordem crescente: o de menor posição (mais citado) é sempre o mantido.
    pairs.sort((left, right) => left.a - right.a || left.b - right.b);

    for (const pair of pairs) {
      const keepIndex = candidates[pair.a] as number;
      const dropIndex = candidates[pair.b] as number;
      if (excluded.has(keepIndex) || excluded.has(dropIndex)) continue;

      excluded.add(dropIndex);
      keptReference.set(
        dropIndex,
        String((ordered[keepIndex] as { doc: SimetricsDoc }).doc[titleColumn] ?? ''),
      );
    }
  }

  const kept: Dataset = [];
  const removed: DuplicateRecord[] = [];

  for (let i = 0; i < ordered.length; i += 1) {
    const doc = (ordered[i] as { doc: SimetricsDoc }).doc;
    if (excluded.has(i)) {
      removed.push({ ...doc, [KEPT_REFERENCE]: keptReference.get(i) ?? '' } as DuplicateRecord);
    } else {
      kept.push(doc);
    }
  }

  return { kept, removed };
}
