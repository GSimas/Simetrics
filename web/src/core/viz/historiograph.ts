import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { collectColumns, pickColumn, titleCase, toNumeric } from '../text';

/**
 * Historiograph: linha do tempo de citações diretas — ⇄ `gerar_historiograph`
 * (utils.py:1817).
 *
 * Mostra quais documentos da própria base citam quais outros, posicionados pelo ano. É o
 * que revela a linhagem intelectual: quem são os trabalhos seminais e por onde a ideia
 * se propagou.
 *
 * A detecção de citação é por heurística — procura sobrenome do primeiro autor E ano
 * dentro da string de referências. Não há identificador confiável ligando referências a
 * documentos: as bases exportam referências como texto livre, em formatos incompatíveis
 * entre si. A heurística erra nos dois sentidos (homônimos geram falsos positivos,
 * grafias divergentes geram falsos negativos), e é o mesmo compromisso do original.
 */

export interface HistoriographNode {
  id: string;
  title: string;
  year: number;
  citations: number;
  /** Tamanho visual já escalonado. */
  size: number;
  /** Posição vertical dentro do ano, para separar documentos do mesmo período. */
  offset: number;
}

export interface HistoriographEdge {
  /** Documento que cita. */
  from: string;
  /** Documento citado. */
  to: string;
}

export interface HistoriographData {
  nodes: HistoriographNode[];
  edges: HistoriographEdge[];
}

const MIN_SIZE = 20;
const MAX_SIZE = 60;

/** Sobrenome do primeiro autor, em minúsculas — a chave da busca nas referências. */
function firstAuthorSurname(value: unknown): string {
  const first = String(value ?? '').split(';')[0] ?? '';
  return (first.split(',')[0] ?? '').trim().toLowerCase();
}

export function historiograph(rows: Dataset, topN = 30): HistoriographData | null {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const referencesColumn = pickColumn(columns, FIELD_CANDIDATES.references);

  if (!titleColumn || !authorsColumn || !referencesColumn) return null;

  // Só os mais citados: o grafo fica ilegível muito antes de esgotar a base, e são os
  // documentos de alto impacto que formam a linhagem visível.
  const candidates = rows
    .filter((doc) => {
      const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
      return year !== null && String(doc[referencesColumn] ?? '').trim() !== '';
    })
    .sort(
      (left, right) =>
        (toNumeric(right[FIELD.TOTAL_CITATIONS]) ?? 0) -
        (toNumeric(left[FIELD.TOTAL_CITATIONS]) ?? 0),
    )
    .slice(0, topN);

  if (candidates.length === 0) return null;

  interface Entry {
    doc: SimetricsDoc;
    id: string;
    year: number;
    citations: number;
    surname: string;
    referencesText: string;
  }

  const entries: Entry[] = candidates.map((doc) => {
    const year = Math.trunc(toNumeric(doc[FIELD.YEAR_CLEAN]) as number);
    const surname = firstAuthorSurname(doc[authorsColumn]);

    return {
      doc,
      id: `${titleCase(surname) || 'Anônimo'}, ${year}`,
      year,
      citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
      surname,
      referencesText: String(doc[referencesColumn] ?? '').toLowerCase(),
    };
  });

  const edges: HistoriographEdge[] = [];
  for (const citing of entries) {
    for (const cited of entries) {
      if (citing === cited) continue;
      // Um documento só pode citar algo anterior a ele.
      if (citing.year <= cited.year) continue;
      if (!cited.surname) continue;

      if (
        citing.referencesText.includes(cited.surname) &&
        citing.referencesText.includes(String(cited.year))
      ) {
        edges.push({ from: citing.id, to: cited.id });
      }
    }
  }

  // Escalonamento do tamanho pelas citações.
  const citationValues = entries.map((entry) => entry.citations);
  const minCitations = Math.min(...citationValues);
  const maxCitations = Math.max(...citationValues);
  const range = maxCitations - minCitations + 1;

  // Documentos do mesmo ano são distribuídos verticalmente para não se sobreporem.
  const byYear = new Map<number, Entry[]>();
  for (const entry of entries) {
    let bucket = byYear.get(entry.year);
    if (!bucket) byYear.set(entry.year, (bucket = []));
    bucket.push(entry);
  }

  const nodes: HistoriographNode[] = [];
  for (const [, group] of byYear) {
    group.forEach((entry, index) => {
      // Espaçamento uniforme entre 0,1 e 0,9, deixando margem nas bordas.
      const offset = group.length === 1 ? 0.5 : 0.1 + (0.8 * index) / (group.length - 1);

      nodes.push({
        id: entry.id,
        title: String(entry.doc[titleColumn] ?? ''),
        year: entry.year,
        citations: entry.citations,
        size: MIN_SIZE + ((entry.citations - minCitations) / range) * (MAX_SIZE - MIN_SIZE),
        offset,
      });
    });
  }

  // Nomes podem colidir (mesmo sobrenome e ano); as arestas ligam por id, então basta
  // manter um nó por id para o desenho não duplicar.
  const unique = new Map(nodes.map((node) => [node.id, node]));

  return { nodes: [...unique.values()], edges };
}
