import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { normalizeDataset, type RawRow } from '../normalize';
import { extractCountriesPubmed } from './countries';

/**
 * Importador de PubMed / Medline (.txt, .nbib) — ⇄ `processar_pubmed` (utils.py:210).
 *
 * O formato Medline usa `TAG - valor`, com continuações indentadas e tags repetidas para
 * valores múltiplos. Cada registro começa numa linha PMID.
 */

/** Tags que podem repetir dentro do mesmo registro e por isso acumulam em lista. */
const LIST_TAGS = new Set(['FAU', 'AU', 'AD', 'OT', 'MH', 'PT', 'LID', 'AID']);

const TAG_LINE = /^([A-Z0-9]{2,4})\s*-\s(.*)$/;

type MedlineRecord = Record<string, string | string[]>;

function parseMedline(text: string): MedlineRecord[] {
  const records: MedlineRecord[] = [];
  let current: MedlineRecord | null = null;
  let lastTag: string | null = null;

  for (const line of text.split(/\r\n|\r|\n/)) {
    if (!line.trim()) continue;

    // Linha indentada continua a tag anterior (abstracts longos, endereços).
    if (/^\s/.test(line)) {
      if (current && lastTag && lastTag in current) {
        const existing = current[lastTag];
        if (Array.isArray(existing)) {
          const lastIndex = existing.length - 1;
          existing[lastIndex] = `${existing[lastIndex]} ${line.trim()}`;
        } else {
          current[lastTag] = `${existing} ${line.trim()}`;
        }
      }
      continue;
    }

    const match = TAG_LINE.exec(line);
    if (!match) continue;

    const tag = match[1] as string;
    const value = (match[2] as string).trim();

    if (tag === 'PMID') {
      if (current) records.push(current);
      current = { PMID: value };
    } else if (current) {
      if (LIST_TAGS.has(tag)) {
        const existing = current[tag];
        if (Array.isArray(existing)) existing.push(value);
        else current[tag] = [value];
      } else {
        current[tag] = value;
      }
    }

    lastTag = tag;
  }

  if (current) records.push(current);
  return records;
}

function asList(value: string | string[] | undefined): string[] {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

export function processPubmed(text: string): Dataset {
  const records = parseMedline(text);
  if (records.length === 0) return [];

  const rows: RawRow[] = records.map((record) => {
    const row: RawRow = { ...record };

    row[FIELD.TITLE] = record['TI'] ?? '';
    row[FIELD.ABSTRACT] = record['AB'] ?? '';
    // JT é o título completo do periódico; TA é a abreviação, usada como reserva.
    row[FIELD.SECONDARY_TITLE] = record['JT'] ?? record['TA'] ?? '';

    // FAU traz o nome completo do autor; AU só as iniciais.
    const authors = asList(record['FAU'] ?? record['AU']);
    row[FIELD.AUTHORS] = authors.join('; ');

    // OT são termos livres, MH são descritores MeSH; o asterisco marca o termo principal.
    const keywords = new Set<string>();
    for (const tag of ['OT', 'MH'] as const) {
      for (const value of asList(record[tag])) {
        const cleaned = value.replace(/\*/g, '').trim();
        if (cleaned) keywords.add(cleaned);
      }
    }
    row[FIELD.KEYWORDS] = [...keywords].join('; ');

    // DP é a data de publicação, em formatos variados; só o ano interessa.
    const publicationDate = String(record['DP'] ?? '');
    const year = /(\d{4})/.exec(publicationDate);
    row[FIELD.YEAR_CLEAN] = year ? Number(year[1]) : null;

    // O DOI vem marcado com o sufixo [doi] dentro de LID ou AID.
    let doi = '';
    for (const tag of ['LID', 'AID'] as const) {
      for (const value of asList(record[tag])) {
        if (value.toLowerCase().includes('[doi]')) {
          doi = value.replace(/\[doi\]/gi, '').trim();
          break;
        }
      }
      if (doi) break;
    }
    row[FIELD.DOI] = doi;

    row[FIELD.DOCUMENT_TYPE] = asList(record['PT']).join(', ');
    row[FIELD.COUNTRY] = extractCountriesPubmed(asList(record['AD'])) ?? '';

    // Exports .txt/.nbib do PubMed não trazem contagem de citações recebidas.
    row[FIELD.TOTAL_CITATIONS] = 0;

    return row;
  });

  return normalizeDataset(rows);
}
