/**
 * Parser RIS equivalente ao `rispy`, incluindo suas particularidades.
 *
 * Reimplementado em vez de adaptado porque o pipeline Python depende de detalhes exatos
 * do rispy: quais tags viram lista, como linhas de continuação são anexadas, e o balde
 * `unknown_tag` que carrega as tags proprietárias (TC, Z9, C7...) de onde saem as
 * citações do WoS e do Scopus.
 */

/** ⇄ `rispy.config.TAG_KEY_MAPPING`. */
const TAG_KEY_MAPPING: Readonly<Record<string, string>> = {
  TY: 'type_of_reference',
  A1: 'first_authors',
  A2: 'secondary_authors',
  A3: 'tertiary_authors',
  A4: 'subsidiary_authors',
  AB: 'abstract',
  AD: 'author_address',
  AN: 'accession_number',
  AU: 'authors',
  C1: 'custom1',
  C2: 'custom2',
  C3: 'custom3',
  C4: 'custom4',
  C5: 'custom5',
  C6: 'custom6',
  C7: 'custom7',
  C8: 'custom8',
  CA: 'caption',
  CN: 'call_number',
  CY: 'place_published',
  DA: 'date',
  DB: 'name_of_database',
  DO: 'doi',
  DP: 'database_provider',
  ET: 'edition',
  EP: 'end_page',
  ID: 'id',
  IS: 'number',
  J2: 'alternate_title1',
  JA: 'alternate_title2',
  JF: 'alternate_title3',
  JO: 'journal_name',
  KW: 'keywords',
  L1: 'file_attachments1',
  L2: 'file_attachments2',
  L4: 'figure',
  LA: 'language',
  LB: 'label',
  M1: 'note',
  M3: 'type_of_work',
  N1: 'notes',
  N2: 'notes_abstract',
  NV: 'number_of_volumes',
  OP: 'original_publication',
  PB: 'publisher',
  PY: 'year',
  RI: 'reviewed_item',
  RN: 'research_notes',
  RP: 'reprint_edition',
  SE: 'section',
  SN: 'issn',
  SP: 'start_page',
  ST: 'short_title',
  T1: 'primary_title',
  T2: 'secondary_title',
  T3: 'tertiary_title',
  TA: 'translated_author',
  TI: 'title',
  TT: 'translated_title',
  UR: 'urls',
  VL: 'volume',
  Y1: 'publication_year',
  Y2: 'access_date',
  ER: 'end_of_reference',
  UK: 'unknown_tag',
};

/** Tags que acumulam múltiplos valores — ⇄ `rispy.config.LIST_TYPE_TAGS`. */
const LIST_TYPE_TAGS = new Set(['A1', 'A2', 'A3', 'A4', 'AU', 'KW', 'N1', 'UR']);

/** ⇄ `rispy.config.DELIMITED_TAG_MAPPING`. */
const DELIMITED_TAGS: Readonly<Record<string, string>> = { UR: ';' };

const START_TAG = 'TY';
const END_TAG = 'ER';

export type RisValue = string | string[] | Record<string, string[]>;
export type RisRecord = Record<string, RisValue>;

/**
 * Detecta uma linha de tag exatamente como o `RisParser.parse_line` do rispy:
 * duas letras maiúsculas, `"  -"` nas posições 2-4, conteúdo a partir da 6.
 * Qualquer outra linha é continuação da tag anterior.
 */
function parseLine(line: string): { tag: string; content: string } | { tag: null; content: string } {
  const head = line.slice(0, 2);
  if (line.slice(2, 5) === '  -' && head === head.toUpperCase() && /^[A-Za-z]/.test(head)) {
    return { tag: head, content: line.slice(6).trim() };
  }
  return { tag: null, content: line.trim() };
}

function addListValue(record: RisRecord, key: string, value: string | string[]): void {
  const incoming = Array.isArray(value) ? value : [value];
  const existing = record[key];

  if (existing === undefined) {
    record[key] = [...incoming];
  } else if (Array.isArray(existing)) {
    existing.push(...incoming);
  } else {
    record[key] = [existing as string, ...incoming];
  }
}

function addTag(record: RisRecord, tag: string, rawContent: string, isContinuation: boolean): void {
  const key = TAG_KEY_MAPPING[tag];

  if (key === undefined) {
    // Tag desconhecida vai para o balde `unknown_tag`, preservando o rótulo original.
    // É daí que saem TC/Z9 (citações do WoS) mais adiante no pipeline.
    const bucket = (record['unknown_tag'] ??= {} as Record<string, string[]>) as Record<
      string,
      string[]
    >;
    (bucket[tag] ??= []).push(rawContent);
    return;
  }

  const delimiter = DELIMITED_TAGS[tag];
  const content: string | string[] = delimiter
    ? rawContent.split(delimiter).map((part) => part.trim())
    : rawContent;

  if (LIST_TYPE_TAGS.has(tag)) {
    addListValue(record, key, content);
    return;
  }

  if (isContinuation) {
    // Continuação de tag simples: o rispy junta com um espaço.
    const existing = record[key];
    if (Array.isArray(existing)) existing.push(...(Array.isArray(content) ? content : [content]));
    else record[key] = `${String(existing ?? '')} ${String(content)}`;
    return;
  }

  // `enforce_list_tags=True` usa setdefault: em tags repetidas, o PRIMEIRO valor vence.
  record[key] ??= content;
}

/** ⇄ `rispy.loads`. */
export function parseRis(text: string): RisRecord[] {
  // Divide APENAS em \n, nunca em \r isolado.
  //
  // Não é descuido: o app Python alimenta o rispy com `io.StringIO(bytes.decode("utf-8"))`,
  // e o StringIO usa `newline='\n'` por padrão — ou seja, não traduz CR. O rispy então faz
  // `text.split('\n')`, e blocos separados por CR sozinho permanecem numa única linha.
  //
  // Isso importa: exports do WoS trazem afiliações como `AD - ...\rAD - ...\rAD - ...`.
  // Dividindo no \r, cada AD vira uma tag separada — e como AD não é tag de lista, o rispy
  // descarta todas menos a primeira, perdendo os demais países. Mantendo o blob junto, o
  // extrator geográfico varre o texto inteiro e recupera todos. Daí também o regex que
  // limpa prefixos `AD -`/`C3 -` residuais em countries.ts.
  const lines = text.split('\n');
  const records: RisRecord[] = [];

  let record: RisRecord | null = null;
  let lastTag: string | null = null;

  for (const line of lines) {
    const { tag, content } = parseLine(line);

    // Fora de um registro, só a tag de início abre um novo.
    if (record === null) {
      if (tag === START_TAG) {
        record = { [TAG_KEY_MAPPING[START_TAG] as string]: content };
        lastTag = START_TAG;
      }
      continue;
    }

    if (tag === null) {
      if (lastTag) addTag(record, lastTag, content, true);
      continue;
    }

    if (tag === END_TAG) {
      records.push(record);
      record = null;
      lastTag = null;
      continue;
    }

    addTag(record, tag, content, false);
    lastTag = tag;
  }

  return records;
}

/**
 * Achata um registro RIS em linha tabular — ⇄ o pós-processamento de
 * `process_multiple_ris` (utils.py:2745).
 *
 * O `unknown_tag` é dissolvido de volta no nível superior sob o rótulo cru da tag, e
 * listas viram texto separado por "; ". Depois disso os nomes viram MAIÚSCULAS com
 * underscore trocado por espaço, produzindo `AUTHOR ADDRESS`, `TYPE OF REFERENCE` etc.
 */
export function flattenRisRecord(record: RisRecord): Record<string, string> {
  const flat: Record<string, string> = {};

  const assign = (key: string, value: RisValue): void => {
    const column = key.toUpperCase().replace(/_/g, ' ');
    flat[column] = Array.isArray(value) ? value.map(String).join('; ') : String(value);
  };

  for (const [key, value] of Object.entries(record)) {
    if (key === 'unknown_tag') {
      const bucket = value as Record<string, string[]>;
      for (const [tag, values] of Object.entries(bucket)) {
        assign(tag, values.length === 1 ? (values[0] as string) : values);
      }
      continue;
    }
    assign(key, value);
  }

  return flat;
}
