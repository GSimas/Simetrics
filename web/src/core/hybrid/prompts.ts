import { truncate, type HybridDoc } from './documents';
import { OTHER_CATEGORY_ID, type HybridCategory } from './types';

/**
 * Prompts do modelo gerativo (DeepSeek) e a leitura das respostas.
 *
 * O modelo gerativo só faz o que o Jev não faz: escrever. Ele propõe as categorias,
 * reescreve descrições ambíguas e sugere categorias que faltaram. Toda resposta é JSON e
 * passa por validação aqui — nada do que o modelo devolve entra na taxonomia sem ser
 * conferido (nomes duplicados, exemplos que não existem na amostra, rótulos inválidos).
 *
 * As descrições (`what`/`not_for`) são pedidas em inglês de propósito: a documentação do
 * Jev informa que outros idiomas são aceitos, mas com acurácia menor. Só o nome da
 * categoria, que o usuário lê, fica no idioma da interface.
 */

export type PromptLocale = 'pt' | 'en';

const ABSTRACT_LIMIT = 600;
const KEYWORDS_LIMIT = 160;
const EXAMPLE_TITLE_LIMIT = 140;
const MAX_EXAMPLES = 3;

export interface ChatPrompt {
  system: string;
  user: string;
}

/** Rótulo curto de cada documento no prompt: D1, D2… (índice na lista enviada, não na base). */
export function docLabel(position: number): string {
  return `D${position + 1}`;
}

function describeDoc(doc: HybridDoc, position: number): string {
  const lines = [`[${docLabel(position)}] ${doc.title || '(untitled)'}`];
  if (doc.year !== null) lines.push(`Year: ${doc.year}`);
  if (doc.keywords) lines.push(`Keywords: ${truncate(doc.keywords, KEYWORDS_LIMIT)}`);
  lines.push(`Abstract: ${doc.abstract ? truncate(doc.abstract, ABSTRACT_LIMIT) : '(no abstract)'}`);
  return lines.join('\n');
}

function languageName(locale: PromptLocale): string {
  return locale === 'pt' ? 'Brazilian Portuguese' : 'English';
}

const SYSTEM = `You are a senior scientometrics researcher building a thematic taxonomy for a bibliometric study.
You always answer with a single valid JSON object and nothing else.`;

// ---------------------------------------------------------------------------------------
// Descoberta

export interface DiscoveryOptions {
  locale: PromptLocale;
  maxCategories: number;
  /** Foco da pesquisa informado pelo usuário, opcional. */
  focus?: string;
}

export function buildDiscoveryPrompt(docs: readonly HybridDoc[], options: DiscoveryOptions): ChatPrompt {
  const focus = options.focus?.trim();
  return {
    system: SYSTEM,
    user: `Below is a stratified sample of ${docs.length} scientific documents from a larger corpus.
${focus ? `The researcher describes the study focus as: "${focus}".\n` : ''}
Task:
1. Propose between 3 and ${options.maxCategories} mutually exclusive research themes that together cover at least 90% of the sample.
2. Assign every document of the sample to exactly one theme, or to "OTHER" if none fits.

Rules for each theme:
- "name": at most 5 words, in ${languageName(options.locale)}, specific to this corpus (avoid generic names like "Research" or "Studies").
- "what": one sentence in English stating what documents belong to the theme.
- "not_for": one sentence in English naming the neighbouring themes or topics it must NOT be confused with.
- "examples": up to ${MAX_EXAMPLES} document ids from the sample (like "D4") that are clear, typical members.
- Themes must not overlap: a document should clearly belong to a single theme.

Respond with JSON in exactly this shape:
{"categories":[{"name":"...","what":"...","not_for":"...","examples":["D1"]}],"assignments":{"D1":"<theme name or OTHER>"}}

Sample:
${docs.map(describeDoc).join('\n\n')}`,
  };
}

// ---------------------------------------------------------------------------------------
// Esclarecimento das descrições (após discordância na validação)

export interface ConfusionCase {
  doc: HybridDoc;
  referenceId: string;
  classifierId: string;
}

export function buildClarifyPrompt(
  categories: readonly HybridCategory[],
  cases: readonly ConfusionCase[],
  otherName: string,
): ChatPrompt {
  const nameOf = (id: string) =>
    id === OTHER_CATEGORY_ID ? otherName : (categories.find((c) => c.id === id)?.name ?? id);

  return {
    system: SYSTEM,
    user: `A fast classifier applied the taxonomy below to documents that you had already labelled. On the cases listed, it disagreed with your labels. The classifier reads descriptions very literally, so disagreements usually mean a "what" or "not_for" description is ambiguous.

Task: rewrite "what" and "not_for" (in English) so that the boundary between the confused themes is explicit. Keep every "id" and "name" exactly as given; do not add or remove themes. You may replace "examples" with document ids from the cases below (up to ${MAX_EXAMPLES}).

Current taxonomy:
${JSON.stringify(categories.map(({ id, name, what, notFor }) => ({ id, name, what, not_for: notFor })))}

Disagreements:
${cases
  .map(
    (item, position) =>
      `${describeDoc(item.doc, position)}\nYour label: ${nameOf(item.referenceId)} | Classifier: ${nameOf(item.classifierId)}`,
  )
  .join('\n\n')}

Respond with JSON in exactly this shape:
{"categories":[{"id":"...","name":"...","what":"...","not_for":"...","examples":["D1"]}]}`,
  };
}

// ---------------------------------------------------------------------------------------
// Expansão (documentos que sobraram em "outro" ou com baixa confiança)

export function buildExpandPrompt(
  categories: readonly HybridCategory[],
  docs: readonly HybridDoc[],
  options: { locale: PromptLocale; maxNew: number },
): ChatPrompt {
  return {
    system: SYSTEM,
    user: `A classifier applied the taxonomy below to a corpus. The documents listed afterwards were left as "OTHER" or classified with low confidence.

Task:
1. Decide whether these documents reveal research themes that are MISSING from the taxonomy. Propose at most ${options.maxNew} new themes, only if each one groups at least 3 of the listed documents. Proposing zero new themes is a valid answer.
2. Assign each listed document to an existing theme, a new theme, or "OTHER".

Rules for new themes: "name" in ${languageName(options.locale)} with at most 5 words; "what" and "not_for" in English, one sentence each; "not_for" must mention the closest existing themes; "examples" are ids of the listed documents. New themes must not overlap existing ones.

Existing taxonomy:
${JSON.stringify(categories.map(({ name, what }) => ({ name, what })))}

Respond with JSON in exactly this shape:
{"new_categories":[{"name":"...","what":"...","not_for":"...","examples":["D1"]}],"assignments":{"D1":"<theme name or OTHER>"}}

Documents:
${docs.map(describeDoc).join('\n\n')}`,
  };
}

// ---------------------------------------------------------------------------------------
// Leitura das respostas

/** Resposta de modelo ilegível. Módulo puro: leva as duas línguas e a tela escolhe (`en`). */
export class TaxonomyParseError extends Error {
  constructor(
    message: string,
    readonly en: string = message,
  ) {
    super(message);
    this.name = 'TaxonomyParseError';
  }
}

/** Extrai o objeto JSON da resposta, tolerando cercas de código e texto ao redor. */
export function extractJson(raw: string): Record<string, unknown> {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end <= start) throw new TaxonomyParseError('O modelo não devolveu um objeto JSON.', 'The model did not return a JSON object.');
  try {
    const parsed: unknown = JSON.parse(raw.slice(start, end + 1));
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new TaxonomyParseError('O JSON devolvido não é um objeto.', 'The returned JSON is not an object.');
    }
    return parsed as Record<string, unknown>;
  } catch (cause) {
    if (cause instanceof TaxonomyParseError) throw cause;
    throw new TaxonomyParseError('O JSON devolvido pelo modelo está malformado.', 'The JSON returned by the model is malformed.');
  }
}

/** Identificador estável a partir do nome: minúsculas, sem acento, só [a-z0-9-]. */
export function slugify(name: string): string {
  const slug = name
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'categoria';
}

function uniqueId(base: string, taken: Set<string>): string {
  let id = base;
  let suffix = 2;
  while (taken.has(id) || id === OTHER_CATEGORY_ID) id = `${base}-${suffix++}`;
  taken.add(id);
  return id;
}

function cleanSentence(value: unknown, limit: number): string {
  return String(value ?? '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, limit);
}

function cleanName(value: unknown): string {
  return String(value ?? '')
    .replace(/[\n\r"*`]+/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/[.:;,]+$/, '')
    .trim()
    .slice(0, 60);
}

const isOther = (label: string, otherName: string) => {
  const normalized = label.trim().toLowerCase();
  return (
    normalized === 'other' ||
    normalized === 'outro' ||
    normalized === 'outros' ||
    normalized === otherName.trim().toLowerCase()
  );
};

/** Converte ids de exemplo ("D4") nos títulos correspondentes da lista enviada. */
function resolveExamples(value: unknown, docs: readonly HybridDoc[]): string[] {
  if (!Array.isArray(value)) return [];
  const titles: string[] = [];
  for (const item of value) {
    const match = /^D(\d+)$/i.exec(String(item).trim());
    if (!match) continue;
    const doc = docs[Number(match[1]) - 1];
    if (doc?.title) titles.push(truncate(doc.title, EXAMPLE_TITLE_LIMIT));
    if (titles.length >= MAX_EXAMPLES) break;
  }
  return [...new Set(titles)];
}

function parseCategoryList(
  value: unknown,
  docs: readonly HybridDoc[],
  takenIds: Set<string>,
  takenNames: Set<string>,
  otherName: string,
  limit: number,
): HybridCategory[] {
  if (!Array.isArray(value)) return [];
  const categories: HybridCategory[] = [];

  for (const item of value) {
    if (categories.length >= limit) break;
    if (typeof item !== 'object' || item === null) continue;
    const record = item as Record<string, unknown>;
    const name = cleanName(record['name']);
    // Nome vazio, repetido ou igual ao de "outro" quebraria a chave única do Choice.
    if (!name || takenNames.has(name.toLowerCase()) || isOther(name, otherName)) continue;
    takenNames.add(name.toLowerCase());
    categories.push({
      id: uniqueId(slugify(name), takenIds),
      name,
      what: cleanSentence(record['what'], 400),
      notFor: cleanSentence(record['not_for'] ?? record['notFor'], 400),
      examples: resolveExamples(record['examples'], docs),
    });
  }

  return categories;
}

/** Rótulos por documento (posição na lista enviada → id da categoria). */
function parseAssignments(
  value: unknown,
  docs: readonly HybridDoc[],
  categories: readonly HybridCategory[],
  otherName: string,
): Map<number, string> {
  const labels = new Map<number, string>();
  if (typeof value !== 'object' || value === null) return labels;

  const byName = new Map(categories.map((category) => [category.name.toLowerCase(), category.id]));
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    const match = /^D(\d+)$/i.exec(key.trim());
    if (!match) continue;
    const doc = docs[Number(match[1]) - 1];
    if (!doc) continue;
    const label = String(raw ?? '').trim();
    // Rótulo que não corresponde a nenhuma categoria vale como "outro", não como erro.
    labels.set(doc.index, isOther(label, otherName) ? OTHER_CATEGORY_ID : (byName.get(label.toLowerCase()) ?? OTHER_CATEGORY_ID));
  }
  return labels;
}

export interface ParsedDiscovery {
  categories: HybridCategory[];
  /** Posição do documento na base → id da categoria atribuída pelo DeepSeek. */
  labels: Map<number, string>;
}

export function parseDiscovery(
  raw: string,
  docs: readonly HybridDoc[],
  options: { maxCategories: number; otherName: string },
): ParsedDiscovery {
  const json = extractJson(raw);
  const categories = parseCategoryList(
    json['categories'],
    docs,
    new Set(),
    new Set(),
    options.otherName,
    options.maxCategories,
  );
  if (categories.length < 2) {
    throw new TaxonomyParseError(
      'O modelo propôs menos de duas categorias válidas.',
      'The model proposed fewer than two valid categories.',
    );
  }
  return { categories, labels: parseAssignments(json['assignments'], docs, categories, options.otherName) };
}

/**
 * Aplica as descrições reescritas. Só `what`, `not_for` e exemplos mudam: ids e nomes são
 * preservados mesmo que o modelo tente alterá-los, e categorias ausentes na resposta
 * continuam como estavam.
 */
export function parseClarify(
  raw: string,
  current: readonly HybridCategory[],
  caseDocs: readonly HybridDoc[],
): HybridCategory[] {
  const json = extractJson(raw);
  const revised = new Map<string, Record<string, unknown>>();
  if (Array.isArray(json['categories'])) {
    for (const item of json['categories']) {
      if (typeof item === 'object' && item !== null) {
        const id = String((item as Record<string, unknown>)['id'] ?? '');
        if (id) revised.set(id, item as Record<string, unknown>);
      }
    }
  }

  return current.map((category) => {
    const update = revised.get(category.id);
    if (!update) return category;
    const what = cleanSentence(update['what'], 400);
    const notFor = cleanSentence(update['not_for'] ?? update['notFor'], 400);
    const examples = resolveExamples(update['examples'], caseDocs);
    return {
      ...category,
      what: what || category.what,
      notFor: notFor || category.notFor,
      examples: examples.length > 0 ? examples : category.examples,
    };
  });
}

export interface ParsedExpansion {
  newCategories: HybridCategory[];
  labels: Map<number, string>;
}

export function parseExpansion(
  raw: string,
  current: readonly HybridCategory[],
  docs: readonly HybridDoc[],
  options: { maxNew: number; otherName: string },
): ParsedExpansion {
  const json = extractJson(raw);
  const newCategories = parseCategoryList(
    json['new_categories'],
    docs,
    new Set(current.map((category) => category.id)),
    new Set(current.map((category) => category.name.toLowerCase())),
    options.otherName,
    options.maxNew,
  );
  return {
    newCategories,
    labels: parseAssignments(json['assignments'], docs, [...current, ...newCategories], options.otherName),
  };
}
