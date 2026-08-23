/**
 * Helpers de texto que replicam a semântica do pandas/Python usada em utils.py.
 * Estas funções são a base de toda a paridade: se `titleCase` divergir do
 * `str.title()` do Python, todas as tabelas de entidades divergem junto.
 */

/**
 * Replica `str.title()` do Python.
 *
 * A definição do CPython é independente de idioma: uma "palavra" é um grupo de letras
 * consecutivas. Logo dígitos e pontuação encerram a palavra — `"abc123def"` vira
 * `"Abc123Def"` e `"silva, a.b."` vira `"Silva, A.B."`.
 *
 * NÃO substituir por um regex `\w\S*`: `\w` inclui dígitos e underscore, o que produz
 * `"Abc123def"` e quebra a paridade com as tabelas geradas pelo pandas.
 */
export function titleCase(value: string): string {
  let out = '';
  let prevIsLetter = false;

  for (const char of value) {
    const isLetter = /\p{L}/u.test(char);
    out += isLetter && !prevIsLetter ? char.toUpperCase() : char.toLowerCase();
    prevIsLetter = isLetter;
  }

  return out;
}

/**
 * ⇄ `_split_semicolon_tokens` (utils.py:533).
 * Separa por ';', descarta vazios e aplica o caso pedido.
 */
export function splitTokens(
  value: unknown,
  transform?: 'lower' | 'title',
): string[] {
  if (value === null || value === undefined) return [];

  const tokens = String(value)
    .split(';')
    .map((token) => token.trim())
    .filter((token) => token.length > 0);

  if (transform === 'lower') return tokens.map((t) => t.toLowerCase());
  if (transform === 'title') return tokens.map(titleCase);
  return tokens;
}

/** ⇄ `_join_sorted` (utils.py:545): únicos, ordenados, unidos por `sep`. */
export function joinSorted(values: Iterable<string>, sep = ', '): string {
  const cleaned = new Set<string>();
  for (const value of values) {
    const trimmed = String(value).trim();
    if (trimmed) cleaned.add(trimmed);
  }
  return [...cleaned].sort().join(sep);
}

/**
 * Valores que o pandas produz ao converter nulos para texto. O Python filtra com
 * `!= 'Nan'` depois de `.str.title()`, e `'nan'.title()` é `'Nan'`.
 */
const NULL_LIKE = new Set(['', 'nan', 'none', 'nat', '<na>']);

/** True para os textos que o pipeline Python descarta como ausentes. */
export function isNullLike(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === 'number' && Number.isNaN(value)) return true;
  return NULL_LIKE.has(String(value).trim().toLowerCase());
}

/**
 * ⇄ `pd.to_numeric(..., errors='coerce')`: devolve `null` quando não é número.
 * Aceita o número já tipado, evitando o custo de String() no caminho quente.
 */
export function toNumeric(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (value === null || value === undefined) return null;

  const text = String(value).trim();
  if (!text) return null;

  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Primeiro candidato presente entre as chaves do objeto — ⇄ `_pick_column`. */
export function pickField<T extends string>(
  row: Record<string, unknown> | undefined,
  candidates: readonly T[],
): T | null {
  if (!row) return null;
  return candidates.find((candidate) => candidate in row) ?? null;
}

/** Primeiro candidato presente no conjunto de colunas do dataset. */
export function pickColumn<T extends string>(
  columns: ReadonlySet<string>,
  candidates: readonly T[],
): T | null {
  return candidates.find((candidate) => columns.has(candidate)) ?? null;
}

/** União de todas as chaves presentes no dataset — ⇄ `df.columns`. */
export function collectColumns(rows: readonly Record<string, unknown>[]): Set<string> {
  const columns = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) columns.add(key);
  }
  return columns;
}
