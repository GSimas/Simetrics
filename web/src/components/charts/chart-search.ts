/**
 * Correspondência da busca dentro dos gráficos (ver ChartSearch.tsx).
 */

/** Sem acento e sem caixa: "sao paulo" encontra "São Paulo". */
export function normalizeSearch(text: string): string {
  return text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase().trim();
}

/** Chaves cujo rótulo contém a busca; `null` quando a busca está vazia. */
export function matchKeys<T>(items: readonly T[], query: string, keyOf: (item: T) => string, labelOf: (item: T) => string): Set<string> | null {
  const needle = normalizeSearch(query);
  if (!needle) return null;
  return new Set(items.filter((item) => normalizeSearch(labelOf(item)).includes(needle)).map(keyOf));
}
