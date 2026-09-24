import { isNullLike } from '@/core/text';

/** Link resolvível para o DOI de um documento, ou null quando o campo está vazio. */
export function cleanDoiUrl(rawDoi: unknown): string | null {
  if (!rawDoi || typeof rawDoi !== 'string') return null;
  const trimmed = rawDoi.trim();
  if (!trimmed || isNullLike(trimmed)) return null;
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return trimmed;
  const clean = trimmed.replace(/^doi:\s*/i, '');
  return `https://doi.org/${clean}`;
}
