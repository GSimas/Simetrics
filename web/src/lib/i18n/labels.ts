import type { BoxplotDimension, BoxplotMetric } from '@/core/viz/boxplot';
import type { CooccurrenceKind } from '@/core/graph/build';
import type { SizeMetric } from '@/core/graph';
import type { NodeKind, SearchEntityType } from '@/lib/types';
import type { Locale } from './translations';

/**
 * Rótulos de exibição dos valores internos que estão em português (tipos de entidade,
 * tipos de rede, métricas…). Os valores continuam em português porque são chaves de dados
 * e de navegação; só o que aparece na tela passa por aqui.
 */

const ENTITY_TYPE_EN: Record<SearchEntityType | NodeKind | 'Outro', string> = {
  Documento: 'Document',
  Autor: 'Author',
  País: 'Country',
  'Local de Publicação (Venue)': 'Venue',
  'Palavra-chave': 'Keyword',
  Tema: 'Theme',
  Outro: 'Other',
};

const COOCCURRENCE_EN: Record<CooccurrenceKind, string> = {
  Coautoria: 'Co-authorship',
  'Palavras-chave': 'Keywords',
  Países: 'Countries',
};

const SIZE_METRIC_EN: Record<SizeMetric, string> = {
  'Tamanho Fixo': 'Fixed size',
  'Grau Absoluto': 'Absolute degree',
  'Centralidade (Eigen)': 'Centrality (eigenvector)',
  Betweenness: 'Betweenness',
  Closeness: 'Closeness',
};

const BOXPLOT_DIMENSION_EN: Record<BoxplotDimension, string> = {
  Países: 'Countries',
  'Palavras-chave': 'Keywords',
  'Temas (IA)': 'Themes (AI)',
};

const BOXPLOT_METRIC_EN: Record<BoxplotMetric, string> = {
  'Documentos por autor': 'Documents per author',
  'Documentos por ano': 'Documents per year',
  'Citações por documento': 'Citations per document',
  'Citações por autor': 'Citations per author',
  'Citações por ano': 'Citations per year',
};

const pick = <K extends string>(map: Record<K, string>) => (value: K, locale: Locale): string =>
  locale === 'en' ? (map[value] ?? value) : value;

export const entityTypeLabel = pick(ENTITY_TYPE_EN);
export const cooccurrenceLabel = pick(COOCCURRENCE_EN);
export const sizeMetricLabel = pick(SIZE_METRIC_EN);
export const boxplotDimensionLabel = pick(BOXPLOT_DIMENSION_EN);
export const boxplotMetricLabel = pick(BOXPLOT_METRIC_EN);

/** Formatação de números no idioma da interface. */
export function numberLocale(locale: Locale): string {
  return locale === 'en' ? 'en-US' : 'pt-BR';
}
