import type { WorkerProgress } from '@/lib/types';
import type { Locale } from './translations';

/**
 * Tradução das fases de progresso na hora de exibir. Workers e parsers escrevem as fases
 * em português (não têm acesso ao idioma da interface); a tela traduz aqui.
 *
 * ponytail: dicionário de frases fixas; uma fase nova sem entrada aparece em português.
 * O teste em tests/progress-i18n.test.ts varre o código e falha se alguma faltar.
 */
export const PHASES_EN: Record<string, string> = {
  // state/dataset.store.ts
  'Lendo arquivos': 'Reading files',
  'Baixando bases de exemplo': 'Downloading sample datasets',
  Deduplicando: 'Deduplicating',
  'Iniciando análise de redes': 'Starting network analysis',
  'Agrupando documentos': 'Grouping documents',
  'Nomeando temas': 'Naming themes',
  // workers/ingest.worker.ts
  'Consolidando estrutura': 'Consolidating structure',
  'Comparando títulos': 'Comparing titles',
  'Deduplicando por DOI': 'Deduplicating by DOI',
  'Comparando títulos por similaridade': 'Comparing titles by similarity',
  // core/parsers/*
  'Lendo arquivos RIS': 'Reading RIS files',
  'Resolvendo citações, tipos e países': 'Resolving citations, types and countries',
  'Padronizando estrutura': 'Standardizing structure',
  'Lendo Cochrane': 'Reading Cochrane',
  'Lendo CSV (Scopus)': 'Reading CSV (Scopus)',
  'Lendo PubMed/Medline': 'Reading PubMed/Medline',
  'Lendo Excel (Web of Science)': 'Reading Excel (Web of Science)',
  // core/clustering.ts
  'Vetorizando textos': 'Vectorizing texts',
  'Comprimindo semântica (LSA)': 'Compressing semantics (LSA)',
  'Selecionando documentos representativos': 'Selecting representative documents',
  // core/graph/index.ts
  'Mapeando topologia': 'Mapping topology',
  'Calculando centralidades': 'Computing centralities',
  'Betweenness (exato)': 'Betweenness (exact)',
  'Betweenness (amostrado)': 'Betweenness (sampled)',
  Betweenness: 'Betweenness',
  'Métricas de ecologia profunda': 'Deep ecology metrics',
  Concluído: 'Done',
};

type Pattern = [RegExp, (...groups: string[]) => string];

/** Fases montadas com template literal. */
const PHASE_PATTERNS: Pattern[] = [[/^Testando (\d+) agrupamentos$/, (k) => `Testing ${k} clusters`]];

/** Padrões dos detalhes com números ("12 documentos", "3 de 10"…). */
const DETAIL_PATTERNS: Pattern[] = [
  [/^([\d.,]+) documentos$/, (n) => `${n} documents`],
  [/^([\d.,]+) de ([\d.,]+) documentos$/, (a, b) => `${a} of ${b} documents`],
  [/^([\d.,]+) de ([\d.,]+)$/, (a, b) => `${a} of ${b}`],
];

function translate(text: string, patterns: Pattern[]): string {
  for (const [pattern, render] of patterns) {
    const match = pattern.exec(text);
    if (match) return render(...match.slice(1));
  }
  return text;
}

export function localizeProgress<T extends Pick<WorkerProgress, 'phase' | 'detail'>>(progress: T, locale: Locale): T {
  if (locale !== 'en') return progress;
  const phase = PHASES_EN[progress.phase] ?? translate(progress.phase, PHASE_PATTERNS);
  const detail = progress.detail === undefined ? undefined : translate(progress.detail, DETAIL_PATTERNS);
  return { ...progress, phase, ...(detail !== undefined ? { detail } : {}) };
}
