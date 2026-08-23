/**
 * Benchmark do pipeline no teto de 10.000 documentos.
 *
 * Fica fora da suíte de testes de propósito: medição de tempo é sensível à máquina e
 * viraria um teste intermitente em CI. Rode manualmente ao mexer em algo do caminho
 * quente — ingestão, deduplicação ou agregação das tabelas.
 *
 *     npx vite-node scripts/benchmark.ts
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath, URL } from 'node:url';

import { dedupByDoi, dedupBySimilarity } from '@/core/dedup';
import {
  analyzeCooccurrenceNetwork,
  analyzeHeterogeneousNetwork,
  betweennessCentrality,
  buildHeterogeneousGraph,
  planBetweenness,
  toCompact,
} from '@/core/graph';
import { processRisFiles } from '@/core/parsers/pipeline-ris';
import { lotkaDistribution } from '@/core/scientometrics';
import { docsPerAuthor, metadataCompleteness, summarize } from '@/core/summary';
import { authorsTable, countriesTable, keywordsTable, venuesTable } from '@/core/tables';
import { MAX_DOCUMENTS } from '@/lib/schema';
import type { Dataset } from '@/lib/types';

const repoRoot = fileURLToPath(new URL('../..', import.meta.url));
const BASE_YEAR = 2026;

function timed<T>(label: string, run: () => T): T {
  const started = performance.now();
  const result = run();
  const elapsed = (performance.now() - started).toFixed(0);
  console.log(`  ${label.padEnd(34)} ${elapsed.padStart(6)} ms`);
  return result;
}

/**
 * Monta um corpus sintético de 10.000 registros a partir dos arquivos reais.
 *
 * Título, DOI e citações variam por cópia. Sem isso o corpus vira uma pilha de
 * duplicatas exatas, a deduplicação corta quase tudo logo na primeira etapa e o
 * benchmark deixa de medir justamente o caminho caro.
 */
function buildCorpus(): string {
  const sources = ['wos.ris', 'scopus.ris', 'scielo.ris']
    .map((name) => readFileSync(`${repoRoot}/${name}`, 'utf8'))
    .join('\n');

  const records = sources.split(/(?=TY {2}- )/).filter((record) => record.includes('ER  -'));
  const out: string[] = [];

  let counter = 0;
  while (out.length < MAX_DOCUMENTS) {
    for (const record of records) {
      if (out.length >= MAX_DOCUMENTS) break;
      counter += 1;
      out.push(
        record
          .replace(/^TI {2}- (.*)$/m, (_match, title: string) => `TI  - ${title} [v${counter}]`)
          .replace(/^DO {2}- (.*)$/m, (_match, doi: string) => `DO  - ${doi}.v${counter}`)
          .replace(/^TC {2}- \d+/m, () => `TC  - ${(counter * 7) % 400}`),
      );
    }
  }

  return out.join('');
}

function main(): void {
  const text = buildCorpus();
  console.log(`\nCorpus sintético: ${(text.length / 1_048_576).toFixed(1)} MB\n`);

  const dataset: Dataset = timed('ingestão (parse + normalize)', () =>
    processRisFiles([{ name: 'benchmark.ris', text, database: 'Web of Science' }]),
  );
  console.log(`  -> ${dataset.length} documentos\n`);

  const byDoi = timed('dedup por DOI', () => dedupByDoi(dataset));
  const bySimilarity = timed('dedup por similaridade (0.90)', () =>
    dedupBySimilarity(dataset, { threshold: 0.9 }),
  );
  console.log(
    `  -> DOI removeu ${byDoi.removed.length}, similaridade removeu ${bySimilarity.removed.length}\n`,
  );

  timed('resumo + completude', () => {
    summarize(dataset, BASE_YEAR);
    metadataCompleteness(dataset);
  });
  timed('Lei de Lotka', () => lotkaDistribution(docsPerAuthor(dataset)));

  const authors = timed('tabela de autores', () => authorsTable(dataset, BASE_YEAR));
  const countries = timed('tabela de países', () => countriesTable(dataset, BASE_YEAR));
  const venues = timed('tabela de venues', () => venuesTable(dataset, BASE_YEAR));
  const keywords = timed('tabela de keywords', () => keywordsTable(dataset, BASE_YEAR));

  console.log(
    `\n  linhas: autores=${authors.length} países=${countries.length} ` +
      `venues=${venues.length} keywords=${keywords.length}\n`,
  );

  const compact = toCompact(buildHeterogeneousGraph(dataset).graph);
  const plan = planBetweenness(compact);
  console.log(`  grafo heterogêneo: ${compact.order} nós, ${compact.size} arestas`);
  console.log(`  estratégia de betweenness: ${plan.exact ? 'exata' : `amostrada (k=${plan.sampleSize})`}\n`);

  timed('betweenness amostrado (k=100)', () =>
    betweennessCentrality(compact, { sampleSize: 100, seed: 42 }),
  );
  timed('análise SNA completa', () => analyzeHeterogeneousNetwork(dataset));
  timed('rede de coautoria (top 50)', () =>
    analyzeCooccurrenceNetwork(dataset, 'Coautoria', 50, 'Grau Absoluto'),
  );
  console.log();
}

main();
