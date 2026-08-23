import { readFileSync } from 'node:fs';
import { fileURLToPath, URL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { processRisFiles, type RisSource } from '@/core/parsers/pipeline-ris';
import {
  boxplotOptions,
  boxplotSeries,
  MAX_BOXPLOT_ITEMS,
  type BoxplotMetric,
} from '@/core/viz/boxplot';
import { circularPositions, collaborationNetwork } from '@/core/viz/collaboration';
import { conceptMap } from '@/core/viz/concept-map';
import { keywordGenetics } from '@/core/viz/genetics';
import { historiograph } from '@/core/viz/historiograph';
import { sankeyEvolution, suggestPeriods } from '@/core/viz/sankey';
import { thematicMap } from '@/core/viz/thematic-map';
import type { Dataset } from '@/lib/types';

/**
 * Segunda onda de visualizações.
 *
 * Estes testes não comparam com o Python — várias destas visualizações divergem dele de
 * propósito, e duas nem chegam a rodar lá. O que se verifica aqui são as INVARIANTES de
 * cada estrutura: coisas que, se quebrarem, produzem um gráfico visualmente plausível e
 * silenciosamente errado.
 */

const repoRoot = fileURLToPath(new URL('../../..', import.meta.url));

const sources: RisSource[] = [
  { name: 'scielo.ris', database: 'SciELO' },
  { name: 'wos.ris', database: 'Web of Science' },
  { name: 'scopus.ris', database: 'Scopus' },
].map(({ name, database }) => ({
  name,
  database,
  text: readFileSync(`${repoRoot}/${name}`, 'utf8'),
}));

const dataset: Dataset = processRisFiles(sources);

describe('genética das ideias', () => {
  const genetics = keywordGenetics(dataset);

  it('extrai o ciclo de vida de cada palavra-chave', () => {
    expect(genetics.length).toBeGreaterThan(1000);
  });

  it('mantém nascimento antes da última aparição, com longevidade coerente', () => {
    for (const term of genetics) {
      expect(term.birthYear, term.keyword).toBeLessThanOrEqual(term.lastYear);
      expect(term.lifespan, term.keyword).toBe(term.lastYear - term.birthYear);
      expect(term.occurrences, term.keyword).toBeGreaterThan(0);
    }
  });

  it('ordena por replicação decrescente', () => {
    const counts = genetics.map((term) => term.occurrences);
    expect([...counts].sort((a, b) => b - a)).toEqual(counts);
  });
});

describe('fluxo de evolução temática', () => {
  const periods = suggestPeriods(dataset);

  it('divide o intervalo em três períodos contíguos e sem sobreposição', () => {
    expect(periods).not.toBeNull();
    if (!periods) return;

    const [first, second, third] = periods;
    expect(first[1]).toBeLessThan(second[0]);
    expect(second[1]).toBeLessThan(third[0]);
  });

  it('liga apenas períodos adjacentes, nunca o primeiro ao terceiro', () => {
    if (!periods) return;
    const data = sankeyEvolution(dataset, periods, 10);
    expect(data).not.toBeNull();
    if (!data) return;

    for (const link of data.links) {
      const source = data.nodes[link.source];
      const target = data.nodes[link.target];
      expect(source, 'nó de origem existe').toBeDefined();
      expect(target, 'nó de destino existe').toBeDefined();
      // Um salto de dois períodos indicaria erro de indexação dos nós.
      expect((target?.period ?? 0) - (source?.period ?? 0)).toBe(1);
    }
  });

  it('dá às ligações de continuidade peso maior que às de interseção', () => {
    if (!periods) return;
    const data = sankeyEvolution(dataset, periods, 10);
    if (!data) return;

    const continuity = data.links.filter((link) => link.kind === 'continuidade');
    const intersection = data.links.filter((link) => link.kind === 'intersecção');

    expect(continuity.length).toBeGreaterThan(0);
    // A continuidade precisa dominar visualmente: é ela que mostra o tema sobrevivendo.
    const maxContinuity = Math.max(...continuity.map((link) => link.value));
    const maxIntersection = Math.max(...intersection.map((link) => link.value), 0);
    expect(maxContinuity).toBeGreaterThan(maxIntersection);
  });
});

describe('mapa conceitual', () => {
  const terms = conceptMap(dataset, { topTerms: 50, clusters: 4 });

  it('projeta os termos pedidos em três dimensões', () => {
    expect(terms).toHaveLength(50);
    for (const term of terms) {
      expect(Number.isFinite(term.x), term.term).toBe(true);
      expect(Number.isFinite(term.y), term.term).toBe(true);
      expect(Number.isFinite(term.z), term.term).toBe(true);
    }
  });

  /**
   * O agrupamento precisa ser informativo, não apenas existir.
   *
   * Sem a normalização dos vetores, o K-Means separa por frequência do termo e produz
   * 45 dos 50 termos num só grupo — um gráfico que renderiza perfeitamente e não diz
   * nada. Este teste é o que protege essa decisão de ser desfeita sem querer.
   */
  it('distribui os termos entre os agrupamentos, sem concentrar tudo num só', () => {
    const sizes = new Map<number, number>();
    for (const term of terms) sizes.set(term.cluster, (sizes.get(term.cluster) ?? 0) + 1);

    const largest = Math.max(...sizes.values());
    expect(sizes.size).toBeGreaterThan(1);
    expect(largest, 'maior agrupamento não pode conter quase tudo').toBeLessThan(
      terms.length * 0.7,
    );
  });

  it('é determinístico entre execuções', () => {
    const again = conceptMap(dataset, { topTerms: 50, clusters: 4 });
    expect(again.map((term) => term.cluster)).toEqual(terms.map((term) => term.cluster));
  });
});

describe('colaboração internacional', () => {
  const network = collaborationNetwork(dataset, 30);

  it('recorta os países mais produtivos e liga apenas entre eles', () => {
    expect(network.nodes).toHaveLength(30);

    const countries = new Set(network.nodes.map((node) => node.country));
    for (const edge of network.edges) {
      expect(countries.has(edge.source), edge.source).toBe(true);
      expect(countries.has(edge.target), edge.target).toBe(true);
    }
  });

  it('resolve coordenadas para todos os países exibidos', () => {
    // Sem coordenada o país é pintado no mapa mas fica sem arestas, o que sugere
    // isolamento onde há apenas uma lacuna na tabela.
    const semCoordenada = network.nodes.filter((node) => node.latitude === null);
    expect(semCoordenada.map((node) => node.country)).toEqual([]);
  });

  it('usa nome apresentável no rótulo, e não a chave interna', () => {
    const usa = network.nodes.find((node) => node.country.toLowerCase() === 'usa');
    expect(usa?.label).toBe('United States');
    expect(usa?.plotlyName).toBe('United States');
  });

  it('posiciona os nós sobre um círculo de raio unitário', () => {
    const positions = circularPositions(network.nodes);
    expect(positions.size).toBe(network.nodes.length);

    for (const [country, position] of positions) {
      expect(Math.hypot(position.x, position.y), country).toBeCloseTo(1, 10);
    }
  });

  it('dá a cada país uma posição distinta no círculo', () => {
    const positions = circularPositions(network.nodes);
    const unique = new Set([...positions.values()].map((p) => `${p.x.toFixed(6)},${p.y.toFixed(6)}`));
    expect(unique.size).toBe(network.nodes.length);
  });
});

describe('mapa temático', () => {
  const map = thematicMap(dataset, 'ABSTRACT', 150);

  it('forma agrupamentos com centralidade e densidade finitas', () => {
    expect(map).not.toBeNull();
    if (!map) return;

    expect(map.clusters.length).toBeGreaterThan(1);
    for (const cluster of map.clusters) {
      expect(Number.isFinite(cluster.centrality), cluster.label).toBe(true);
      expect(Number.isFinite(cluster.density), cluster.label).toBe(true);
      expect(cluster.terms.length).toBeGreaterThan(0);
    }
  });

  it('posiciona as médias dentro da faixa dos agrupamentos', () => {
    if (!map) return;

    const centralities = map.clusters.map((cluster) => cluster.centrality);
    const densities = map.clusters.map((cluster) => cluster.density);

    // As médias desenham as linhas dos quadrantes; fora da faixa, todos os pontos
    // cairiam no mesmo quadrante e a leitura perderia o sentido.
    expect(map.meanCentrality).toBeGreaterThanOrEqual(Math.min(...centralities));
    expect(map.meanCentrality).toBeLessThanOrEqual(Math.max(...centralities));
    expect(map.meanDensity).toBeGreaterThanOrEqual(Math.min(...densities));
    expect(map.meanDensity).toBeLessThanOrEqual(Math.max(...densities));
  });
});

describe('historiograph', () => {
  it('devolve nulo quando a base não traz referências citadas', () => {
    // Os três .ris de exemplo não têm tag CR. O painel precisa distinguir "sem dados"
    // de "erro" para poder explicar ao usuário como exportar com as referências.
    expect(historiograph(dataset, 30)).toBeNull();
  });

  it('monta a linhagem quando há referências, citando apenas trabalhos anteriores', () => {
    const withReferences: Dataset = [
      {
        TITLE: 'Trabalho seminal',
        AUTHORS: 'Dawkins, R',
        'YEAR CLEAN': 1976,
        'TOTAL CITATIONS': 500,
        REFERENCES_UNIFIED: 'algo anterior',
      },
      {
        TITLE: 'Continuação',
        AUTHORS: 'Blackmore, S',
        'YEAR CLEAN': 1999,
        'TOTAL CITATIONS': 300,
        REFERENCES_UNIFIED: 'Dawkins, R, 1976, The Selfish Gene',
      },
      {
        TITLE: 'Anterior que não pode citar o futuro',
        AUTHORS: 'Antigo, A',
        'YEAR CLEAN': 1970,
        'TOTAL CITATIONS': 10,
        REFERENCES_UNIFIED: 'Blackmore, S, 1999',
      },
    ] as unknown as Dataset;

    const graph = historiograph(withReferences, 10);
    expect(graph).not.toBeNull();
    if (!graph) return;

    expect(graph.edges).toHaveLength(1);
    expect(graph.edges[0]?.from).toContain('Blackmore');
    expect(graph.edges[0]?.to).toContain('Dawkins');
  });
});

describe('distribuição comparativa', () => {
  const options = boxplotOptions(dataset, 'Países');

  it('oferece as entidades mais frequentes, em ordem', () => {
    expect(options.length).toBeGreaterThan(10);
    expect(options[0]).toBe('China');
  });

  it.each<BoxplotMetric>([
    'Citações por documento',
    'Citações por autor',
    'Citações por ano',
    'Documentos por autor',
    'Documentos por ano',
  ])('%s produz uma série por entidade selecionada', (metric) => {
    const selected = options.slice(0, MAX_BOXPLOT_ITEMS);
    const series = boxplotSeries(dataset, 'Países', metric, selected);

    expect(series.map((entry) => entry.entity)).toEqual(selected);

    for (const entry of series) {
      expect(entry.values.length, `${metric} / ${entry.entity}`).toBeGreaterThan(0);
      // Um rótulo por ponto: sem isso o tooltip mostra o dado de outra observação.
      expect(entry.labels.length, `${metric} / ${entry.entity}`).toBe(entry.values.length);
      for (const value of entry.values) {
        expect(Number.isFinite(value), `${metric} / ${entry.entity}`).toBe(true);
      }
    }
  });

  it('preserva a ordem de seleção do usuário', () => {
    const selected = [options[2], options[0], options[1]].filter(
      (entity): entity is string => entity !== undefined,
    );
    const series = boxplotSeries(dataset, 'Países', 'Citações por documento', selected);
    expect(series.map((entry) => entry.entity)).toEqual(selected);
  });
});
