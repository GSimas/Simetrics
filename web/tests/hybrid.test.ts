import { describe, expect, it } from 'vitest';

import { applyHybridThemes } from '@/core/hybrid/apply';
import type { HybridDoc } from '@/core/hybrid/documents';
import { toHybridDocs } from '@/core/hybrid/documents';
import { buildJevRequest, buildThemeQuestion, parseJevDecision, type JevResponseBody } from '@/core/hybrid/jev-request';
import { categoryCountsOf, compareLabels, coverageOf, leftoverIndices, tierOf, weakCategories } from '@/core/hybrid/metrics';
import {
  buildDiscoveryPrompt,
  extractJson,
  parseClarify,
  parseDiscovery,
  parseExpansion,
  slugify,
  TaxonomyParseError,
} from '@/core/hybrid/prompts';
import { hybridMethodsText } from '@/core/hybrid/report';
import { stratifiedSample } from '@/core/hybrid/sampling';
import { OTHER_CATEGORY_ID, type DocumentDecision, type HybridCategory, type HybridRun } from '@/core/hybrid/types';
import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';

function makeDocs(count: number, clusterOf: (i: number) => number = (i) => i % 3): { docs: HybridDoc[]; assignments: number[] } {
  const docs: HybridDoc[] = [];
  const assignments: number[] = [];
  for (let i = 0; i < count; i += 1) {
    const cluster = clusterOf(i);
    assignments.push(cluster);
    docs.push({
      index: i,
      title: `Document ${i} about topic ${cluster}`,
      keywords: `topic${cluster}; keyword${i}`,
      abstract: `A long enough abstract for document ${i} discussing topic${cluster} in reasonable depth.`,
      year: 2000 + (i % 20),
    });
  }
  return { docs, assignments };
}

const CATEGORIES: HybridCategory[] = [
  { id: 'vacinas', name: 'Vacinas', what: 'Vaccine studies', notFor: 'Drug trials', examples: [] },
  { id: 'farmacos', name: 'Fármacos', what: 'Drug trials', notFor: 'Vaccines', examples: ['A drug trial'] },
];

const decision = (categoryId: string, confidence: number): DocumentDecision => ({ categoryId, confidence, probability: confidence });

describe('stratifiedSample', () => {
  it('returns the whole corpus when it is smaller than the sample', () => {
    const { docs, assignments } = makeDocs(40);
    const result = stratifiedSample(docs, assignments, new Map(), { size: 120, seed: 42 });
    expect(result.indices).toHaveLength(40);
    expect(result.outliers).toBe(0);
  });

  it('is deterministic for a given seed and respects the size', () => {
    const { docs, assignments } = makeDocs(1000);
    const a = stratifiedSample(docs, assignments, new Map(), { size: 120, seed: 42 });
    const b = stratifiedSample(docs, assignments, new Map(), { size: 120, seed: 42 });
    expect(a.indices).toEqual(b.indices);
    expect(a.indices).toHaveLength(120);
    expect(new Set(a.indices).size).toBe(120);
  });

  it('covers a minority cluster that simple random sampling could miss', () => {
    // 2% da base num agrupamento minoritário (id 9).
    const { docs, assignments } = makeDocs(2000, (i) => (i % 50 === 0 ? 9 : i % 2));
    const result = stratifiedSample(docs, assignments, new Map(), { size: 60, seed: 7 });
    const minority = result.indices.filter((i) => assignments[i] === 9);
    expect(minority.length).toBeGreaterThan(0);
  });

  it('reserves a share of the sample to atypical documents', () => {
    const { docs, assignments } = makeDocs(600);
    const terms = new Map([
      [0, ['topic0']],
      [1, ['topic1']],
      [2, ['topic2']],
    ]);
    const result = stratifiedSample(docs, assignments, terms, { size: 100, seed: 1, outlierShare: 0.1 });
    expect(result.outliers).toBe(10);
    expect(result.indices).toHaveLength(100);
  });
});

describe('prompts and parsing', () => {
  const { docs } = makeDocs(5);

  it('asks for names in the interface language and descriptions in English', () => {
    const prompt = buildDiscoveryPrompt(docs, { locale: 'pt', maxCategories: 8, focus: 'saúde pública' });
    expect(prompt.user).toContain('Brazilian Portuguese');
    expect(prompt.user).toContain('"what": one sentence in English');
    expect(prompt.user).toContain('saúde pública');
    expect(prompt.user).toContain('[D5]');
  });

  it('extracts JSON wrapped in code fences', () => {
    expect(extractJson('```json\n{"a": 1}\n```')).toEqual({ a: 1 });
    expect(() => extractJson('no json here')).toThrow(TaxonomyParseError);
  });

  it('parses the discovery, deduplicating names and mapping labels and examples', () => {
    const raw = JSON.stringify({
      categories: [
        { name: 'Vacinas', what: 'Vaccine studies', not_for: 'Drug trials', examples: ['D1', 'D9', 'x'] },
        { name: 'vacinas', what: 'duplicate', not_for: '' },
        { name: 'Outros', what: 'reserved' },
        { name: 'Fármacos', what: 'Drug trials', not_for: 'Vaccines', examples: ['D2'] },
      ],
      assignments: { D1: 'Vacinas', D2: 'Fármacos', D3: 'OTHER', D4: 'Inexistente', D99: 'Vacinas' },
    });
    const parsed = parseDiscovery(raw, docs, { maxCategories: 10, otherName: 'Outros' });
    expect(parsed.categories.map((c) => c.name)).toEqual(['Vacinas', 'Fármacos']);
    expect(parsed.categories[1]?.id).toBe('farmacos');
    expect(parsed.categories[0]?.examples).toEqual([docs[0]?.title]);
    expect(parsed.labels.get(0)).toBe('vacinas');
    expect(parsed.labels.get(1)).toBe('farmacos');
    expect(parsed.labels.get(2)).toBe(OTHER_CATEGORY_ID);
    expect(parsed.labels.get(3)).toBe(OTHER_CATEGORY_ID);
    expect(parsed.labels.size).toBe(4);
  });

  it('rejects a discovery with fewer than two categories', () => {
    const raw = JSON.stringify({ categories: [{ name: 'Só uma' }], assignments: {} });
    expect(() => parseDiscovery(raw, docs, { maxCategories: 10, otherName: 'Outros' })).toThrow(TaxonomyParseError);
  });

  it('clarify only rewrites descriptions, keeping ids and names', () => {
    const raw = JSON.stringify({
      categories: [{ id: 'vacinas', name: 'Renamed!', what: 'Sharper vaccine definition', not_for: 'Anything about drugs' }],
    });
    const revised = parseClarify(raw, CATEGORIES, docs);
    expect(revised[0]).toMatchObject({ id: 'vacinas', name: 'Vacinas', what: 'Sharper vaccine definition' });
    expect(revised[1]).toEqual(CATEGORIES[1]);
  });

  it('expansion adds only new, non-conflicting categories', () => {
    const raw = JSON.stringify({
      new_categories: [
        { name: 'Vacinas', what: 'dup' },
        { name: 'Saúde Mental', what: 'Mental health', not_for: 'Vaccines', examples: ['D3'] },
      ],
      assignments: { D3: 'Saúde Mental', D4: 'Vacinas' },
    });
    const parsed = parseExpansion(raw, CATEGORIES, docs, { maxNew: 4, otherName: 'Outros' });
    expect(parsed.newCategories.map((c) => c.id)).toEqual(['saude-mental']);
    expect(parsed.labels.get(2)).toBe('saude-mental');
    expect(parsed.labels.get(3)).toBe('vacinas');
  });

  it('slugify strips accents and punctuation', () => {
    expect(slugify('Saúde Pública & IA')).toBe('saude-publica-ia');
  });
});

describe('Jev request', () => {
  it('builds a single choice question with "other" as the last option', () => {
    const question = buildThemeQuestion(CATEGORIES, 'Outros');
    expect(Object.keys(question.criteria)).toEqual(['Vacinas', 'Fármacos', 'Outros']);
    expect(question.criteria['Fármacos']).toEqual({ what: 'Drug trials', not_for: 'Vaccines', examples: ['A drug trial'] });

    const doc: HybridDoc = { index: 0, title: 'T', keywords: '', abstract: 'x'.repeat(5000), year: null };
    const body = buildJevRequest(doc, question, 'jev-latest');
    expect(body.model).toBe('jev-latest');
    expect(body.state.keywords).toBeUndefined();
    expect(body.state.abstract!.length).toBeLessThanOrEqual(2001);
  });

  it('maps the chosen option back to the category id', () => {
    const response: JevResponseBody = {
      model: 'jev-1.13.0',
      answers: { theme: { type: 'choice', choice: 'Fármacos', probabilities: { Vacinas: 0.1, Fármacos: 0.85, Outros: 0.05 }, confidence: 0.77 } },
    };
    expect(parseJevDecision(response, CATEGORIES, 'Outros')).toEqual({ categoryId: 'farmacos', confidence: 0.77, probability: 0.85 });

    const other: JevResponseBody = {
      model: 'jev-1.13.0',
      answers: { theme: { type: 'choice', choice: 'Outros', probabilities: { Outros: 1 }, confidence: 0.9 } },
    };
    expect(parseJevDecision(other, CATEGORIES, 'Outros').categoryId).toBe(OTHER_CATEGORY_ID);
  });
});

describe('metrics', () => {
  const thresholds = { accept: 0.7, review: 0.4 };

  it('routes by confidence', () => {
    expect(tierOf(decision('vacinas', 0.9), thresholds)).toBe('auto');
    expect(tierOf(decision('vacinas', 0.5), thresholds)).toBe('review');
    expect(tierOf(decision('vacinas', 0.1), thresholds)).toBe('unclassified');
  });

  it('computes agreement and Cohen kappa', () => {
    const reference = new Map([
      [0, 'vacinas'],
      [1, 'vacinas'],
      [2, 'farmacos'],
      [3, 'farmacos'],
    ]);
    const perfect = new Map([...reference].map(([i, id]) => [i, decision(id, 0.9)]));
    const round = compareLabels(reference, perfect, CATEGORIES, 'Outros', 1);
    expect(round.agreement).toBe(1);
    expect(round.kappa).toBe(1);

    const half = new Map([
      [0, decision('vacinas', 0.9)],
      [1, decision('farmacos', 0.9)],
      [2, decision('farmacos', 0.9)],
      [3, decision('vacinas', 0.9)],
    ]);
    const chance = compareLabels(reference, half, CATEGORIES, 'Outros', 1);
    expect(chance.agreement).toBe(0.5);
    expect(chance.kappa).toBeCloseTo(0);
  });

  it('flags weak categories with enough support', () => {
    const reference = new Map([
      [0, 'vacinas'],
      [1, 'vacinas'],
      [2, 'vacinas'],
      [3, 'farmacos'],
    ]);
    const decisions = new Map([
      [0, decision('farmacos', 0.9)],
      [1, decision('farmacos', 0.9)],
      [2, decision('vacinas', 0.9)],
      [3, decision('farmacos', 0.9)],
    ]);
    expect(weakCategories(compareLabels(reference, decisions, CATEGORIES, 'Outros', 1))).toEqual(['vacinas']);
  });

  it('ignores reference labels of categories removed by the user', () => {
    const reference = new Map([
      [0, 'removida'],
      [1, 'vacinas'],
    ]);
    const decisions = new Map([
      [0, decision('vacinas', 0.9)],
      [1, decision('vacinas', 0.9)],
    ]);
    expect(compareLabels(reference, decisions, CATEGORIES, 'Outros', 1).compared).toBe(1);
  });

  it('computes coverage, leftovers and counts', () => {
    const decisions = new Map([
      [0, decision('vacinas', 0.9)],
      [1, decision('farmacos', 0.5)],
      [2, decision(OTHER_CATEGORY_ID, 0.95)],
      [3, decision('vacinas', 0.2)],
    ]);
    expect(coverageOf(decisions, thresholds)).toEqual({ total: 4, auto: 2, review: 1, unclassified: 1, other: 1 });
    expect(leftoverIndices(decisions, thresholds).sort()).toEqual([2, 3]);
    const counts = categoryCountsOf(decisions, CATEGORIES, 'Outros');
    expect(counts[0]).toMatchObject({ categoryId: 'vacinas', documents: 2 });
    expect(counts.find((c) => c.categoryId === OTHER_CATEGORY_ID)?.documents).toBe(1);
  });
});

describe('applyHybridThemes', () => {
  it('writes theme, confidence and status fields', () => {
    const rows = [{ TITLE: 'a' }, { TITLE: 'b' }, { TITLE: 'c' }] as unknown as Dataset;
    const decisions = new Map([
      [0, decision('vacinas', 0.91234)],
      [1, decision('farmacos', 0.2)],
      [2, decision(OTHER_CATEGORY_ID, 0.8)],
    ]);
    const themed = applyHybridThemes(rows, decisions, CATEGORIES, { accept: 0.7, review: 0.4 }, 'pt');
    expect(themed[0]).toMatchObject({ [FIELD.THEME]: 'Vacinas', [FIELD.THEME_CONFIDENCE]: 0.912, [FIELD.THEME_STATUS]: 'auto' });
    expect(themed[1]).toMatchObject({ [FIELD.THEME]: 'Não classificado', [FIELD.THEME_STATUS]: 'nao_classificado' });
    // "Nenhuma categoria serve" nunca vira um tema "Outros", mesmo com confiança alta.
    expect(themed[2]).toMatchObject({ [FIELD.THEME]: 'Não classificado', [FIELD.THEME_STATUS]: 'nao_classificado' });
    expect(themed.some((doc) => doc[FIELD.THEME] === 'Outros')).toBe(false);
  });

  it('reads only title, keywords, abstract and year', () => {
    const rows = [
      { TITLE: 'T', KEYWORDS: 'k1; k2', ABSTRACT: 'nan', 'YEAR CLEAN': 2020.0, 'AUTHOR ADDRESS': 'x' },
    ] as unknown as Dataset;
    expect(toHybridDocs(rows)[0]).toEqual({ index: 0, title: 'T', keywords: 'k1; k2', abstract: '', year: 2020 });
  });
});

describe('hybridMethodsText', () => {
  const run: HybridRun = {
    id: 'r',
    startedAt: '2026-09-24T00:00:00Z',
    finishedAt: '2026-09-24T00:05:00Z',
    locale: 'pt',
    models: { discovery: 'deepseek-flash', classifier: 'jev-1.13.0' },
    sample: { size: 120, seed: 42, strata: 18, outliers: 12, clusterCount: 6, silhouette: 0.1 },
    thresholds: { accept: 0.7, review: 0.4 },
    taxonomyVersions: [
      { version: 1, origin: 'discovery', createdAt: '', categories: CATEGORIES },
      { version: 2, origin: 'clarify', createdAt: '', categories: CATEGORIES },
    ],
    finalTaxonomy: CATEGORIES,
    validation: [
      { taxonomyVersion: 1, compared: 118, agreement: 0.71, kappa: 0.62, perCategory: [] },
      { taxonomyVersion: 2, compared: 118, agreement: 0.84, kappa: 0.79, perCategory: [] },
    ],
    finalAgreement: null,
    expansionRounds: 0,
    coverage: { total: 1000, auto: 800, review: 150, unclassified: 50, other: 30 },
    categoryCounts: [],
    usage: { discoveryInputTokens: 0, discoveryOutputTokens: 0, classifierInputTokens: 0, classifierRequests: 0 },
    userEdited: true,
  };

  it('reports models, sample, agreement evolution and thresholds', () => {
    const text = hybridMethodsText(run, 'pt');
    expect(text).toContain('deepseek-flash');
    expect(text).toContain('jev-1.13.0');
    expect(text).toContain('semente 42');
    expect(text).toContain('revisadas em seguida pelo pesquisador');
    expect(text).toContain('71%');
    expect(text).toContain('84%');
    expect(text).toContain('0,79');
    expect(text).toContain('(800 documentos)');
  });

  it('describes a manual taxonomy without sample validation', () => {
    const manual: HybridRun = {
      ...run,
      models: { discovery: null, classifier: 'jev-1.13.0' },
      taxonomyVersions: [{ version: 1, origin: 'manual', createdAt: '', categories: CATEGORIES }],
      validation: [],
    };
    const text = hybridMethodsText(manual, 'en');
    expect(text).toContain('defined by the researcher');
    expect(text).not.toContain('agreement');
  });
});

describe('free document cap', () => {
  it('blocks only runs that would use a server key on a larger dataset', async () => {
    const { freeDocsLimitError } = await import('@/state/hybrid.store');
    const { useFreeTier } = await import('@/state/free-tier.store');
    const { DEFAULT_HYBRID_CONFIG } = await import('@/state/hybrid-config.store');
    useFreeTier.setState({ status: { freeMaxDocs: 1000 } as never });
    const free = DEFAULT_HYBRID_CONFIG;
    const ownJev = { ...free, jev: { ...free.jev, apiKey: 'jev' } };
    const ownBoth = { ...ownJev, generative: { ...free.generative, apiKey: 'ds' } };

    expect(freeDocsLimitError(free, 1000, true)).toBeNull();
    expect(freeDocsLimitError(free, 1001, true)).toContain('1.000');
    expect(freeDocsLimitError(ownJev, 1001, false)).toBeNull(); // manual: só o Jev
    expect(freeDocsLimitError(ownJev, 1001, true)).not.toBeNull(); // descoberta pelo DeepSeek do servidor
    expect(freeDocsLimitError(ownBoth, 50_000, true)).toBeNull();
  });
});
