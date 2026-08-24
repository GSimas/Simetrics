import { describe, expect, it } from 'vitest';
import {
  ANALYTICAL_TOOLS,
  executeAnalyticalTool,
  toClaudeTools,
  toGeminiTools,
  toOpenAiTools,
} from '@/core/tools';
import { FIELD } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';

const mockDataset: Dataset = [
  {
    [FIELD.TITLE]: 'Deep Learning for Scientometrics',
    [FIELD.AUTHORS]: 'Silva, A. B.; Santos, C. D.',
    [FIELD.YEAR_CLEAN]: 2021,
    [FIELD.SECONDARY_TITLE]: 'SCIENTOMETRICS',
    [FIELD.TOTAL_CITATIONS]: 50,
    [FIELD.COUNTRY]: 'Brazil; United States',
    [FIELD.KEYWORDS]: 'deep learning; bibliometrics',
    [FIELD.THEME]: 'Machine Learning in Science',
    [FIELD.ABSTRACT]: 'We explore deep learning techniques applied to citation analysis.',
    [FIELD.DOI]: '10.1000/182',
    [FIELD.REFERENCES_UNIFIED]: 'Ref 1; Ref 2',
    [FIELD.DATABASE]: 'Scopus',
  } as unknown as SimetricsDoc,
  {
    [FIELD.TITLE]: 'Graph Neural Networks in Citation Networks',
    [FIELD.AUTHORS]: 'Silva, A. B.; Johnson, M.',
    [FIELD.YEAR_CLEAN]: 2022,
    [FIELD.SECONDARY_TITLE]: 'JOURNAL OF INFORMETRICS',
    [FIELD.TOTAL_CITATIONS]: 30,
    [FIELD.COUNTRY]: 'Brazil; United Kingdom',
    [FIELD.KEYWORDS]: 'graph networks; citation analysis',
    [FIELD.THEME]: 'Machine Learning in Science',
    [FIELD.ABSTRACT]: 'Graph neural networks provide advanced representation of scholarly papers.',
    [FIELD.DOI]: '10.1000/183',
    [FIELD.REFERENCES_UNIFIED]: 'Ref 3; Ref 4',
    [FIELD.DATABASE]: 'Scopus',
  } as unknown as SimetricsDoc,
  {
    [FIELD.TITLE]: 'Overview of Topic Modeling',
    [FIELD.AUTHORS]: 'Johnson, M.; Muller, K.',
    [FIELD.YEAR_CLEAN]: 2019,
    [FIELD.SECONDARY_TITLE]: 'SCIENTOMETRICS',
    [FIELD.TOTAL_CITATIONS]: 10,
    [FIELD.COUNTRY]: 'United Kingdom; Germany',
    [FIELD.KEYWORDS]: 'topic modeling; lda',
    [FIELD.THEME]: 'Topic Modeling',
    [FIELD.ABSTRACT]: 'A comprehensive survey on probabilistic topic models.',
    [FIELD.DOI]: '10.1000/184',
    [FIELD.REFERENCES_UNIFIED]: 'Ref 5',
    [FIELD.DATABASE]: 'Scopus',
  } as unknown as SimetricsDoc,
];

describe('Analytical Tools Engine', () => {
  it('converts tools to Gemini format correctly', () => {
    const geminiTools = toGeminiTools(ANALYTICAL_TOOLS);
    expect(geminiTools).toHaveLength(1);
    const decls = geminiTools[0]?.function_declarations;
    expect(decls).toBeDefined();
    expect(decls).toHaveLength(4);
    expect(decls?.[0]?.name).toBe('query_analytical_table');
    expect(decls?.[0]?.parameters?.type).toBe('OBJECT');
  });

  it('converts tools to OpenAI format correctly', () => {
    const openAiTools = toOpenAiTools(ANALYTICAL_TOOLS);
    expect(openAiTools).toHaveLength(4);
    expect(openAiTools[0]?.type).toBe('function');
    expect(openAiTools[0]?.function.name).toBe('query_analytical_table');
    expect(openAiTools[0]?.function.parameters.type).toBe('object');
  });

  it('converts tools to Claude format correctly', () => {
    const claudeTools = toClaudeTools(ANALYTICAL_TOOLS);
    expect(claudeTools).toHaveLength(4);
    expect(claudeTools[0]?.name).toBe('query_analytical_table');
    expect(claudeTools[0]?.input_schema.type).toBe('object');
  });

  describe('query_analytical_table', () => {
    it('queries authors table with sorting by citations', () => {
      const res = executeAnalyticalTool(
        'query_analytical_table',
        { table: 'authors', sortBy: 'citations', limit: 5 },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.table).toBe('authors');
      expect(Number(data.returnedCount)).toBeGreaterThan(0);

      const rows = data.rows as Array<{ entity: string; citations: number; docCount: number; h_index: number }>;
      expect(rows[0]?.entity).toBe('Silva, A. B.');
      expect(rows[0]?.citations).toBe(80); // 50 + 30
      expect(rows[0]?.docCount).toBe(2);
      expect(rows[0]?.h_index).toBe(2);
    });

    it('filters authors by text query', () => {
      const res = executeAnalyticalTool(
        'query_analytical_table',
        { table: 'authors', filterText: 'Muller' },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      const rows = data.rows as Array<{ entity: string }>;
      // Muller, K. (como autor) e Johnson, M. (como coautor de Muller)
      expect(rows.length).toBeGreaterThanOrEqual(1);
      expect(rows.some((r) => r.entity === 'Muller, K.')).toBe(true);
    });

    it('queries countries table accurately', () => {
      const res = executeAnalyticalTool(
        'query_analytical_table',
        { table: 'countries', sortBy: 'citations' },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      const rows = data.rows as Array<{ entity: string; citations: number }>;
      expect(rows[0]?.entity).toBe('Brazil');
      expect(rows[0]?.citations).toBe(80);
    });
  });

  describe('filter_and_aggregate_documents', () => {
    it('filters documents by year and country', () => {
      const res = executeAnalyticalTool(
        'filter_and_aggregate_documents',
        { yearMin: 2021, country: 'Brazil' },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.totalMatchingDocuments).toBe(2);
      expect(data.totalCitations).toBe(80);
      expect(data.meanCitations).toBe(40);
    });

    it('filters by minCitations', () => {
      const res = executeAnalyticalTool(
        'filter_and_aggregate_documents',
        { minCitations: 40 },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.totalMatchingDocuments).toBe(1);
      const sampleDocs = data.sampleDocuments as Array<{ title: string; citations: number }>;
      expect(sampleDocs[0]?.title).toBe('Deep Learning for Scientometrics');
      expect(sampleDocs[0]?.citations).toBe(50);
    });
  });

  describe('get_dataset_general_metrics', () => {
    it('calculates macro bibliometric metrics', () => {
      const res = executeAnalyticalTool(
        'get_dataset_general_metrics',
        {},
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.totalDocuments).toBe(3);
      expect(data.timespan).toBe('2019:2022');
      expect(data.yearlyProduction).toBeDefined();
    });
  });

  describe('get_entity_profile', () => {
    it('retrieves detailed author profile with publications and scientometrics', () => {
      const res = executeAnalyticalTool(
        'get_entity_profile',
        { entityType: 'author', name: 'Silva' },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.found).toBe(true);
      expect(data.entity).toBe('Silva, A. B.');
      expect(data.totalDocuments).toBe(2);
      expect(data.totalCitations).toBe(80);

      const pubs = data.publications as Array<{ title: string; citations: number }>;
      expect(pubs).toHaveLength(2);
      expect(pubs[0]?.citations).toBe(50);
    });

    it('retrieves theme profile with papers and citation stats', () => {
      const res = executeAnalyticalTool(
        'get_entity_profile',
        { entityType: 'theme', name: 'Machine Learning' },
        mockDataset,
      );

      expect(res.success).toBe(true);
      const data = res.result as Record<string, unknown>;
      expect(data.entityType).toBe('theme');
      expect(data.totalDocuments).toBe(2);
      expect(data.totalCitations).toBe(80);
      expect(data.meanCitations).toBe(40);
    });
  });
});
