import { describe, expect, it } from 'vitest';

import { computeReviewFlow, isScreeningComplete, NO_REASON_ID, QUALITY_REASON_ID } from '@/core/review/flow';
import { PRISMALAB_COUNT_KEYS, toPrismaLabProject } from '@/core/review/prismalab';
import {
  defaultQualityAnswers,
  finalSelection,
  includedStudies,
  isExcludedByQuality,
  maxQualityScore,
  scoreStudy,
} from '@/core/review/quality';
import { normalizeDoi, recordKey, toScreeningRecords } from '@/core/review/records';
import { buildHighlighter, buildSearchString } from '@/core/review/search-string';
import { createReview, decisionRows, hasReviewContent, normalizeReview } from '@/core/review/state';
import { extractionRows, qualityRows, summarizeField } from '@/core/review/synthesis';
import type { Criterion, RecordScreening, ReviewState, SearchConcept } from '@/core/review/types';
import { parseProjectEnvelope } from '@/lib/project';
import type { Dataset } from '@/lib/types';

const doc = (fields: Record<string, unknown>) => fields as Dataset[number];

describe('record keys', () => {
  it('prefers the normalized DOI', () => {
    expect(normalizeDoi('https://doi.org/10.1000/ABC')).toBe('10.1000/abc');
    expect(normalizeDoi('doi: 10.1000/x')).toBe('10.1000/x');
    expect(recordKey('https://dx.doi.org/10.1000/ABC', 'Anything', 2020)).toBe('doi:10.1000/abc');
  });

  it('falls back to title + year, ignoring case, accents and punctuation', () => {
    const a = recordKey('', 'Saúde Digital: uma revisão', 2021);
    const b = recordKey('nan', 'saude digital uma revisao', 2021);
    expect(a).toBe(b);
    expect(a).toMatch(/^ti:/);
    expect(recordKey('', 'Saúde Digital: uma revisão', 2022)).not.toBe(a);
  });

  it('is independent of the record position and suffixes repeated records', () => {
    const rows: Dataset = [
      doc({ TITLE: 'First', DOI: '10.1/a', 'YEAR CLEAN': 2020 }),
      doc({ TITLE: 'Second', DOI: '', 'YEAR CLEAN': 2021 }),
      doc({ TITLE: 'First again', DOI: '10.1/A', 'YEAR CLEAN': 2020 }),
    ];
    const keys = toScreeningRecords(rows).map((record) => record.key);
    const reversed = toScreeningRecords([...rows].reverse()).map((record) => record.key);

    expect(keys[0]).toBe('doi:10.1/a');
    expect(keys[2]).toBe('doi:10.1/a#2');
    expect(reversed[1]).toBe(keys[1]);
  });
});

describe('search string builder', () => {
  const concepts: SearchConcept[] = [
    { id: '1', label: 'Population', terms: ['older adults', 'elderly', 'aged'] },
    { id: '2', label: 'Intervention', terms: ['telehealth', 'telemedicin*'] },
    { id: '3', label: 'Empty', terms: ['  '] },
  ];

  it('joins synonyms with OR and concepts with AND', () => {
    expect(buildSearchString(concepts, 'generic')).toBe(
      '("older adults" OR elderly OR aged) AND (telehealth OR telemedicin*)',
    );
  });

  it('applies the field syntax of each database', () => {
    expect(buildSearchString(concepts, 'scopus')).toBe(
      'TITLE-ABS-KEY(("older adults" OR elderly OR aged) AND (telehealth OR telemedicin*))',
    );
    expect(buildSearchString(concepts, 'wos')).toBe(
      'TS=(("older adults" OR elderly OR aged) AND (telehealth OR telemedicin*))',
    );
    expect(buildSearchString(concepts, 'pubmed')).toBe(
      '("older adults"[tiab] OR elderly[tiab] OR aged[tiab]) AND (telehealth[tiab] OR telemedicin*[tiab])',
    );
    expect(buildSearchString(concepts, 'cochrane')).toBe(
      '("older adults" OR elderly OR aged):ti,ab,kw AND (telehealth OR telemedicin*):ti,ab,kw',
    );
  });

  it('does not double the parentheses of a single concept', () => {
    const single = [concepts[0]!];
    expect(buildSearchString(single, 'scopus')).toBe('TITLE-ABS-KEY("older adults" OR elderly OR aged)');
    expect(buildSearchString(single, 'wos')).toBe('TS=("older adults" OR elderly OR aged)');
    expect(buildSearchString(single, 'generic')).toBe('("older adults" OR elderly OR aged)');
  });

  it('returns an empty string without terms', () => {
    expect(buildSearchString([], 'scopus')).toBe('');
  });

  it('highlights terms, expanding the wildcard', () => {
    const regex = buildHighlighter(concepts)!;
    const text = 'Telemedicine for Older  adults and the aged; not agedness.';
    expect(text.match(regex)).toEqual(['Telemedicine', 'Older  adults', 'aged']);
  });
});

describe('review flow', () => {
  const original: Dataset = [
    doc({ TITLE: 'A', DOI: '10.1/a', 'BASE DE DADOS': 'Scopus' }),
    doc({ TITLE: 'A', DOI: '10.1/a', 'BASE DE DADOS': 'Web of Science' }),
    doc({ TITLE: 'B', DOI: '10.1/b', 'BASE DE DADOS': 'Scopus' }),
    doc({ TITLE: 'C', DOI: '10.1/c', 'BASE DE DADOS': 'Scopus' }),
    doc({ TITLE: 'D', DOI: '10.1/d', 'BASE DE DADOS': 'Web of Science' }),
    doc({ TITLE: 'E', DOI: '10.1/e', 'BASE DE DADOS': 'Web of Science' }),
  ];
  const active = [original[0]!, original[2]!, original[3]!, original[4]!, original[5]!];
  const records = toScreeningRecords(active);
  const criteria: Criterion[] = [{ id: 'x1', kind: 'exclusion', text: 'Wrong population' }];
  const at = '2026-01-01T00:00:00.000Z';
  const decisions: Record<string, RecordScreening> = {
    'doi:10.1/a': { ta: 'include', ft: 'include', updatedAt: at },
    'doi:10.1/b': { ta: 'maybe', ft: 'exclude', ftReason: 'x1', updatedAt: at },
    'doi:10.1/c': { ta: 'exclude', ft: 'include', updatedAt: at }, // ft ignorado: saiu antes
    'doi:10.1/d': { ta: 'include', ft: 'not-retrieved', updatedAt: at },
    'doi:gone': { ta: 'include', updatedAt: at }, // registro que não está mais na base
  };

  const flow = computeReviewFlow(original, records, decisions, criteria, 'No reason');

  it('counts each PRISMA stage', () => {
    expect(flow.identified).toBe(6);
    expect(flow.identifiedBySource).toEqual([
      { name: 'Scopus', count: 3 },
      { name: 'Web of Science', count: 3 },
    ]);
    expect(flow.duplicatesRemoved).toBe(1);
    expect(flow.screened).toBe(5);
    expect(flow.titleAbstract).toEqual({ pending: 1, include: 2, maybe: 1, exclude: 1 });
    expect(flow.fullText).toEqual({ eligible: 3, pending: 0, include: 1, exclude: 1, notRetrieved: 1 });
    expect(flow.fullTextExclusions).toEqual([{ id: 'x1', label: 'Wrong population', count: 1 }]);
    expect(flow.included).toBe(1);
    expect(isScreeningComplete(flow)).toBe(false);
  });

  it('groups full-text exclusions without a reason', () => {
    const withoutReason = computeReviewFlow(
      original,
      records,
      { 'doi:10.1/a': { ta: 'include', ft: 'exclude', updatedAt: at } },
      criteria,
      'No reason',
    );
    expect(withoutReason.fullTextExclusions).toEqual([{ id: NO_REASON_ID, label: 'No reason', count: 1 }]);
  });

  it('exports a complete PRISMALab v2 project', () => {
    let n = 0;
    const project = toPrismaLabProject({
      title: 'Telehealth review',
      reviewType: 'scoping',
      flow,
      locale: 'en',
      now: new Date(at),
      makeId: () => `id-${(n += 1)}`,
    });

    expect(project.schemaVersion).toBe(2);
    expect(Object.keys(project.counts).sort()).toEqual([...PRISMALAB_COUNT_KEYS].sort());
    expect(project.counts).toMatchObject({
      databases: 6,
      registers: 0,
      duplicates: 1,
      screened: 5,
      recordsExcluded: 1,
      reportsSought: 3,
      reportsNotRetrieved: 1,
      reportsAssessed: 2,
      reportsExcluded: 1,
      newStudies: 1,
      websites: null,
      previousStudies: null,
    });
    expect(project.reviewType).toBe('scoping');
    expect(project.extensions).toEqual(['PRISMA-ScR']);
    expect(project.sources.map((s) => [s.name, s.count])).toEqual([
      ['Scopus', 3],
      ['Web of Science', 3],
    ]);
    expect(project.exclusionReasons).toEqual([{ id: expect.any(String), label: 'Wrong population', count: 1 }]);
    expect(project.checklist).toHaveLength(27);
    expect(project.locale).toBe('en');
  });

  it('builds one decision row per record', () => {
    const review = { ...createReview('me'), criteria, decisions };
    const rows = decisionRows(records, review, { decision: (value) => value ?? '' });
    expect(rows).toHaveLength(5);
    expect(rows[1]).toMatchObject({ key: 'doi:10.1/b', title_abstract: 'maybe', full_text: 'exclude', full_text_reason: 'Wrong population' });
  });
});

describe('review persistence', () => {
  it('treats a blank protocol as nothing to save', () => {
    const review = createReview('me');
    expect(hasReviewContent(review)).toBe(false);
    expect(hasReviewContent({ ...review, title: 'x' })).toBe(true);
    expect(hasReviewContent(null)).toBe(false);
  });

  it('normalizes hand-edited or partial data', () => {
    const review = normalizeReview({
      type: 'scoping',
      title: 'T',
      framework: 'bogus',
      criteria: [{ id: 'c', kind: 'inclusion', text: 'ok' }, { id: 'd', kind: 'bogus' }, 'junk'],
      decisions: { k: { ta: 'include', ft: 'nope', note: 'n' }, bad: 3 },
    })!;
    expect(review.type).toBe('scoping');
    expect(review.framework).toBe('PCC');
    expect(review.criteria).toEqual([{ id: 'c', kind: 'inclusion', text: 'ok' }]);
    expect(review.decisions).toEqual({ k: { ta: 'include', note: 'n', updatedAt: expect.any(String) } });
    expect(normalizeReview(undefined)).toBeNull();
  });

  it('keeps the review through a project export/import', () => {
    const review = { ...createReview('me'), title: 'My review' };
    const parsed = parseProjectEnvelope({
      kind: 'simetrics-project',
      schemaVersion: 1,
      exportedAt: '',
      project: { id: 'p', name: 'P', original: [], active: [], review },
    });
    expect(parsed.review?.title).toBe('My review');
    expect(parsed.active).toEqual([]);
  });
});

describe('quality assessment and extraction', () => {
  const at = '2026-01-01T00:00:00.000Z';
  const records = toScreeningRecords([
    doc({ TITLE: 'A', DOI: '10.1/a', 'YEAR CLEAN': 2020 }),
    doc({ TITLE: 'B', DOI: '10.1/b', 'YEAR CLEAN': 2021 }),
    doc({ TITLE: 'C', DOI: '10.1/c', 'YEAR CLEAN': 2021 }),
    doc({ TITLE: 'D', DOI: '10.1/d', 'YEAR CLEAN': 2022 }),
  ]);
  let n = 0;
  const answers = defaultQualityAnswers('pt', () => `ans-${(n += 1)}`);
  const [yes, partial, no] = answers.map((answer) => answer.id) as [string, string, string];
  const base = createReview('me');
  const review = {
    ...base,
    decisions: {
      'doi:10.1/a': { ta: 'include', ft: 'include', updatedAt: at },
      'doi:10.1/b': { ta: 'include', ft: 'include', updatedAt: at },
      'doi:10.1/c': { ta: 'maybe', ft: 'include', updatedAt: at },
      'doi:10.1/d': { ta: 'include', ft: 'exclude', updatedAt: at },
    },
    qualityQuestions: [
      { id: 'q1', text: 'Aims clear?' },
      { id: 'q2', text: 'Method adequate?' },
    ],
    qualityAnswers: answers,
    qualityCutoff: 1.5,
    excludeBelowCutoff: true,
    quality: {
      'doi:10.1/a': { q1: yes, q2: partial }, // 1.5 — passa
      'doi:10.1/b': { q1: partial, q2: no }, // 0.5 — abaixo
      'doi:10.1/c': { q1: yes }, // incompleta — nunca exclui
    },
    extractionFields: [
      { id: 'f1', label: 'Country', type: 'select' as const, options: ['Brazil', 'Chile'] },
      { id: 'f2', label: 'Sample', type: 'number' as const, options: [] },
      { id: 'f3', label: 'RCT', type: 'boolean' as const, options: [] },
    ],
    extraction: {
      'doi:10.1/a': { values: { f1: 'Brazil', f2: 10, f3: true }, done: true },
      'doi:10.1/c': { values: { f1: 'Brazil', f2: 30 }, done: false },
    },
  } satisfies ReviewState;

  it('scores studies with the sum of answer weights', () => {
    expect(maxQualityScore(review)).toBe(2);
    expect(scoreStudy(review, 'doi:10.1/a')).toMatchObject({ score: 1.5, complete: true, passes: true });
    expect(scoreStudy(review, 'doi:10.1/b')).toMatchObject({ score: 0.5, complete: true, passes: false });
    expect(scoreStudy(review, 'doi:10.1/c')).toMatchObject({ score: 1, answered: 1, complete: false, passes: null });
  });

  it('drops only complete assessments below the cutoff from the final selection', () => {
    expect(includedStudies(records, review).map((r) => r.key)).toEqual(['doi:10.1/a', 'doi:10.1/b', 'doi:10.1/c']);
    expect(finalSelection(records, review).map((r) => r.key)).toEqual(['doi:10.1/a', 'doi:10.1/c']);
    expect(isExcludedByQuality({ ...review, excludeBelowCutoff: false }, 'doi:10.1/b')).toBe(false);
  });

  it('counts quality exclusions as full-text exclusions in the PRISMA flow', () => {
    const flow = computeReviewFlow([], records, review.decisions, review.criteria, 'No reason', {
      isExcluded: (key) => isExcludedByQuality(review, key),
      label: 'Low quality',
    });
    expect(flow.fullText).toMatchObject({ eligible: 4, include: 2, exclude: 2 });
    expect(flow.included).toBe(2);
    expect(flow.fullTextExclusions).toEqual(
      expect.arrayContaining([{ id: QUALITY_REASON_ID, label: 'Low quality', count: 1 }]),
    );
  });

  it('summarizes extraction fields over the final selection', () => {
    const studies = finalSelection(records, review);
    const labels = { yes: 'Sim', no: 'Não' };
    expect(summarizeField(review.extractionFields[0]!, studies, review, labels)).toEqual({
      kind: 'counts',
      filled: 2,
      counts: [
        { label: 'Brazil', count: 2 },
        { label: 'Chile', count: 0 },
      ],
    });
    expect(summarizeField(review.extractionFields[1]!, studies, review, labels)).toEqual({
      kind: 'numeric',
      filled: 2,
      mean: 20,
      min: 10,
      max: 30,
    });
    expect(summarizeField(review.extractionFields[2]!, studies, review, labels)).toMatchObject({
      counts: [
        { label: 'Sim', count: 1 },
        { label: 'Não', count: 0 },
      ],
    });
    expect(extractionRows(studies, review, labels)[0]).toMatchObject({ Country: 'Brazil', Sample: '10', RCT: 'Sim' });
    expect(qualityRows(studies, review)[0]).toMatchObject({ Q1: 'Sim', Q2: 'Parcialmente', score: 1.5, max: 2 });
  });

  it('normalizes phase 2 fields from saved data', () => {
    const normalized = normalizeReview({
      ...review,
      qualityAnswers: [...review.qualityAnswers, { id: 'bad', label: 'x', weight: 'heavy' }],
      qualityCutoff: 'high',
      extractionFields: [...review.extractionFields, { id: 'f9', label: 'x', type: 'bogus' }],
      extraction: { k: { values: { f1: 'ok', f2: { nested: true } }, done: 'yes' } },
    })!;
    expect(normalized.qualityAnswers).toHaveLength(3);
    expect(normalized.qualityCutoff).toBeNull();
    expect(normalized.extractionFields).toHaveLength(3);
    expect(normalized.extraction).toEqual({ k: { values: { f1: 'ok' }, done: false } });
    expect(normalized.quality['doi:10.1/a']).toEqual({ q1: yes, q2: partial });
  });
});
