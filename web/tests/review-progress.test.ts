import { describe, expect, it } from 'vitest';

import { nextReviewAction, reviewProgress } from '@/core/review/progress';
import type { ScreeningRecord } from '@/core/review/records';
import { createReview } from '@/core/review/state';
import type { ReviewState } from '@/core/review/types';

const record = (key: string): ScreeningRecord => ({
  key,
  index: 0,
  title: key,
  abstract: '',
  authors: '',
  year: 2020,
  venue: '',
  keywords: '',
  doi: '',
  database: 'Scopus',
});

const records = ['a', 'b', 'c', 'd'].map(record);
const now = '2026-01-01T00:00:00.000Z';

describe('reviewProgress', () => {
  it('starts empty, with quality and extraction off until configured', () => {
    const progress = reviewProgress(createReview('me'), records);
    expect(progress.steps.protocol.state).toBe('todo');
    expect(progress.steps.screening).toMatchObject({ state: 'todo', done: 0, total: 4 });
    expect(progress.steps.quality.state).toBe('off');
    expect(progress.steps.extraction.state).toBe('off');
    expect(progress.overall).toBe(0);
  });

  it('counts both screening stages and completes when nothing is pending', () => {
    const review: ReviewState = {
      ...createReview('me'),
      decisions: {
        a: { ta: 'include', ft: 'include', updatedAt: now },
        b: { ta: 'maybe', updatedAt: now },
        c: { ta: 'exclude', updatedAt: now },
      },
    };
    // 3 de 4 decididos na 1ª etapa; 1 de 2 no texto completo.
    expect(reviewProgress(review, records).steps.screening).toMatchObject({ state: 'active', done: 4, total: 6 });

    review.decisions = { ...review.decisions, b: { ta: 'maybe', ft: 'exclude', updatedAt: now }, d: { ta: 'exclude', updatedAt: now } };
    const done = reviewProgress(review, records);
    expect(done.steps.screening.state).toBe('done');
    expect(done.steps.prisma.state).toBe('done');
  });

  it('measures quality over included studies and waits without them', () => {
    const base: ReviewState = {
      ...createReview('me'),
      qualityQuestions: [{ id: 'q1', text: 'Clear aim?' }],
      qualityAnswers: [{ id: 'y', label: 'Yes', weight: 1 }],
    };
    expect(reviewProgress(base, records).steps.quality.state).toBe('waiting');

    const review: ReviewState = {
      ...base,
      decisions: {
        a: { ta: 'include', ft: 'include', updatedAt: now },
        b: { ta: 'include', ft: 'include', updatedAt: now },
      },
      quality: { a: { q1: 'y' } },
    };
    expect(reviewProgress(review, records).steps.quality).toMatchObject({ state: 'active', done: 1, total: 2, ratio: 0.5 });
  });

  it('fills the protocol checklist', () => {
    const review: ReviewState = {
      ...createReview('me', 'systematic'),
      framework: 'none',
      title: 'T',
      objective: 'O',
      questions: [{ id: '1', text: 'Q?' }],
      concepts: [{ id: 'c', label: 'C', terms: ['x'] }],
      criteria: [
        { id: 'i', kind: 'inclusion', text: 'In' },
        { id: 'e', kind: 'exclusion', text: 'Out' },
      ],
    };
    expect(reviewProgress(review, records).steps.protocol).toMatchObject({ state: 'done', done: 6, total: 6 });
  });
});

describe('nextReviewAction', () => {
  const protocol: ReviewState = {
    ...createReview('me'),
    framework: 'none',
    title: 'T',
    objective: 'O',
    questions: [{ id: '1', text: 'Q?' }],
    concepts: [{ id: 'c', label: 'C', terms: ['x'] }],
    criteria: [
      { id: 'i', kind: 'inclusion', text: 'In' },
      { id: 'e', kind: 'exclusion', text: 'Out' },
    ],
  };

  it('walks the protocol field by field', () => {
    expect(nextReviewAction(createReview('me'), records).kind).toBe('title');
    expect(nextReviewAction({ ...protocol, criteria: protocol.criteria.slice(0, 1) }, records).kind).toBe('exclusion');
  });

  it('asks to import, then to screen both stages, then for the extraction form', () => {
    expect(nextReviewAction(protocol, []).kind).toBe('import');
    expect(nextReviewAction(protocol, records)).toEqual({ kind: 'screen-ta', step: 'screening', count: 4 });

    const decisions = {
      a: { ta: 'include' as const, updatedAt: now },
      b: { ta: 'exclude' as const, updatedAt: now },
      c: { ta: 'exclude' as const, updatedAt: now },
      d: { ta: 'exclude' as const, updatedAt: now },
    };
    expect(nextReviewAction({ ...protocol, decisions }, records)).toEqual({ kind: 'screen-ft', step: 'screening', count: 1 });

    const screened = { ...protocol, decisions: { ...decisions, a: { ta: 'include' as const, ft: 'include' as const, updatedAt: now } } };
    expect(nextReviewAction(screened, records).kind).toBe('extraction-form');
    expect(
      nextReviewAction({ ...screened, extractionFields: [{ id: 'f', label: 'F', type: 'text', options: [] }] }, records),
    ).toEqual({ kind: 'extraction', step: 'extraction', count: 1 });
  });
});
