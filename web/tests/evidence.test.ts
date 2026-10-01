import { describe, expect, it } from 'vitest';

import { aiQuestions, buildEvidencePrompt, parseEvidenceResponse, prepareArticle } from '@/core/review/ai-evidence';
import {
  canAccept,
  coerceToField,
  coerceToQualityAnswer,
  detectTextLayer,
  locateInPages,
  locateQuote,
  matchPdfToStudy,
  pilotMetrics,
  relocate,
  statusAfterAnswer,
} from '@/core/review/evidence';
import { createReview, normalizeReview } from '@/core/review/state';
import type { ExtractionField, TargetEvidence } from '@/core/review/types';

const PAGE =
  'Methods\nWe recruited 128 older adults with chro-\nnic heart failure from three “primary care” clinics in Porto Alegre. ' +
  'Participants were randomized to tele­monitoring or usual care for 12 months.';

describe('locateQuote', () => {
  it('matches across line breaks, hyphenation and typographic quotes', () => {
    const match = locateQuote(PAGE, 'We recruited 128 older adults with chronic heart failure from three "primary care" clinics');
    expect(match?.location).toBe('exact');
    expect(PAGE.slice(match!.start, match!.end)).toMatch(/^We recruited 128.*clinics$/s);
  });

  it('ignores accents and case', () => {
    expect(locateQuote('A amostra incluiu PACIENTES idosos.', 'pacientes idosos')?.location).toBe('exact');
    expect(locateQuote('Saúde digital', 'saude DIGITAL')?.location).toBe('exact');
  });

  it('accepts a quote whose middle differs slightly, but not a short invented one', () => {
    const paraphrased = 'We recruited 128 older adults with chronic cardiac failure from three primary care clinics in Porto Alegre';
    expect(locateQuote(PAGE, paraphrased)?.location).toBe('approximate');
    // A palavra trocada pode estar logo no começo.
    const conjugated = locateQuote(PAGE, 'They recruited 128 older adults with chronic heart failure from three primary care clinics');
    expect(conjugated?.location).toBe('approximate');
    expect(PAGE.slice(conjugated!.start, conjugated!.end)).toMatch(/recruited 128.*clinics/s);
    expect(locateQuote(PAGE, 'recruited 200 adults')).toBeNull();
    expect(locateQuote(PAGE, 'The trial was double blinded and placebo controlled across sites')).toBeNull();
  });

  it('prefers the hinted page and falls back to the others', () => {
    const pages = ['Introduction text only.', PAGE];
    expect(locateInPages(pages, 'usual care for 12 months', 1)?.page).toBe(2);
    expect(locateInPages(pages, 'Introduction text', 2)?.page).toBe(1);
    expect(locateInPages(pages, 'nowhere to be seen in this article')).toBeNull();
  });

  it('uses the surrounding text to pick the right repeated quote', () => {
    const text = 'Group A: 12 months of follow-up. Group B: 12 months of follow-up.';
    const match = relocate(text, '12 months', 'Group B: ', ' of follow-up.');
    expect(match?.start).toBe(text.lastIndexOf('12 months'));
  });
});

describe('PDF text layer', () => {
  it('flags scanned and partially scanned PDFs', () => {
    const full = 'x'.repeat(200);
    expect(detectTextLayer([full, full])).toBe('ok');
    expect(detectTextLayer([full, '  '])).toBe('partial');
    expect(detectTextLayer(['', ' 1 '])).toBe('none');
    expect(detectTextLayer([])).toBe('none');
  });
});

describe('AI answers → form values', () => {
  const field = (type: ExtractionField['type'], options: string[] = []): ExtractionField => ({ id: 'f', label: 'F', type, options });

  it('coerces each field type and drops what does not fit', () => {
    expect(coerceToField(field('number'), '1,5')).toBe(1.5);
    expect(coerceToField(field('number'), 'about forty')).toBeNull();
    expect(coerceToField(field('boolean'), 'Sim')).toBe(true);
    expect(coerceToField(field('boolean'), 'no')).toBe(false);
    expect(coerceToField(field('date'), '2021')).toBe('2021-01-01');
    expect(coerceToField(field('select', ['Ensaio clínico', 'Coorte']), 'ensaio clinico')).toBe('Ensaio clínico');
    expect(coerceToField(field('select', ['Coorte']), 'Caso-controle')).toBeNull();
    expect(coerceToField(field('multiselect', ['A', 'B', 'C']), ['b', 'x', 'A'])).toEqual(['B', 'A']);
    expect(coerceToField(field('text'), '  128 participantes ')).toBe('128 participantes');
    expect(coerceToField(field('text'), null)).toBeNull();
  });

  it('maps a quality answer label to its id', () => {
    const answers = [
      { id: 'y', label: 'Sim', weight: 1 },
      { id: 'n', label: 'Não', weight: 0 },
    ];
    expect(coerceToQualityAnswer(answers, 'nao')).toBe('n');
    expect(coerceToQualityAnswer(answers, 'Talvez')).toBeNull();
  });
});

describe('human verification', () => {
  const suggested: TargetEvidence = {
    evidence: [],
    suggestion: { value: 128, rationale: '', model: 'm', createdAt: '' },
    status: 'suggested',
  };

  it('confirms when the reviewer keeps the AI answer and marks edits otherwise', () => {
    expect(statusAfterAnswer(suggested, 128)).toBe('confirmed');
    expect(statusAfterAnswer(suggested, 130)).toBe('edited');
    expect(statusAfterAnswer(suggested, null)).toBe('suggested');
    expect(statusAfterAnswer({ ...suggested, status: 'confirmed' }, null)).toBe('rejected');
    expect(statusAfterAnswer(undefined, 5)).toBe('manual');
  });

  it('only lets a suggestion be accepted as is when a quote was found in the PDF', () => {
    const item = { id: 'e', page: 1, quote: 'q', prefix: '', suffix: '', rects: [], origin: 'ai' as const, reviewerId: '', createdAt: '' };
    expect(canAccept({ ...suggested, evidence: [{ ...item, location: 'not-found' }] })).toBe(false);
    expect(canAccept({ ...suggested, evidence: [{ ...item, location: 'approximate' }] })).toBe(true);
    expect(canAccept({ evidence: [{ ...item, location: 'exact' }], status: 'manual' })).toBe(false);
  });
});

describe('AI prompt and response', () => {
  const review = {
    ...createReview('me'),
    extractionFields: [
      { id: 'n', label: 'Tamanho da amostra', type: 'number' as const, options: [] },
      { id: 'blank', label: ' ', type: 'text' as const, options: [] },
    ],
    qualityQuestions: [{ id: 'q', text: 'Há grupo controle?' }],
    qualityAnswers: [
      { id: 'y', label: 'Sim', weight: 1 },
      { id: 'n', label: 'Não', weight: 0 },
    ],
  };

  it('asks only the labelled questions, with options for quality', () => {
    expect(aiQuestions(review, 'extraction').map((q) => q.target)).toEqual(['extraction:n']);
    expect(aiQuestions(review, 'quality')).toEqual([
      { target: 'quality:q', label: 'Há grupo controle?', type: 'quality', options: ['Sim', 'Não'] },
    ]);
  });

  it('marks pages and drops the reference list', () => {
    const article = prepareArticle(['Intro', 'Methods', 'Results\nReferences\n1. Smith', 'More refs'], 'Página');
    expect(article.droppedReferences).toBe(true);
    expect(article.text).toContain('=== Página 3 ===\nResults');
    expect(article.text).not.toContain('Smith');
    expect(article.text).not.toContain('More refs');
    const prompt = buildEvidencePrompt(aiQuestions(review, 'extraction'), article, { title: 'T', locale: 'pt' });
    expect(prompt.user).toContain('"id": "q1"');
    expect(prompt.system).toMatch(/LITERALMENTE/);
  });

  it('reads fenced JSON and skips unanswered or unknown questions', () => {
    const questions = aiQuestions(review, 'extraction');
    const raw =
      '```json\n{"answers":[{"id":"q1","answer":128,"quotes":[{"text":"We recruited 128","page":"2"},{"text":""}],"rationale":"Methods"},' +
      '{"id":"q2","answer":"x"},{"id":"q1","answer":null}]}\n```';
    expect(parseEvidenceResponse(raw, questions)).toEqual([
      { target: 'extraction:n', answer: 128, quotes: [{ text: 'We recruited 128', page: 2 }], rationale: 'Methods' },
    ]);
    expect(parseEvidenceResponse('not json', questions)).toEqual([]);
  });
});

describe('saved evidence', () => {
  it('survives normalization and feeds the pilot numbers', () => {
    const raw = {
      ...createReview('me'),
      documents: { a: { hash: 'h', name: 'a.pdf', size: 10, pages: 3, textLayer: 'none', addedAt: '' }, bad: { name: 'x' } },
      evidence: {
        a: {
          'extraction:n': {
            status: 'edited',
            suggestion: { value: 128, rationale: '', model: 'm', createdAt: '' },
            evidence: [
              { id: '1', page: 2, quote: 'q', origin: 'ai', location: 'not-found' },
              { id: '2', page: 0, quote: 'invalid page' },
            ],
          },
          'quality:q': { status: 'suggested', evidence: [] },
          'nonsense:x': { status: 'manual', evidence: [] },
        },
      },
    };
    const review = normalizeReview(JSON.parse(JSON.stringify(raw)))!;
    expect(Object.keys(review.documents)).toEqual(['a']);
    expect(review.evidence.a?.['extraction:n']?.evidence).toHaveLength(1);
    // Proposta sem valor não pode ficar "em aberto".
    expect(review.evidence.a?.['quality:q']?.status).toBe('manual');
    expect(Object.keys(review.evidence.a ?? {})).toEqual(['extraction:n', 'quality:q']);

    const metrics = pilotMetrics(review, ['a', 'b']);
    expect(metrics).toMatchObject({ studies: 2, withPdf: 1, scanned: 1, suggested: 1, edited: 1, manual: 1, aiQuotes: 1, notFound: 1 });
  });

  it('upgrades a version 1 review without losing anything', () => {
    const v1 = { ...createReview('me'), schemaVersion: 1 } as Record<string, unknown>;
    delete v1.documents;
    delete v1.evidence;
    const review = normalizeReview(v1)!;
    expect(review.schemaVersion).toBe(2);
    expect(review.documents).toEqual({});
    expect(review.evidence).toEqual({});
  });
});

describe('bulk upload matching', () => {
  const studies = [
    { key: 'a', doi: 'https://doi.org/10.1000/ABC', title: 'Telemonitoring for older adults with heart failure' },
    { key: 'b', doi: '', title: 'A long enough title about digital health in primary care' },
    { key: 'c', doi: '', title: 'Editorial' },
  ];

  it('matches by DOI first, then by a long title on the first page', () => {
    expect(matchPdfToStudy(studies, { doi: '10.1000/abc.', firstPageText: '' })?.key).toBe('a');
    expect(matchPdfToStudy(studies, { doi: null, firstPageText: 'Journal X\nA long enough title about\ndigital health in primary care\nAuthors' })?.key).toBe('b');
    expect(matchPdfToStudy(studies, { doi: null, firstPageText: 'Editorial board' })).toBeNull();
  });
});
