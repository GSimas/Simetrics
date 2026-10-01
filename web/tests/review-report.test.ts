import { Packer } from 'docx';
import JSZip from 'jszip';
import { describe, expect, it } from 'vitest';

import { computeReviewFlow } from '@/core/review/flow';
import { toScreeningRecords } from '@/core/review/records';
import { createReview } from '@/core/review/state';
import type { ReviewState } from '@/core/review/types';
import type { Dataset } from '@/lib/types';
import { REVIEW_COPY } from '@/features/review/copy';
import { buildReportModel, fullTextParagraphs, quotesCell } from '@/features/review/report-model';
import { buildReviewDocx } from '@/features/review/report-docx';
import { buildReviewPdf, pdfSafe } from '@/features/review/report-pdf';
import { REPORT_TEXT } from '@/features/review/report-shared';

const doc = (fields: Record<string, unknown>) => fields as Dataset[number];

const records = toScreeningRecords([
  doc({ TITLE: 'Telemonitoring trial', AUTHORS: 'Silva, A.; Souza, B.', DOI: '10.1/a', 'YEAR CLEAN': 2020 }),
  doc({ TITLE: 'Excluded cohort', AUTHORS: 'Lima, C.', DOI: '10.1/b', 'YEAR CLEAN': 2021 }),
  doc({ TITLE: 'Screened out', AUTHORS: 'Rosa, D.', DOI: '10.1/c', 'YEAR CLEAN': 2022 }),
]);
const [a, b, c] = records.map((record) => record.key) as [string, string, string];

function sampleReview(): ReviewState {
  const base = createReview('me');
  const now = '2026-10-01T12:00:00.000Z';
  return {
    ...base,
    title: 'Telessaúde',
    criteria: [
      { id: 'out', kind: 'exclusion', text: 'Fora do tema' },
      { id: 'design', kind: 'exclusion', text: 'Desenho inadequado' },
    ],
    decisions: {
      [a]: { ta: 'include', ft: 'include', updatedAt: now },
      [b]: { ta: 'include', ft: 'exclude', ftReason: 'design', note: 'Coorte sem grupo controle', updatedAt: now },
      [c]: { ta: 'exclude', taReason: 'out', updatedAt: now },
    },
    extractionFields: [
      { id: 'n', label: 'Tamanho da amostra', type: 'number', options: [] },
      { id: 'design', label: 'Desenho', type: 'select', options: ['Ensaio', 'Coorte'] },
    ],
    qualityQuestions: [{ id: 'q1', text: 'Há grupo controle?' }],
    qualityAnswers: [
      { id: 'y', label: 'Sim', weight: 1 },
      { id: 'no', label: 'Não', weight: 0 },
    ],
    quality: { [a]: { q1: 'y' } },
    extraction: { [a]: { values: { n: 130, design: 'Ensaio' }, done: true } },
    documents: {
      [a]: { hash: 'h1', name: 'silva.pdf', size: 1000, pages: 12, textLayer: 'ok', addedAt: now },
      [b]: { hash: 'h2', name: 'lima.pdf', size: 1000, pages: 8, textLayer: 'none', addedAt: now },
    },
    evidence: {
      [a]: {
        'extraction:n': {
          status: 'edited',
          suggestion: { value: 128, rationale: 'Métodos', model: 'deepseek-v4', createdAt: now },
          evidence: [
            { id: 'e1', page: 3, quote: 'We recruited 130 adults', prefix: '', suffix: '', rects: [], origin: 'ai', location: 'exact', reviewerId: 'me', createdAt: now },
          ],
        },
        'quality:q1': {
          status: 'confirmed',
          suggestion: { value: 'y', rationale: '', model: 'deepseek-v4', createdAt: now },
          evidence: [
            { id: 'e2', page: 4, quote: 'usual care group', prefix: '', suffix: '', rects: [], origin: 'ai', location: 'approximate', reviewerId: 'me', createdAt: now },
            { id: 'e3', page: 9, quote: 'invented sentence', prefix: '', suffix: '', rects: [], origin: 'ai', location: 'not-found', reviewerId: 'me', createdAt: now },
          ],
        },
      },
      [b]: {
        'ft-exclusion': {
          status: 'manual',
          evidence: [{ id: 'e4', page: 2, quote: '', prefix: '', suffix: '', rects: [{ x: 0, y: 0, w: 0.5, h: 0.2 }], origin: 'manual', location: 'exact', reviewerId: 'me', createdAt: now }],
        },
      },
    },
  };
}

describe('review report model', () => {
  const review = sampleReview();
  const model = buildReportModel(review, records, REVIEW_COPY.pt, 'pt');

  it('lists the extraction form and the exclusions with their reasons', () => {
    expect(model.extractionForm).toEqual([
      ['Tamanho da amostra', 'Número', ''],
      ['Desenho', 'Escolha única', 'Ensaio; Coorte'],
    ]);
    expect(model.taReasons).toEqual([{ label: 'Fora do tema', count: 1 }]);
    expect(model.fullTextExcluded).toHaveLength(1);
    expect(model.fullTextExcluded[0]).toMatchObject({ reason: 'Desenho inadequado', quotes: [{ page: 2, text: 'Área da página' }] });
  });

  it('summarizes the final selection', () => {
    expect(model.synthesis.map((row) => row.summary)).toEqual(['média 130 · mín. 130 · máx. 130', 'Ensaio: 1']);
    expect(model.byYear).toEqual([{ year: '2020', count: 1 }]);
  });

  it('builds an audit trail per study: answer, check, AI suggestion and quotes', () => {
    expect(model.evidence.map((block) => block.key)).toEqual([a, b]);
    const [included, excluded] = model.evidence as [(typeof model.evidence)[0], (typeof model.evidence)[0]];
    expect(included.pdf).toBe('silva.pdf · 12 p.');
    expect(included.answers.map((answer) => [answer.group, answer.question, answer.answer, answer.status])).toEqual([
      ['extraction', 'Tamanho da amostra', '130', 'Corrigida'],
      ['extraction', 'Desenho', 'Ensaio', ''],
      ['quality', 'Há grupo controle?', 'Sim', 'Confirmada'],
    ]);
    expect(included.answers[0]!.suggestion).toBe('A IA sugere: 128 (deepseek-v4) — Métodos');
    expect(included.answers[2]!.suggestion).toBe('A IA sugere: Sim (deepseek-v4)');
    expect(quotesCell(included.answers[2]!.quotes, 'p.', '—')).toBe(
      'p. 4: “usual care group” (Citado pela IA · Achado (aproximado))\np. 9: “invented sentence” (Citado pela IA · Não localizado no PDF)',
    );
    expect(excluded.note).toBe('Coorte sem grupo controle');
    expect(excluded.answers).toHaveLength(1);
    expect(excluded.answers[0]).toMatchObject({ group: 'exclusion', answer: 'Desenho inadequado', status: 'Resposta manual' });
  });

  it('declares the AI use with the human check numbers', () => {
    const [method, ai] = fullTextParagraphs(model, REPORT_TEXT.pt, 'pt');
    expect(method).toContain('2 estudo(s)');
    expect(method).toContain('1 escaneado(s)');
    expect(ai).toContain('deepseek-v4');
    expect(ai).toContain('2 de 3 achado(s), 1 não localizado(s)');
    expect(ai).toContain('1 confirmada(s), 1 corrigida(s), 0 rejeitada(s)');
    const manual = buildReportModel({ ...review, evidence: {} }, records, REVIEW_COPY.pt, 'pt');
    expect(fullTextParagraphs(manual, REPORT_TEXT.pt, 'pt')[1]).toBe(REPORT_TEXT.pt.aiNotUsed);
  });
});

describe('review report PDF', () => {
  it('keeps what the standard PDF fonts can draw and decomposes the rest', () => {
    expect(pdfSafe('“Eﬃcacy” — São Paulo, 5 µg')).toBe('“Efficacy” — São Paulo, 5 µg');
    expect(pdfSafe('α-helix ≥ 3')).toBe('?-helix ? 3');
    expect(pdfSafe('linha 1\nlinha 2')).toBe('linha 1\nlinha 2');
  });

  it('includes the evidence sections and the quoted passages', () => {
    const review = sampleReview();
    const copy = REVIEW_COPY.pt;
    const flow = computeReviewFlow([], records, review.decisions, review.criteria, copy.noReason);
    const output = buildReviewPdf({ review, flow, records, copy, locale: 'pt' }).output();
    for (const expected of ['Evid', 'We recruited 130 adults', 'usual care group', 'silva.pdf', 'Estudos exclu', 'Formul', 'deepseek-v4']) {
      expect(output).toContain(expected);
    }
  });
});

describe('review report Word', () => {
  it('includes the evidence sections, one quote per paragraph', async () => {
    const review = sampleReview();
    const copy = REVIEW_COPY.pt;
    const flow = computeReviewFlow([], records, review.decisions, review.criteria, copy.noReason);
    const zip = await JSZip.loadAsync(await Packer.toBuffer(buildReviewDocx({ review, flow, records, copy, locale: 'pt' })));
    const xml = await zip.file('word/document.xml')!.async('string');
    for (const expected of ['Evidências por estudo', 'Formulário de extração', 'Estudos excluídos na leitura do texto completo', 'Síntese dos resultados', 'EXTRAÇÃO DE DADOS']) {
      expect(xml).toContain(expected);
    }
    expect(xml).toContain('p. 4: “usual care group” (Citado pela IA · Achado (aproximado))');
    expect(xml).toContain('p. 9: “invented sentence” (Citado pela IA · Não localizado no PDF)');
    expect(xml).not.toContain('group:');
  });
});
