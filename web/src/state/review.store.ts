import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import { toScreeningRecords, type ScreeningRecord } from '@/core/review/records';
import { createReview } from '@/core/review/state';
import {
  REVIEW_DEFAULTS,
  type CriterionKind,
  type ExtractionValue,
  type FullTextDecision,
  type QualityAnswer,
  type RecordScreening,
  type ReviewState,
  type ReviewType,
  type TitleAbstractDecision,
} from '@/core/review/types';
import { getDeviceId } from '@/lib/device-id';
import type { Dataset } from '@/lib/types';
import { useDataset } from './dataset.store';

/**
 * Estado da revisão sistematizada do projeto aberto. Salvo junto do projeto (ver
 * `project.store.ts`, que assina este store e grava a cada mudança).
 */

type Editable = Omit<
  ReviewState,
  'schemaVersion' | 'decisions' | 'quality' | 'extraction' | 'reviewerId' | 'createdAt' | 'updatedAt'
>;

type ListName = 'questions' | 'concepts' | 'criteria' | 'qualityQuestions' | 'qualityAnswers' | 'extractionFields';

interface ReviewStoreState {
  review: ReviewState | null;

  /** Substitui a revisão inteira — ao abrir um projeto ou limpar o workspace. */
  hydrate: (review: ReviewState | null) => void;
  update: (patch: Partial<Editable>) => void;
  setType: (type: ReviewType) => void;
  /** Os `add*` devolvem o id do item criado — a tela põe o foco nele. */
  addQuestion: () => string;
  addConcept: () => string;
  addCriterion: (kind: CriterionKind) => string;
  /** A primeira pergunta de qualidade traz junto as respostas padrão (Sim/Parcialmente/Não). */
  addQualityQuestion: (defaultAnswers: () => QualityAnswer[]) => string;
  addQualityAnswer: () => string;
  addExtractionField: () => string;
  removeItem: (list: ListName, id: string) => void;
  decideTitleAbstract: (key: string, decision: TitleAbstractDecision | null, reason?: string) => void;
  decideFullText: (key: string, decision: FullTextDecision | null, reason?: string) => void;
  setNote: (key: string, note: string) => void;
  answerQuality: (key: string, questionId: string, answerId: string | null) => void;
  setExtractionValue: (key: string, fieldId: string, value: ExtractionValue | null) => void;
  setExtractionDone: (key: string, done: boolean) => void;
}

/** O exemplo é só visualização: nenhuma ação altera a revisão dele. */
function readOnly(): boolean {
  return useDataset.getState().isDemo;
}

function current(review: ReviewState | null): ReviewState {
  return review ?? createReview(getDeviceId());
}

function touch(review: ReviewState, patch: Partial<ReviewState>): ReviewState {
  return { ...review, ...patch, updatedAt: new Date().toISOString() };
}

/** Grava a decisão de um registro; um registro sem decisão nem nota sai do mapa. */
function withScreening(
  review: ReviewState,
  key: string,
  change: (screening: RecordScreening) => RecordScreening,
): ReviewState {
  const now = new Date().toISOString();
  const next = change({ ...(review.decisions[key] ?? { updatedAt: now }) });
  next.updatedAt = now;
  const decisions = { ...review.decisions };
  if (!next.ta && !next.ft && !next.note) delete decisions[key];
  else decisions[key] = next;
  return touch(review, { decisions });
}

export const useReview = create<ReviewStoreState>()(
  subscribeWithSelector((set, get) => ({
    review: null,

    hydrate: (review) => set({ review }),

    update(patch) {
      if (readOnly()) return;
      set({ review: touch(current(get().review), patch) });
    },

    setType(type) {
      if (readOnly()) return;
      const review = current(get().review);
      // A estrutura acompanha o tipo enquanto o usuário não preencheu nenhum campo dela.
      const untouched = Object.values(review.frameworkValues).every((value) => !value.trim());
      set({
        review: touch(review, {
          type,
          ...(untouched ? { framework: REVIEW_DEFAULTS[type].framework } : {}),
        }),
      });
    },

    addQuestion() {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { questions: [...review.questions, { id, text: '' }] }) });
      return id;
    },

    addConcept() {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { concepts: [...review.concepts, { id, label: '', terms: [] }] }) });
      return id;
    },

    addCriterion(kind) {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { criteria: [...review.criteria, { id, kind, text: '' }] }) });
      return id;
    },

    addQualityQuestion(defaultAnswers) {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({
        review: touch(review, {
          qualityQuestions: [...review.qualityQuestions, { id, text: '' }],
          ...(review.qualityAnswers.length === 0 ? { qualityAnswers: defaultAnswers() } : {}),
        }),
      });
      return id;
    },

    addQualityAnswer() {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { qualityAnswers: [...review.qualityAnswers, { id, label: '', weight: 0 }] }) });
      return id;
    },

    addExtractionField() {
      if (readOnly()) return '';
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({
        review: touch(review, {
          extractionFields: [...review.extractionFields, { id, label: '', type: 'text', options: [] }],
        }),
      });
      return id;
    },

    removeItem(list, id) {
      if (readOnly()) return;
      const review = current(get().review);
      const items = review[list] as { id: string }[];
      set({ review: touch(review, { [list]: items.filter((item) => item.id !== id) }) });
    },

    decideTitleAbstract(key, decision, reason) {
      if (readOnly()) return;
      set({
        review: withScreening(current(get().review), key, (screening) => {
          if (decision) screening.ta = decision;
          else delete screening.ta;
          if (decision === 'exclude' && reason) screening.taReason = reason;
          else delete screening.taReason;
          return screening;
        }),
      });
    },

    decideFullText(key, decision, reason) {
      if (readOnly()) return;
      set({
        review: withScreening(current(get().review), key, (screening) => {
          if (decision) screening.ft = decision;
          else delete screening.ft;
          if (decision === 'exclude' && reason) screening.ftReason = reason;
          else delete screening.ftReason;
          return screening;
        }),
      });
    },

    answerQuality(key, questionId, answerId) {
      if (readOnly()) return;
      const review = current(get().review);
      const responses = { ...(review.quality[key] ?? {}) };
      if (answerId) responses[questionId] = answerId;
      else delete responses[questionId];
      const quality = { ...review.quality };
      if (Object.keys(responses).length > 0) quality[key] = responses;
      else delete quality[key];
      set({ review: touch(review, { quality }) });
    },

    setExtractionValue(key, fieldId, value) {
      if (readOnly()) return;
      const review = current(get().review);
      const study = review.extraction[key] ?? { values: {}, done: false };
      const values = { ...study.values };
      if (value === null || value === '' || (Array.isArray(value) && value.length === 0)) delete values[fieldId];
      else values[fieldId] = value;
      set({ review: touch(review, { extraction: { ...review.extraction, [key]: { ...study, values } } }) });
    },

    setExtractionDone(key, done) {
      if (readOnly()) return;
      const review = current(get().review);
      const study = review.extraction[key] ?? { values: {}, done: false };
      set({ review: touch(review, { extraction: { ...review.extraction, [key]: { ...study, done } } }) });
    },

    setNote(key, note) {
      if (readOnly()) return;
      set({
        review: withScreening(current(get().review), key, (screening) => {
          if (note.trim()) screening.note = note;
          else delete screening.note;
          return screening;
        }),
      });
    },
  })),
);

// Registros da triagem, calculados uma vez por base ativa.
const recordsCache = new WeakMap<Dataset, ScreeningRecord[]>();
const EMPTY: ScreeningRecord[] = [];

export function screeningRecordsOf(active: Dataset | null): ScreeningRecord[] {
  if (!active) return EMPTY;
  let records = recordsCache.get(active);
  if (!records) {
    records = toScreeningRecords(active);
    recordsCache.set(active, records);
  }
  return records;
}

export function useScreeningRecords(): ScreeningRecord[] {
  return screeningRecordsOf(useDataset((state) => state.active));
}

/** A revisão aberta é só para ver (a de amostra do exemplo). */
export function useReviewReadOnly(): boolean {
  return useDataset((state) => state.isDemo);
}

export type ReviewStep = 'protocol' | 'screening' | 'quality' | 'extraction' | 'synthesis' | 'prisma';

/** Etapa aberta na aba da revisão — num store para o tour guiado poder trocá-la. */
export const useReviewNav = create<{ step: ReviewStep; setStep: (step: ReviewStep) => void }>()((set) => ({
  step: 'protocol',
  setStep: (step) => set({ step }),
}));
