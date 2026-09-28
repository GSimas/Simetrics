import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import { toScreeningRecords, type ScreeningRecord } from '@/core/review/records';
import { createReview } from '@/core/review/state';
import {
  REVIEW_DEFAULTS,
  type CriterionKind,
  type FullTextDecision,
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

type Editable = Omit<ReviewState, 'schemaVersion' | 'decisions' | 'reviewerId' | 'createdAt' | 'updatedAt'>;

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
  removeItem: (list: 'questions' | 'concepts' | 'criteria', id: string) => void;
  decideTitleAbstract: (key: string, decision: TitleAbstractDecision | null, reason?: string) => void;
  decideFullText: (key: string, decision: FullTextDecision | null, reason?: string) => void;
  setNote: (key: string, note: string) => void;
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

    update: (patch) => set({ review: touch(current(get().review), patch) }),

    setType(type) {
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
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { questions: [...review.questions, { id, text: '' }] }) });
      return id;
    },

    addConcept() {
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { concepts: [...review.concepts, { id, label: '', terms: [] }] }) });
      return id;
    },

    addCriterion(kind) {
      const review = current(get().review);
      const id = crypto.randomUUID();
      set({ review: touch(review, { criteria: [...review.criteria, { id, kind, text: '' }] }) });
      return id;
    },

    removeItem(list, id) {
      const review = current(get().review);
      const items = review[list] as { id: string }[];
      set({ review: touch(review, { [list]: items.filter((item) => item.id !== id) }) });
    },

    decideTitleAbstract(key, decision, reason) {
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

    setNote(key, note) {
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
