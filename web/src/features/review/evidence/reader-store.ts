import { create } from 'zustand';

import type { EvidenceTargetKey } from '@/core/review/types';

/** De onde o leitor foi aberto: define quais perguntas aparecem ao lado do PDF. */
export type ReaderScope = 'extraction' | 'quality' | 'full-text';

export interface ReaderRequest {
  studyKey: string;
  scope: ReaderScope;
  /** Pergunta a deixar ativa e, opcionalmente, o trecho a mostrar. */
  target?: EvidenceTargetKey;
  evidenceId?: string;
}

/**
 * O leitor de PDF aberto. Num store porque o pedem lugares diferentes — o botão do texto
 * completo, a página de um trecho embaixo de cada campo — e ele é um só.
 */
export const useEvidenceReader = create<{
  request: ReaderRequest | null;
  open: (request: ReaderRequest) => void;
  close: () => void;
}>()((set) => ({
  request: null,
  open: (request) => set({ request }),
  close: () => set({ request: null }),
}));
