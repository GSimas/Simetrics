import { create } from 'zustand';

import type { ConfidenceThresholds } from '@/core/hybrid/types';

/**
 * Configuração da classificação híbrida — separada da configuração do chat
 * (`ai-config.store.ts`) porque aqui são dois provedores ao mesmo tempo, com papéis
 * diferentes: um gerativo que descobre as categorias e o Jev que classifica.
 *
 * Como no chat, as chaves ficam só no localStorage deste navegador.
 */

export interface HybridConfig {
  generative: { apiKey: string; model: string; baseUrl: string };
  jev: { apiKey: string; model: string };
  thresholds: ConfidenceThresholds;
  /** Tamanho da amostra enviada ao modelo gerativo. */
  sampleSize: number;
  maxCategories: number;
  /** Pausa para o usuário revisar as categorias antes de classificar a base. */
  reviewBeforeClassify: boolean;
  maxClarifyRounds: number;
  maxExpansionRounds: number;
  /** Fração de sobras ("outro" + não classificados) que dispara uma rodada de expansão. */
  leftoverTrigger: number;
  /** Requisições simultâneas ao Jev. */
  concurrency: number;
}

export const DEFAULT_HYBRID_CONFIG: HybridConfig = {
  generative: { apiKey: '', model: 'deepseek-flash', baseUrl: 'https://api.deepseek.com' },
  jev: { apiKey: '', model: 'jev-latest' },
  thresholds: { accept: 0.7, review: 0.4 },
  sampleSize: 120,
  maxCategories: 12,
  reviewBeforeClassify: true,
  maxClarifyRounds: 1,
  maxExpansionRounds: 1,
  leftoverTrigger: 0.12,
  concurrency: 8,
};

const STORAGE_KEY = 'simetrics_hybrid_config';

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}

/** Normaliza o que vier do armazenamento (ou do formulário) para valores seguros. */
export function normalizeHybridConfig(raw: Partial<HybridConfig> | null | undefined): HybridConfig {
  const d = DEFAULT_HYBRID_CONFIG;
  const accept = clamp(raw?.thresholds?.accept, 0.05, 1, d.thresholds.accept);
  return {
    generative: {
      apiKey: String(raw?.generative?.apiKey ?? ''),
      model: String(raw?.generative?.model || d.generative.model),
      baseUrl: String(raw?.generative?.baseUrl || d.generative.baseUrl),
    },
    jev: {
      apiKey: String(raw?.jev?.apiKey ?? ''),
      model: String(raw?.jev?.model || d.jev.model),
    },
    thresholds: { accept, review: Math.min(accept, clamp(raw?.thresholds?.review, 0, 1, d.thresholds.review)) },
    sampleSize: Math.round(clamp(raw?.sampleSize, 30, 300, d.sampleSize)),
    maxCategories: Math.round(clamp(raw?.maxCategories, 3, 30, d.maxCategories)),
    reviewBeforeClassify: raw?.reviewBeforeClassify ?? d.reviewBeforeClassify,
    maxClarifyRounds: Math.round(clamp(raw?.maxClarifyRounds, 0, 2, d.maxClarifyRounds)),
    maxExpansionRounds: Math.round(clamp(raw?.maxExpansionRounds, 0, 2, d.maxExpansionRounds)),
    leftoverTrigger: clamp(raw?.leftoverTrigger, 0.02, 0.5, d.leftoverTrigger),
    concurrency: Math.round(clamp(raw?.concurrency, 1, 24, d.concurrency)),
  };
}

function load(): HybridConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return normalizeHybridConfig(JSON.parse(raw) as Partial<HybridConfig>);
  } catch {
    // JSON inválido ou armazenamento bloqueado: segue com o padrão.
  }
  return DEFAULT_HYBRID_CONFIG;
}

interface HybridConfigState {
  config: HybridConfig;
  setConfig: (config: HybridConfig) => void;
  clearKeys: () => void;
}

function persist(config: HybridConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // Cota ou armazenamento bloqueado: a configuração vale só para esta sessão.
  }
}

export const useHybridConfig = create<HybridConfigState>((set, get) => ({
  config: typeof window === 'undefined' ? DEFAULT_HYBRID_CONFIG : load(),
  setConfig(config) {
    const next = normalizeHybridConfig(config);
    persist(next);
    set({ config: next });
  },
  clearKeys() {
    const { config } = get();
    const next = { ...config, generative: { ...config.generative, apiKey: '' }, jev: { ...config.jev, apiKey: '' } };
    persist(next);
    set({ config: next });
  },
}));
