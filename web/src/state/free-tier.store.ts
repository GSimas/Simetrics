import { create } from 'zustand';

import { getDeviceId } from '@/lib/device-id';

/**
 * O que o servidor oferece sem chave própria e quanto resta da cota deste dispositivo:
 * Jev livre, perguntas do Simi e classificações híbridas via DeepSeek do servidor.
 */

export interface FreeQuota {
  limit: number;
  used: number;
  remaining: number;
}

export interface ServerStatus {
  deepseek: { available: boolean; model: string | null };
  jev: { available: boolean };
  simi: FreeQuota;
  hybrid: FreeQuota;
  /** Maior base (documentos) que pode usar o DeepSeek e o Jev do servidor. */
  freeMaxDocs: number;
}

interface FreeTierState {
  status: ServerStatus | null;
  /** Falha ao consultar o servidor (sem as funções rodando, por exemplo). */
  unreachable: boolean;
  refresh: () => Promise<void>;
}

export const useFreeTier = create<FreeTierState>((set) => ({
  status: null,
  unreachable: false,
  async refresh() {
    try {
      const response = await fetch('/api/status', { headers: { 'X-Simetrics-Device': getDeviceId() } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      set({ status: (await response.json()) as ServerStatus, unreachable: false });
    } catch {
      set({ unreachable: true });
    }
  },
}));

if (typeof window !== 'undefined') void useFreeTier.getState().refresh();
