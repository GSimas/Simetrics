import { create } from 'zustand';

import { streamChat, type ChatTurn } from '@/lib/ai-client';
import { getAiWorker } from '@/workers/client';
import { useDataset } from './dataset.store';

/** Quantos documentos o BM25 seleciona por pergunta. */
export const CONTEXT_SIZE = 40;

export interface ChatMessage extends ChatTurn {
  /** Títulos dos documentos enviados ao modelo como contexto desta resposta. */
  sources?: string[] | undefined;
}

interface ChatState {
  messages: ChatMessage[];
  streaming: boolean;
  /** Ferramenta em execução ("Consultando tabela…"), enquanto o texto não chega. */
  status: string | null;
  error: string | null;
  ask: (question: string) => Promise<void>;
  stop: () => void;
  clear: () => void;
}

let controller: AbortController | null = null;

/**
 * A conversa com a Simi — uma só, compartilhada pela tela de conversa do Motor de Busca e
 * pelo widget flutuante: quem pergunta na busca e vai para as Redes encontra a mesma
 * conversa na Simi.
 */
export const useChat = create<ChatState>()((set, get) => ({
  messages: [],
  streaming: false,
  status: null,
  error: null,

  async ask(question) {
    const active = useDataset.getState().active;
    const text = question.trim();
    if (!active || !text || get().streaming) return;

    const history = get().messages;
    set({
      messages: [...history, { role: 'user', content: text }, { role: 'assistant', content: '', toolsExecuted: [] }],
      error: null,
      streaming: true,
      status: null,
    });

    const current = new AbortController();
    controller = current;
    const executedTools: string[] = [];
    // Atualiza a última mensagem (a resposta em andamento).
    const patchReply = (patch: (reply: ChatMessage) => ChatMessage): void =>
      set((state) => {
        const next = [...state.messages];
        const last = next[next.length - 1];
        if (last?.role === 'assistant') next[next.length - 1] = patch(last);
        return { messages: next };
      });

    try {
      const context = await getAiWorker().buildChatContext(active, text, CONTEXT_SIZE);
      const sources = [...new Set(context.documents.map((doc) => doc.title.trim()).filter(Boolean))];
      patchReply((reply) => ({ ...reply, sources }));

      await streamChat({
        question: text,
        history,
        context,
        dataset: active,
        signal: current.signal,
        onStatus: (status) => {
          if (status.type === 'tool_call') {
            set({ status: status.message });
            if (status.toolName && !executedTools.includes(status.toolName)) executedTools.push(status.toolName);
          } else if (status.type === 'tool_result') {
            set({ status: null });
          }
        },
        onChunk: (chunk) =>
          patchReply((reply) => ({
            ...reply,
            content: reply.content + chunk,
            toolsExecuted: executedTools.length > 0 ? [...executedTools] : reply.toolsExecuted,
          })),
      });
    } catch (cause) {
      if (current.signal.aborted) return;
      set((state) => {
        const last = state.messages[state.messages.length - 1];
        return {
          error: cause instanceof Error ? cause.message : String(cause),
          // Resposta que nem começou não fica como balão vazio.
          messages: last?.role === 'assistant' && last.content === '' ? state.messages.slice(0, -1) : state.messages,
        };
      });
    } finally {
      // Uma pergunta mais nova pode já ter assumido; só a dona limpa o estado.
      if (controller === current) {
        controller = null;
        set({ streaming: false, status: null });
      }
    }
  },

  stop() {
    controller?.abort();
    controller = null;
    set({ streaming: false, status: null });
  },

  clear() {
    get().stop();
    set({ messages: [], error: null });
  },
}));

// Uma resposta em andamento não sobrevive à troca de base: continuaria gastando a chave de
// API e rodando ferramentas sobre a base antiga.
useDataset.subscribe(
  (state) => state.active,
  () => useChat.getState().stop(),
);
