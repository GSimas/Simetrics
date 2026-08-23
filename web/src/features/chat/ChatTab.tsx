import { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, Send, Square, User } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { streamChat, type ChatTurn } from '@/lib/gemini-client';
import { useDataset } from '@/state/dataset.store';
import { getAiWorker } from '@/workers/client';
import { EmptyState } from '@/features/EmptyState';
import { cn } from '@/lib/utils';

/**
 * Assistente Científico.
 *
 * O fluxo de cada pergunta: o worker seleciona por BM25 os documentos relevantes na base
 * carregada, a função serverless recebe apenas esse recorte, e a resposta volta em
 * streaming. A base nunca sai do navegador inteira.
 */

const GREETING: ChatTurn = {
  role: 'assistant',
  content:
    'Olá! Sou a IA do Simetrics. Respondo com base nos documentos da sua base — posso ' +
    'recomendar leituras fundamentais, identificar especialistas para parceria ou sugerir ' +
    'periódicos para submissão. O que você precisa investigar?',
};

/** Perguntas de partida, para quem não sabe por onde começar. */
const SUGGESTIONS = [
  'Quais são os documentos fundamentais desta base?',
  'Quem são os autores mais influentes e em que eles trabalham?',
  'Em quais periódicos eu deveria submeter um artigo sobre este tema?',
  'Que lacunas de pesquisa aparecem nesta literatura?',
] as const;

/** Quantos documentos o BM25 seleciona por pergunta. */
const CONTEXT_SIZE = 40;

export default function ChatTab() {
  const active = useDataset((state) => state.active);
  const [messages, setMessages] = useState<ChatTurn[]>([GREETING]);
  const [draft, setDraft] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Rola para o fim a cada mensagem nova, para que o texto em streaming fique visível.
  useEffect(() => {
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [messages]);

  const ask = useCallback(
    async (question: string): Promise<void> => {
      if (!active || !question.trim() || streaming) return;

      const history = messages.filter((message) => message !== GREETING);
      setMessages((current) => [
        ...current,
        { role: 'user', content: question },
        { role: 'assistant', content: '' },
      ]);
      setDraft('');
      setError(null);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const context = await getAiWorker().buildChatContext(active, question, CONTEXT_SIZE);

        await streamChat({
          question,
          history,
          context,
          signal: controller.signal,
          onChunk: (text) =>
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') {
                next[next.length - 1] = { role: 'assistant', content: last.content + text };
              }
              return next;
            }),
        });
      } catch (cause) {
        if (controller.signal.aborted) return;

        const message = cause instanceof Error ? cause.message : String(cause);
        setError(message);
        // Remove a bolha vazia deixada pela resposta que não veio.
        setMessages((current) => {
          const last = current[current.length - 1];
          return last?.role === 'assistant' && last.content === '' ? current.slice(0, -1) : current;
        });
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [active, messages, streaming],
  );

  if (!active) {
    return (
      <EmptyState
        title="Assistente Científico"
        description="Carregue uma base na aba Informações Principais para conversar sobre ela."
      />
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Assistente Científico</CardTitle>
        <CardDescription>
          Conversando com {active.length.toLocaleString('pt-BR')} documentos. A cada
          pergunta, os {CONTEXT_SIZE} documentos mais relevantes são selecionados no seu
          navegador e só eles são enviados ao modelo. A qualidade das respostas depende dos
          metadados — verifique sempre contra os dados brutos.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div ref={scrollRef} className="max-h-[30rem] space-y-3 overflow-y-auto rounded-md border p-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={cn('flex gap-3', message.role === 'user' ? 'flex-row-reverse' : 'flex-row')}
            >
              <div
                className={cn(
                  'grid size-7 shrink-0 place-items-center rounded-full',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground',
                )}
              >
                {message.role === 'user' ? (
                  <User className="size-4" aria-hidden />
                ) : (
                  <Bot className="size-4" aria-hidden />
                )}
              </div>

              <div
                className={cn(
                  'max-w-[80%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm',
                  message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted',
                )}
              >
                {message.content ||
                  (streaming && index === messages.length - 1 ? (
                    <span className="text-muted-foreground">Analisando a base…</span>
                  ) : null)}
              </div>
            </div>
          ))}
        </div>

        {messages.length === 1 && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <Button
                key={suggestion}
                variant="outline"
                size="sm"
                className="h-auto whitespace-normal py-1.5 text-left text-xs font-normal"
                onClick={() => void ask(suggestion)}
              >
                {suggestion}
              </Button>
            ))}
          </div>
        )}

        {error && (
          <p className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
            {error}
          </p>
        )}

        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void ask(draft);
          }}
        >
          <Input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ex.: quais são os documentos fundamentais sobre este tema?"
            aria-label="Pergunta para o assistente"
            disabled={streaming}
          />

          {streaming ? (
            <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}>
              <Square aria-hidden />
              Parar
            </Button>
          ) : (
            <Button type="submit" disabled={!draft.trim()}>
              <Send aria-hidden />
              Enviar
            </Button>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
