import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Cpu, Database, KeyRound, Send, Sparkles, Square, User } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { streamChat, type ChatTurn, type ChatStatusUpdate } from '@/lib/ai-client';
import { useAiConfig } from '@/state/ai-config.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { getAiWorker } from '@/workers/client';
import { EmptyState } from '@/features/EmptyState';
import { cn } from '@/lib/utils';
import { AiSettingsModal } from '@/components/AiSettingsModal';
import { MarkdownContent } from '@/components/MarkdownContent';

/** Quantos documentos o BM25 seleciona por pergunta. */
const CONTEXT_SIZE = 40;

export default function ChatTab() {
  const active = useDataset((state) => state.active);
  const t = useLocale((state) => state.t);
  const isAiConfigured = useAiConfig((state) => state.isConfigured());

  const [aiModalOpen, setAiModalOpen] = useState(false);
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const displayedMessages: ChatTurn[] = useMemo(() => {
    return [{ role: 'assistant', content: t('chat_greeting') }, ...messages];
  }, [messages, t]);

  useEffect(() => {
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [displayedMessages, currentStatus]);

  const suggestions = [
    t('chat_sugg_1'),
    t('chat_sugg_2'),
    t('chat_sugg_3'),
    t('chat_sugg_4'),
  ];

  const ask = useCallback(
    async (question: string): Promise<void> => {
      if (!active || !question.trim() || streaming) return;

      const history = messages;
      setMessages((current) => [
        ...current,
        { role: 'user', content: question },
        { role: 'assistant', content: '', toolsExecuted: [] },
      ]);
      setDraft('');
      setError(null);
      setStreaming(true);
      setCurrentStatus(null);

      const controller = new AbortController();
      abortRef.current = controller;
      const executedTools: string[] = [];

      try {
        const context = await getAiWorker().buildChatContext(active, question, CONTEXT_SIZE);

        await streamChat({
          question,
          history,
          context,
          dataset: active,
          signal: controller.signal,
          onStatus: (status: ChatStatusUpdate) => {
            if (status.type === 'tool_call') {
              setCurrentStatus(status.message);
              if (status.toolName && !executedTools.includes(status.toolName)) {
                executedTools.push(status.toolName);
              }
            } else if (status.type === 'tool_result') {
              setCurrentStatus(null);
            }
          },
          onChunk: (text) =>
            setMessages((current) => {
              const next = [...current];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') {
                const updatedTools = executedTools.length > 0 ? [...executedTools] : last.toolsExecuted;
                next[next.length - 1] = {
                  role: 'assistant',
                  content: last.content + text,
                  ...(updatedTools ? { toolsExecuted: updatedTools } : {}),
                };
              }
              return next;
            }),
        });
      } catch (cause) {
        if (controller.signal.aborted) return;

        const message = cause instanceof Error ? cause.message : String(cause);
        setError(message);
        setMessages((current) => {
          const last = current[current.length - 1];
          return last?.role === 'assistant' && last.content === '' ? current.slice(0, -1) : current;
        });
      } finally {
        setStreaming(false);
        setCurrentStatus(null);
        abortRef.current = null;
      }
    },
    [active, messages, streaming],
  );

  if (!active) {
    return (
      <EmptyState
        title={t('chat_title')}
        description={t('empty_generic_desc')}
      />
    );
  }

  return (
    <>
      <Card className="border-t-4 border-t-emerald-500 shadow-xs">
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2.5 text-base font-bold text-foreground">
              <div className="flex size-7 items-center justify-center rounded-lg bg-emerald-100 text-emerald-600 shadow-2xs dark:bg-emerald-950 dark:text-emerald-400">
                <Bot className="size-4" aria-hidden />
              </div>
              {t('chat_title')}
            </CardTitle>

            <div className="flex items-center gap-2">
              <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50/70 px-2.5 py-1 text-[11px] font-medium text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300">
                <Cpu className="size-3 text-emerald-600 dark:text-emerald-400" />
                <span>{t('chat_tools_badge')}</span>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setAiModalOpen(true)}
                className="gap-1.5 rounded-lg text-xs font-semibold"
              >
                <KeyRound className="size-3.5 text-purple-600" />
                <span>{isAiConfigured ? t('ai_configured') : t('ai_settings_btn')}</span>
              </Button>
            </div>
          </div>
          <CardDescription>
            {t('chat_desc')}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          {!isAiConfigured && (
            <div className="rounded-xl border border-purple-200 bg-purple-50/70 p-3 text-xs text-purple-900 flex flex-wrap items-center justify-between gap-2 dark:border-purple-900 dark:bg-purple-950/40 dark:text-purple-300">
              <div className="flex items-center gap-2">
                <KeyRound className="size-4 shrink-0 text-purple-600" />
                <span>{t('chat_no_key_warning')}</span>
              </div>
              <Button
                variant="ai"
                size="sm"
                onClick={() => setAiModalOpen(true)}
                className="h-7 text-xs font-bold"
              >
                {t('ai_settings_btn')}
              </Button>
            </div>
          )}

          <div
            ref={scrollRef}
            className="max-h-[30rem] space-y-3.5 overflow-y-auto rounded-xl border border-border/80 bg-slate-50/50 p-4 dark:bg-slate-900/30"
          >
            {displayedMessages.map((message, index) => (
              <div
                key={index}
                className={cn('flex gap-3', message.role === 'user' ? 'flex-row-reverse' : 'flex-row')}
              >
                <div
                  className={cn(
                    'grid size-8 shrink-0 place-items-center rounded-full shadow-2xs font-semibold text-xs',
                    message.role === 'user'
                      ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white'
                      : 'bg-emerald-600 text-white',
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
                    'max-w-[85%] rounded-xl px-4 py-2.5 text-sm shadow-2xs leading-relaxed',
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium'
                      : 'border border-border/80 bg-card text-foreground',
                  )}
                >
                  {/* Badge de Ferramentas Executadas Localmente */}
                  {message.role === 'assistant' && message.toolsExecuted && message.toolsExecuted.length > 0 && (
                    <div className="mb-2.5 flex items-center gap-1.5 rounded-md border border-emerald-200/80 bg-emerald-50/60 px-2 py-0.5 text-[11px] font-medium text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
                      <Database className="size-3 text-emerald-600 dark:text-emerald-400" />
                      <span>
                        {message.toolsExecuted.length === 1
                          ? t('chat_tool_executed')
                          : `${message.toolsExecuted.length} ${t('chat_tools_executed_count')}`}
                      </span>
                    </div>
                  )}

                  {message.role === 'user' ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  ) : message.content ? (
                    <MarkdownContent content={message.content} />
                  ) : streaming && index === displayedMessages.length - 1 ? (
                    <div className="flex items-center gap-2 text-muted-foreground text-xs py-1">
                      <span className="relative flex size-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex size-2 rounded-full bg-emerald-500" />
                      </span>
                      <span>{currentStatus || t('chat_analyzing')}</span>
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>

          {messages.length === 0 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="size-3.5 text-amber-500" />
                {t('chat_suggestions_label')}
              </p>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    size="sm"
                    className="h-auto whitespace-normal rounded-lg py-2 text-left text-xs font-normal transition-all hover:border-emerald-400 hover:bg-emerald-50/50 hover:text-emerald-900 dark:hover:bg-emerald-950/40 dark:hover:text-emerald-300"
                    onClick={() => void ask(suggestion)}
                  >
                    💡 {suggestion}
                  </Button>
                ))}
              </div>
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
              placeholder={t('chat_placeholder')}
              aria-label="Pergunta para o assistente"
              disabled={streaming}
              className="rounded-lg shadow-2xs"
            />

            {streaming ? (
              <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}>
                <Square className="size-4" aria-hidden />
                {t('chat_btn_stop')}
              </Button>
            ) : (
              <Button
                type="submit"
                variant="gradient"
                disabled={!draft.trim()}
                className="font-semibold shadow-xs"
              >
                <Send className="size-4" aria-hidden />
                {t('chat_btn_send')}
              </Button>
            )}
          </form>
        </CardContent>
      </Card>

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </>
  );
}
