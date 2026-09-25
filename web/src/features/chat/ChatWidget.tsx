import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Database, KeyRound, MessageSquare, Send, Sparkles, Square, Trash2, User, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { streamChat, type ChatTurn, type ChatStatusUpdate } from '@/lib/ai-client';
import { useAiConfig } from '@/state/ai-config.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { getAiWorker } from '@/workers/client';
import { cn } from '@/lib/utils';
import { AiSettingsModal } from '@/components/AiSettingsModal';
import { FreeQuotaNotice } from '@/components/FreeQuotaNotice';
import { useFreeTier } from '@/state/free-tier.store';
import { MarkdownContent } from '@/components/MarkdownContent';
import { usePresence } from '@/lib/use-presence';

/**
 * Markdown memoizado: durante o streaming o chat re-renderiza a cada trecho, e sem isso
 * todas as respostas anteriores eram re-parseadas a cada vez (custo quadrático na conversa).
 */
const MessageMarkdown = memo(MarkdownContent);

/** Quantos documentos o BM25 seleciona por pergunta. */
const CONTEXT_SIZE = 40;

export function ChatWidget() {
  const active = useDataset((state) => state.active);
  const { t, locale } = useLocale();
  const isEn = locale === 'en';
  const { config, isConfigured } = useAiConfig();
  const isAiConfigured = isConfigured();
  const freeStatus = useFreeTier((state) => state.status);
  const freeSimi = freeStatus?.deepseek.available && freeStatus.simi.limit > 0 ? freeStatus.simi : null;

  const [isOpen, setIsOpen] = useState(false);
  const panel = usePresence(isOpen);
  const [aiModalOpen, setAiModalOpen] = useState(false);
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [draft, setDraft] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const displayedMessages: ChatTurn[] = useMemo(() => {
    return [{ role: 'assistant', content: t('chat_greeting') }, ...messages];
  }, [messages, t]);

  // Rolagem acompanha cada trecho da resposta; o foco vai para o campo só ao abrir (antes
  // era devolvido a cada trecho recebido, com um timer por trecho que ninguém limpava).
  useEffect(() => {
    if (!isOpen) return;
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [isOpen, displayedMessages, currentStatus]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(() => inputRef.current?.focus(), 100);
    return () => clearTimeout(timer);
  }, [isOpen]);

  // Uma resposta em andamento não sobrevive à troca de base nem à saída do workspace:
  // continuaria gastando a chave de API e rodando ferramentas sobre a base antiga.
  useEffect(() => () => abortRef.current?.abort(), [active]);

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

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <>
      {/* Botão de Ação Flutuante (FAB) no Canto Inferior Direito (acima do café) */}
      <div data-tour="chat" className="fixed bottom-[60px] right-5 z-50 flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={() => setIsOpen((prev) => !prev)}
          className={cn(
            'group relative flex items-center rounded-full py-2.5 sm:py-3 text-white shadow-2xl transition-all duration-300 hover:scale-105 active:scale-95 focus:outline-hidden',
            isOpen
              ? 'gap-2.5 px-4 bg-slate-800 dark:bg-slate-700'
              : 'gap-0 px-3.5 hover:gap-2.5 bg-primary text-primary-foreground shadow-[0_0_28px_-8px_var(--highlight)]',
          )}
          title={isOpen ? (isEn ? 'Close Assistant' : 'Fechar Assistente') : t('chat_title')}
          aria-label={t('chat_title')}
          aria-expanded={isOpen}
        >
          {isOpen ? (
            <X className="size-5 transition-transform duration-200 group-hover:rotate-90" />
          ) : (
            <>
              <div className="relative shrink-0">
                <Bot className="size-5" />
                <span className="absolute -right-1 -top-1 flex size-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary-foreground opacity-50" />
                  <span className="relative inline-flex size-2.5 rounded-full bg-primary-foreground" />
                </span>
              </div>
              <span className="grid grid-cols-[0fr] group-hover:grid-cols-[1fr] transition-[grid-template-columns] duration-300 ease-out">
                <span className="flex items-center gap-1.5 overflow-hidden">
                  <span className="text-xs font-bold tracking-wide whitespace-nowrap">
                    {t('chat_title')}
                  </span>
                  <Sparkles className="size-3.5 shrink-0 text-amber-300 animate-pulse" />
                </span>
              </span>
            </>
          )}
        </button>
      </div>

      {/* Janela Flutuante do Widget de Chat */}
      {panel.mounted && (
        <div
          className={cn(
            'fixed bottom-[116px] right-4 sm:right-6 z-50 flex h-[540px] max-h-[72vh] w-[94vw] sm:w-[440px] origin-bottom-right flex-col overflow-hidden rounded-2xl border border-border/90 bg-card/95 shadow-2xl backdrop-blur-md duration-200',
            panel.closing
              ? 'animate-out fade-out-0 zoom-out-95 slide-out-to-bottom-2 fill-mode-forwards'
              : 'animate-in fade-in-0 zoom-in-95 slide-in-from-bottom-4',
          )}
        >
          {/* Cabeçalho do Widget */}
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div className="flex items-center gap-2.5">
              <div className="flex size-8 items-center justify-center bg-primary text-primary-foreground">
                <Bot className="size-4.5" aria-hidden />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <h2 className="text-xs sm:text-sm font-bold text-foreground">
                    {t('chat_title')}
                  </h2>
                  <span className="size-2 rounded-full bg-emerald-500" title="Online" />
                </div>
                <p className="text-[10px] text-muted-foreground truncate max-w-[190px]">
                  {isAiConfigured
                    ? `${config.provider.toUpperCase()} · ${config.model}`
                    : freeSimi
                      ? t('chat_free_subtitle')
                          .replace('{remaining}', String(freeSimi.remaining))
                          .replace('{limit}', String(freeSimi.limit))
                      : t('ai_not_configured')}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setAiModalOpen(true)}
                className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                title={t('ai_settings_btn')}
                aria-label={t('ai_settings_btn')}
              >
                <KeyRound className="size-4 text-purple-600" />
              </button>

              {messages.length > 0 && (
                <button
                  type="button"
                  onClick={clearChat}
                  className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
                  title={isEn ? 'Clear history' : 'Limpar conversa'}
                  aria-label={isEn ? 'Clear history' : 'Limpar conversa'}
                >
                  <Trash2 className="size-4" />
                </button>
              )}

              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                title={isEn ? 'Minimize' : 'Minimizar'}
                aria-label={isEn ? 'Close' : 'Fechar'}
              >
                <X className="size-4" />
              </button>
            </div>
          </div>

          {/* Área de Mensagens */}
          {/* role="log": leitores de tela anunciam mensagens novas; aria-busy segura o
              anúncio até a resposta terminar de chegar, em vez de ler trecho por trecho. */}
          <div
            ref={scrollRef}
            role="log"
            aria-live="polite"
            aria-busy={streaming}
            aria-label={t('chat_title')}
            className="flex-1 space-y-3 overflow-y-auto p-3.5 text-xs bg-slate-50/50 dark:bg-slate-900/30"
          >
            {!active ? (
              <div className="grid h-full place-items-center text-center p-4 text-muted-foreground">
                <div>
                  <MessageSquare className="mx-auto mb-2 size-8 text-muted-foreground/40" />
                  <p className="font-semibold text-foreground text-xs mb-1">
                    {isEn ? 'No active dataset' : 'Nenhuma base carregada'}
                  </p>
                  <p className="text-[11px]">
                    {t('empty_generic_desc')}
                  </p>
                </div>
              </div>
            ) : (
              <>
                <FreeQuotaNotice compact onConfigure={() => setAiModalOpen(true)} />

                {displayedMessages.map((message, index) => (
                  <div
                    key={index}
                    className={cn(
                      'flex gap-2',
                      message.role === 'user' ? 'flex-row-reverse' : 'flex-row',
                    )}
                  >
                    <div
                      className={cn(
                        'grid size-6.5 shrink-0 place-items-center rounded-full text-[10px] font-semibold shadow-2xs',
                        message.role === 'user'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-emerald-600 text-white',
                      )}
                    >
                      {message.role === 'user' ? (
                        <User className="size-3.5" aria-hidden />
                      ) : (
                        <Bot className="size-3.5" aria-hidden />
                      )}
                    </div>

                    <div
                      className={cn(
                        'max-w-[88%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed shadow-2xs',
                        message.role === 'user'
                          ? 'bg-primary text-primary-foreground font-medium'
                          : 'border border-border/80 bg-card text-foreground',
                      )}
                    >
                      {/* Badge de Consulta Local */}
                      {message.role === 'assistant' && message.toolsExecuted && message.toolsExecuted.length > 0 && (
                        <div className="mb-2 flex items-center gap-1.5 rounded-md border border-emerald-200/80 bg-emerald-50/70 px-2 py-0.5 text-[10px] font-medium text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
                          <Database className="size-2.5 text-emerald-600 dark:text-emerald-400" />
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
                        <MessageMarkdown content={message.content} />
                      ) : streaming && index === displayedMessages.length - 1 ? (
                        <span className="flex items-center gap-1.5 text-muted-foreground text-[11px]">
                          <span className="size-1.5 animate-bounce rounded-full bg-emerald-500" />
                          {currentStatus || t('chat_analyzing')}
                        </span>
                      ) : null}
                    </div>
                  </div>
                ))}

                {messages.length === 0 && (
                  <div className="mt-2 space-y-1.5">
                    <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                      {t('chat_suggestions_label')}
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {suggestions.map((suggestion) => (
                        <button
                          key={suggestion}
                          type="button"
                          className="rounded-lg border border-border/80 bg-card/80 px-2.5 py-1.5 text-left text-[11px] font-medium text-foreground transition-all hover:border-emerald-400 hover:bg-emerald-50/50 hover:text-emerald-950 dark:hover:bg-emerald-950/40 dark:hover:text-emerald-300"
                          onClick={() => void ask(suggestion)}
                        >
                          💡 {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {error && (
              <p className="rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-[11px] text-destructive">
                {error}
              </p>
            )}
          </div>

          {/* Rodapé com Campo de Entrada */}
          <div className="border-t border-border/80 bg-card p-2.5">
            <form
              className="flex gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void ask(draft);
              }}
            >
              <Input
                ref={inputRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={t('chat_placeholder')}
                aria-label={isEn ? 'Question for Simi' : 'Pergunta para a Simi'}
                disabled={streaming || !active}
                className="h-9 rounded-lg text-xs"
              />

              {streaming ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => abortRef.current?.abort()}
                  className="h-9 gap-1 text-xs"
                >
                  <Square className="size-3.5" aria-hidden />
                  <span>{t('chat_btn_stop')}</span>
                </Button>
              ) : (
                <Button
                  type="submit"
                  variant="gradient"
                  size="sm"
                  disabled={!draft.trim() || !active}
                  aria-label={t('chat_send')}
                  title={t('chat_send')}
                  className="h-9 px-3 text-xs font-semibold shadow-xs"
                >
                  <Send className="size-3.5" aria-hidden />
                </Button>
              )}
            </form>
          </div>
        </div>
      )}

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </>
  );
}
