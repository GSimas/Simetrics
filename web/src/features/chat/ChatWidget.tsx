import { useEffect, useRef, useState } from 'react';
import { Bot, KeyRound, MessageSquare, Send, Square, Trash2, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useAiConfig } from '@/state/ai-config.store';
import { useChat } from '@/state/chat.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { cn } from '@/lib/utils';
import { AiSettingsModal } from '@/components/AiSettingsModal';
import { useFreeTier } from '@/state/free-tier.store';
import { usePresence } from '@/lib/use-presence';
import { ChatThread, PrivacyNote } from './ChatThread';

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
  const messages = useChat((state) => state.messages);
  const streaming = useChat((state) => state.streaming);
  const status = useChat((state) => state.status);
  const ask = useChat((state) => state.ask);
  const [draft, setDraft] = useState('');

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const fabRef = useRef<HTMLDivElement>(null);

  // Rolagem acompanha cada trecho da resposta; o foco vai para o campo só ao abrir (antes
  // era devolvido a cada trecho recebido, com um timer por trecho que ninguém limpava).
  useEffect(() => {
    if (!isOpen) return;
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [isOpen, messages, status]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(() => inputRef.current?.focus(), 100);
    return () => clearTimeout(timer);
  }, [isOpen]);

  // Clique fora fecha a janela. Não contam como "fora": o botão flutuante (que já alterna
  // sozinho) e o que a própria Simi abre em portal — o modal da chave e menus suspensos.
  useEffect(() => {
    if (!isOpen || aiModalOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (!target || panelRef.current?.contains(target) || fabRef.current?.contains(target)) return;
      if (target.closest('[role="dialog"], [data-radix-popper-content-wrapper]')) return;
      setIsOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [isOpen, aiModalOpen]);

  return (
    <>
      {/* Botão de Ação Flutuante (FAB) no Canto Inferior Direito (acima do café) */}
      <div ref={fabRef} data-tour="chat" className="fixed bottom-[60px] right-5 z-50 flex items-center justify-end gap-2">
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
                <span className="flex items-center overflow-hidden">
                  <span className="text-xs font-bold tracking-wide whitespace-nowrap">
                    {t('chat_title')}
                  </span>
                </span>
              </span>
            </>
          )}
        </button>
      </div>

      {/* Janela Flutuante do Widget de Chat */}
      {panel.mounted && (
        <div
          ref={panelRef}
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
                  onClick={() => useChat.getState().clear()}
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
              <ChatThread compact onConfigure={() => setAiModalOpen(true)} />
            )}
          </div>

          {/* Rodapé com Campo de Entrada */}
          <div className="border-t border-border/80 bg-card p-2.5">
            <form
              className="flex gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void ask(draft);
                setDraft('');
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
                  onClick={() => useChat.getState().stop()}
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
            <PrivacyNote className="mt-1.5 px-0.5" />
          </div>
        </div>
      )}

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </>
  );
}
