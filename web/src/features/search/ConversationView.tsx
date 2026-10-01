import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, KeyRound, Send, Sparkles, Square, Trash2 } from 'lucide-react';

import { AiSettingsModal } from '@/components/AiSettingsModal';
import { Button } from '@/components/ui/button';
import { ChatThread, PrivacyNote } from '@/features/chat/ChatThread';
import { useChat } from '@/state/chat.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';

/**
 * Conversa com a base em tela cheia, dentro do Motor de Busca. É a mesma conversa do
 * widget da Simi (store compartilhado): trocar de tela não a perde.
 */
export function ConversationView() {
  const t = useLocale((state) => state.t);
  const messages = useChat((state) => state.messages);
  const streaming = useChat((state) => state.streaming);
  const status = useChat((state) => state.status);
  const ask = useChat((state) => state.ask);
  const [draft, setDraft] = useState('');
  const [aiModalOpen, setAiModalOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // A página acompanha a resposta enquanto ela chega.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messages, status]);

  useEffect(() => inputRef.current?.focus(), []);

  const send = (): void => {
    if (!draft.trim() || streaming) return;
    void ask(draft);
    setDraft('');
  };

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => useNavigation.setState({ chatOpen: false })}
          className="eyebrow inline-flex cursor-pointer items-center gap-1.5 transition-colors hover:text-highlight"
        >
          <ArrowLeft className="size-3.5" aria-hidden />
          {t('chat_back_search')}
        </button>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setAiModalOpen(true)}
            title={t('ai_settings_btn')}
            className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <KeyRound className="size-4" aria-hidden />
          </button>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => useChat.getState().clear()}
              title={t('chat_clear')}
              className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
            >
              <Trash2 className="size-4" aria-hidden />
            </button>
          )}
        </div>
      </div>

      <h3 className="flex items-center gap-2 text-2xl font-medium tracking-[-0.04em]">
        <Sparkles className="size-5 text-highlight" aria-hidden />
        {t('chat_view_title')}
      </h3>

      {/* role="log": leitores de tela anunciam mensagens novas; aria-busy segura o anúncio
          até a resposta terminar de chegar, em vez de ler trecho por trecho. */}
      <div role="log" aria-live="polite" aria-busy={streaming} aria-label={t('chat_title')} className="space-y-4">
        <ChatThread onConfigure={() => setAiModalOpen(true)} />
        <div ref={endRef} />
      </div>

      {/* pb-14: a faixa de baixo fica para o botão flutuante do café, sem cobrir o campo. */}
      <div className="sticky bottom-0 -mx-2 bg-background/90 px-2 pb-14 pt-2 backdrop-blur-md">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            send();
          }}
          className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 transition-[border-color,box-shadow] duration-200 focus-within:border-highlight focus-within:shadow-[0_0_32px_-12px_var(--highlight)]"
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter envia; Shift+Enter quebra a linha.
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            placeholder={t('chat_placeholder')}
            aria-label={t('chat_placeholder')}
            className="field-sizing-content max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-hidden placeholder:text-muted-foreground"
          />
          {streaming ? (
            <Button type="button" variant="outline" size="sm" onClick={() => useChat.getState().stop()} className="h-10 gap-1">
              <Square className="size-3.5" aria-hidden />
              {t('chat_btn_stop')}
            </Button>
          ) : (
            <Button type="submit" variant="gradient" size="sm" disabled={!draft.trim()} title={t('chat_send')} className="h-10 px-3">
              <Send className="size-4" aria-hidden />
              <span className="sr-only">{t('chat_send')}</span>
            </Button>
          )}
        </form>
        <PrivacyNote className="mt-1.5 px-2" />
      </div>

      <AiSettingsModal open={aiModalOpen} onOpenChange={setAiModalOpen} />
    </div>
  );
}
