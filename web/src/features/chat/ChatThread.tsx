import { memo, useMemo } from 'react';
import { Bot, Database, FileText, User } from 'lucide-react';

import { FreeQuotaNotice } from '@/components/FreeQuotaNotice';
import { MarkdownContent } from '@/components/MarkdownContent';
import { cn } from '@/lib/utils';
import { useAiConfig } from '@/state/ai-config.store';
import { CONTEXT_SIZE, useChat, type ChatMessage } from '@/state/chat.store';
import { useLocale } from '@/state/locale.store';
import { openDossier } from '@/state/navigation.store';

import { useEntityLinks } from './entity-text';

/**
 * Markdown memoizado: durante o streaming o chat re-renderiza a cada trecho, e sem isso
 * todas as respostas anteriores eram re-parseadas a cada vez (custo quadrático na conversa).
 */
const MessageMarkdown = memo(MarkdownContent);

/** Três pontos pulsando + rótulo: "pensando" antes do texto chegar, "escrevendo" durante. */
function TypingIndicator({ label, className }: { label: string; className?: string | undefined }) {
  return (
    <span className={cn('flex items-center gap-2 text-[11px] text-muted-foreground', className)}>
      <span className="flex items-center gap-1" aria-hidden>
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="size-1.5 animate-bounce rounded-full bg-emerald-500"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
      {label}
    </span>
  );
}

/** Os documentos que foram ao modelo — a resposta se apoia neles; cada um abre o dossiê. */
function Sources({ titles, compact }: { titles: string[]; compact: boolean }) {
  const t = useLocale((state) => state.t);
  return (
    <details className="group mt-2.5 border-t border-border/70 pt-2">
      <summary className="eyebrow cursor-pointer list-none text-[10px] transition-colors hover:text-highlight">
        <FileText className="mr-1.5 inline size-3" aria-hidden />
        {t('chat_sources').replace('{count}', String(titles.length))}
      </summary>
      <ol className={cn('mt-2 list-decimal space-y-1 pl-5 text-muted-foreground', compact ? 'text-[11px]' : 'text-xs')}>
        {titles.map((title) => (
          <li key={title}>
            <button
              type="button"
              onClick={() => openDossier('Documento', title)}
              className="cursor-pointer text-left underline-offset-2 transition-colors hover:text-highlight hover:underline"
            >
              {title}
            </button>
          </li>
        ))}
      </ol>
    </details>
  );
}

/**
 * A conversa com a Simi, lida do store compartilhado. `compact` é o desenho do widget
 * flutuante; sem ele, a tela de conversa do Motor de Busca.
 */
export function ChatThread({ compact = false, onConfigure }: { compact?: boolean; onConfigure: () => void }) {
  const t = useLocale((state) => state.t);
  const messages = useChat((state) => state.messages);
  const streaming = useChat((state) => state.streaming);
  const status = useChat((state) => state.status);
  const error = useChat((state) => state.error);
  const ask = useChat((state) => state.ask);
  const renderText = useEntityLinks();

  const displayed: ChatMessage[] = useMemo(
    () => [{ role: 'assistant', content: t('chat_greeting') }, ...messages],
    [messages, t],
  );
  const suggestions = [t('chat_sugg_1'), t('chat_sugg_2'), t('chat_sugg_3'), t('chat_sugg_4')];
  const text = compact ? 'text-xs' : 'text-sm';

  return (
    <>
      <FreeQuotaNotice compact={compact} onConfigure={onConfigure} />

      {displayed.map((message, index) => {
        const isUser = message.role === 'user';
        const isLast = index === displayed.length - 1;
        return (
          <div key={index} className={cn('flex gap-2', isUser ? 'flex-row-reverse' : 'flex-row')}>
            <div
              className={cn(
                'grid shrink-0 place-items-center rounded-full font-semibold shadow-2xs',
                compact ? 'size-6.5 text-[10px]' : 'size-8 text-xs',
                isUser ? 'bg-primary text-primary-foreground' : 'bg-emerald-600 text-white',
              )}
            >
              {isUser ? <User className="size-3.5" aria-hidden /> : <Bot className="size-3.5" aria-hidden />}
            </div>

            <div
              className={cn(
                'max-w-[88%] rounded-xl px-3.5 py-2.5 leading-relaxed shadow-2xs',
                text,
                isUser ? 'bg-primary font-medium text-primary-foreground' : 'border border-border/80 bg-card text-foreground',
              )}
            >
              {!isUser && message.toolsExecuted && message.toolsExecuted.length > 0 && (
                <div className="mb-2 flex items-center gap-1.5 rounded-md border border-emerald-200/80 bg-emerald-50/70 px-2 py-0.5 text-[10px] font-medium text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
                  <Database className="size-2.5 text-emerald-600 dark:text-emerald-400" />
                  <span>
                    {message.toolsExecuted.length === 1
                      ? t('chat_tool_executed')
                      : `${message.toolsExecuted.length} ${t('chat_tools_executed_count')}`}
                  </span>
                </div>
              )}

              {isUser ? (
                <p className="whitespace-pre-wrap">{message.content}</p>
              ) : (
                <>
                  {/* trim: modelos com raciocínio costumam abrir com quebras de linha,
                      que o Markdown não desenha — o balão ficava em branco. */}
                  {message.content.trim() && <MessageMarkdown content={message.content} renderText={renderText} />}
                  {streaming && isLast && (
                    <TypingIndicator
                      label={message.content.trim() ? t('chat_writing') : status || t('chat_thinking')}
                      className={message.content.trim() ? 'mt-2' : undefined}
                    />
                  )}
                  {message.sources && message.sources.length > 0 && !(streaming && isLast) && (
                    <Sources titles={message.sources} compact={compact} />
                  )}
                </>
              )}
            </div>
          </div>
        );
      })}

      {messages.length === 0 && (
        <div className="mt-2 space-y-1.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            {t('chat_suggestions_label')}
          </p>
          <div className={cn('flex gap-1.5', compact ? 'flex-col' : 'flex-wrap')}>
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className={cn(
                  'rounded-lg border border-border/80 bg-card/80 px-2.5 py-1.5 text-left font-medium text-foreground transition-all hover:border-emerald-400 hover:bg-emerald-50/50 hover:text-emerald-950 dark:hover:bg-emerald-950/40 dark:hover:text-emerald-300',
                  compact ? 'text-[11px]' : 'text-xs',
                )}
                onClick={() => void ask(suggestion)}
              >
                💡 {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 p-2 text-[11px] text-destructive">
          {error}
        </p>
      )}
    </>
  );
}

/** O que sai do navegador ao perguntar — dito junto do campo, sempre à vista. */
export function PrivacyNote({ className }: { className?: string }) {
  const t = useLocale((state) => state.t);
  const config = useAiConfig((state) => state.config);
  const isConfigured = useAiConfig((state) => state.isConfigured());
  const provider = isConfigured ? config.provider.toUpperCase() : t('chat_privacy_free_provider');
  return (
    <p className={cn('text-[10px] leading-snug text-muted-foreground', className)}>
      {t('chat_privacy_note').replace('{count}', String(CONTEXT_SIZE)).replace('{provider}', provider)}
    </p>
  );
}
