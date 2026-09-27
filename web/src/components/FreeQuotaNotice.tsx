import { Gift, KeyRound, LogIn } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { startOpenRouterLogin } from '@/lib/openrouter-oauth';
import { cn } from '@/lib/utils';
import { useAiConfig } from '@/state/ai-config.store';
import { useFreeTier } from '@/state/free-tier.store';
import { useLocale } from '@/state/locale.store';

interface FreeQuotaNoticeProps {
  onConfigure: () => void;
  compact?: boolean;
}

/**
 * Aviso do Simi para quem não tem chave própria: quantas perguntas gratuitas restam neste
 * dispositivo, que acabaram, ou — sem chave no servidor — que é preciso configurar uma.
 */
export function FreeQuotaNotice({ onConfigure, compact = false }: FreeQuotaNoticeProps) {
  const t = useLocale((state) => state.t);
  const isAiConfigured = useAiConfig((state) => state.isConfigured());
  const status = useFreeTier((state) => state.status);

  if (isAiConfigured) return null;

  const free = status?.deepseek.available && status.simi.limit > 0 ? status.simi : null;
  const exhausted = free !== null && free.remaining === 0;
  const message = !free
    ? t('chat_no_key_warning')
    : exhausted
      ? t('chat_free_exhausted').replace('{limit}', String(free.limit))
      : t('chat_free_remaining').replace('{remaining}', String(free.remaining)).replace('{limit}', String(free.limit));
  const Icon = free && !exhausted ? Gift : KeyRound;

  // Login com OpenRouter em destaque: é o caminho mais curto para usar a Simi sem colar chave.
  const actions = (
    <div className={cn('flex flex-wrap items-center gap-2', compact && 'mt-2')}>
      <Button
        variant="ai"
        size="sm"
        onClick={() => void startOpenRouterLogin()}
        title={t('ai_openrouter_free_hint')}
        className={cn('font-bold', compact ? 'h-7 text-[11px]' : 'h-8 text-xs')}
      >
        <LogIn className="size-3.5" aria-hidden />
        {t('ai_openrouter_login')}
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={onConfigure}
        className={cn('font-semibold', compact ? 'h-7 text-[10px]' : 'h-8 text-xs')}
      >
        <KeyRound className="size-3.5" aria-hidden />
        {t('ai_settings_btn')}
      </Button>
    </div>
  );

  return (
    <div
      role={exhausted ? 'alert' : 'status'}
      className={cn(
        'rounded-xl border border-purple-200 bg-purple-50/70 text-purple-900 dark:border-purple-900 dark:bg-purple-950/40 dark:text-purple-300',
        compact ? 'p-2.5 text-[11px]' : 'flex flex-wrap items-center justify-between gap-2 p-3 text-xs',
      )}
    >
      <div className="flex items-start gap-2">
        <Icon className="mt-0.5 size-4 shrink-0 text-purple-600" aria-hidden />
        <div className="flex-1">
          <p className="font-medium leading-snug">{message}</p>
          {compact && actions}
        </div>
      </div>
      {!compact && actions}
    </div>
  );
}
