import { Gift, KeyRound } from 'lucide-react';

import { Button } from '@/components/ui/button';
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
          {compact && (
            <Button variant="ai" size="sm" onClick={onConfigure} className="mt-2 h-6 text-[10px] font-bold">
              {t('ai_settings_btn')}
            </Button>
          )}
        </div>
      </div>
      {!compact && (
        <Button variant="ai" size="sm" onClick={onConfigure} className="h-7 text-xs font-bold">
          {t('ai_settings_btn')}
        </Button>
      )}
    </div>
  );
}
