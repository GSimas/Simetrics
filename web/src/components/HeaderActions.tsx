import { Suspense, useState, type ReactNode } from 'react';
import { KeyRound, Settings } from 'lucide-react';

import { lazyWithPreload } from '@/lib/lazy';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { Locale } from '@/lib/i18n/translations';
import { cn } from '@/lib/utils';
import { useAiConfig } from '@/state/ai-config.store';
import { useLocale } from '@/state/locale.store';
import { systemReducesMotion, usePreferences, type FontScale, type Theme } from '@/state/preferences.store';

// Só baixado quando alguém abre as configurações de IA.
const AiSettingsModal = lazyWithPreload(() =>
  import('@/components/AiSettingsModal').then((module) => ({ default: module.AiSettingsModal })),
);

export const ICON_BUTTON =
  'inline-flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full border border-border text-foreground transition-colors hover:border-highlight hover:text-highlight focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring';

/** Grupo de opções mutuamente exclusivas, no traço dos controles do cabeçalho. */
function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex border border-border p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            'flex-1 cursor-pointer px-3 py-1.5 text-sm transition-colors',
            value === option.value
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** Interruptor liga/desliga na largura do campo, com o rótulo à esquerda. */
function Switch({
  label,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex w-full cursor-pointer items-center justify-between border border-border px-3 py-2 text-sm transition-colors hover:border-highlight disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:border-border"
    >
      <span>{label}</span>
      <span
        className={cn(
          'relative h-5 w-9 rounded-full border transition-colors',
          checked ? 'border-primary bg-primary' : 'border-border bg-muted',
        )}
      >
        <span
          className={cn(
            'absolute top-0.5 size-3.5 rounded-full transition-[left,background-color] duration-200',
            checked ? 'left-[18px] bg-primary-foreground' : 'left-0.5 bg-muted-foreground',
          )}
        />
      </span>
    </button>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <p className="eyebrow">{label}</p>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/**
 * Botão de engrenagem com as preferências do app: tema, idioma, tamanho da letra, alto
 * contraste, rolagem da faixa de indicadores e a chave de IA. Substitui os botões
 * soltos que ocupavam o cabeçalho.
 */
export function SettingsButton() {
  const { t, locale, setLocale } = useLocale();
  const theme = usePreferences((state) => state.theme);
  const fontScale = usePreferences((state) => state.fontScale);
  const highContrast = usePreferences((state) => state.highContrast);
  const tickerScroll = usePreferences((state) => state.tickerScroll);
  const reduceMotion = usePreferences((state) => state.reduceMotion);
  const { setTheme, setFontScale, setHighContrast, setTickerScroll, setReduceMotion } = usePreferences.getState();
  // Pedido pelo sistema, vale sempre: a opção aparece ligada e travada.
  const systemMotion = systemReducesMotion();
  const isAiConfigured = useAiConfig((state) => state.isConfigured());

  const [open, setOpen] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  // Montado desde a primeira abertura, para a animação de fechar continuar funcionando.
  const [aiMounted, setAiMounted] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={ICON_BUTTON}
        data-tour="settings"
        title={t('settings_btn')}
        aria-label={t('settings_btn')}
      >
        <Settings className="size-4" aria-hidden />
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[90dvh] max-w-md overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t('settings_title')}</DialogTitle>
            <DialogDescription>{t('settings_desc')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-5">
            <Field label={t('settings_theme')}>
              <Segmented<Theme>
                label={t('settings_theme')}
                value={theme}
                onChange={setTheme}
                options={[
                  { value: 'dark', label: t('settings_theme_dark') },
                  { value: 'light', label: t('settings_theme_light') },
                ]}
              />
            </Field>

            <Field label={t('settings_language')}>
              <Segmented<Locale>
                label={t('settings_language')}
                value={locale}
                onChange={setLocale}
                options={[
                  { value: 'pt', label: 'Português' },
                  { value: 'en', label: 'English' },
                ]}
              />
            </Field>

            <Field label={t('settings_font')}>
              <Segmented<FontScale>
                label={t('settings_font')}
                value={fontScale}
                onChange={setFontScale}
                options={[
                  { value: 'min', label: t('settings_font_min') },
                  { value: 'med', label: t('settings_font_med') },
                  { value: 'max', label: t('settings_font_max') },
                ]}
              />
            </Field>

            <Field label={t('settings_contrast')} hint={t('settings_contrast_desc')}>
              <Switch label={t('settings_contrast')} checked={highContrast} onChange={setHighContrast} />
            </Field>

            <Field
              label={t('settings_motion')}
              hint={systemMotion ? t('settings_motion_system') : t('settings_motion_desc')}
            >
              <Switch
                label={t('settings_motion_reduce')}
                checked={reduceMotion || systemMotion}
                onChange={setReduceMotion}
                disabled={systemMotion}
              />
            </Field>

            <Field label={t('settings_ticker')} hint={t('settings_ticker_desc')}>
              <Switch label={t('settings_ticker_scroll')} checked={tickerScroll} onChange={setTickerScroll} />
            </Field>

            <Field label={t('settings_ai')}>
              <div className="flex items-center justify-between gap-3 border border-border px-3 py-2">
                <span className="flex items-center gap-2 text-sm">
                  <span
                    className={cn(
                      'size-1.5 rounded-full',
                      isAiConfigured ? 'bg-highlight shadow-[0_0_8px_var(--highlight)]' : 'bg-muted-foreground',
                    )}
                  />
                  {isAiConfigured ? t('ai_configured') : t('settings_ai_missing')}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5"
                  onClick={() => {
                    setAiMounted(true);
                    setAiOpen(true);
                  }}
                  onPointerEnter={() => void AiSettingsModal.preload()}
                >
                  <KeyRound className="size-3.5" aria-hidden />
                  {t('settings_ai_configure')}
                </Button>
              </div>
            </Field>
          </div>
        </DialogContent>
      </Dialog>

      {aiMounted && (
        <Suspense fallback={null}>
          <AiSettingsModal open={aiOpen} onOpenChange={setAiOpen} />
        </Suspense>
      )}
    </>
  );
}
