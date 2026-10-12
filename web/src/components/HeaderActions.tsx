import { Suspense, useEffect, useRef, useState, type ReactNode } from 'react';
import { KeyRound, Moon, Settings, Sun } from 'lucide-react';

import { lazyWithPreload } from '@/lib/lazy';
import { Button } from '@/components/ui/button';
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
  /** Com `icon`, o botão mostra só o ícone; o rótulo vira nome acessível e dica. */
  options: readonly { value: T; label: string; icon?: ReactNode }[];
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
          aria-label={option.icon ? option.label : undefined}
          title={option.icon ? option.label : undefined}
          onClick={() => onChange(option.value)}
          className={cn(
            'flex h-9 flex-1 cursor-pointer items-center justify-center px-3 text-sm transition-colors',
            value === option.value
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {option.icon ?? option.label}
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

/** Sigla do idioma no traço mono dos rótulos — um ícone de bandeira trocaria idioma por país. */
function LangCode({ code }: { code: string }) {
  return (
    <span className="font-mono text-xs tracking-wider" aria-hidden>
      {code}
    </span>
  );
}

/** "A" no tamanho que a opção aplica: o próprio ícone mostra a diferença. */
function FontGlyph({ className }: { className: string }) {
  return (
    <span className={cn('font-semibold leading-none', className)} aria-hidden>
      A
    </span>
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

  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Dropdown: fecha com clique fora ou Esc (e devolve o foco à engrenagem).
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={cn(ICON_BUTTON, 'group', open && 'border-highlight text-highlight')}
        data-tour="settings"
        title={t('settings_btn')}
        aria-label={t('settings_btn')}
        aria-haspopup="dialog"
        aria-expanded={open}
      >
        <Settings
          className={cn('size-4 transition-transform duration-500 group-hover:rotate-90', open && 'rotate-90')}
          aria-hidden
        />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label={t('settings_title')}
          className="absolute right-0 top-full z-50 mt-2 max-h-[calc(100dvh-6rem)] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto border border-border bg-card p-5 text-left shadow-xl duration-200 animate-in fade-in-0 slide-in-from-top-1"
        >
          <div className="mb-4 space-y-1">
            <p className="text-base font-semibold">{t('settings_title')}</p>
            <p className="text-xs text-muted-foreground">{t('settings_desc')}</p>
          </div>

          <div className="space-y-5">
            <Field label={t('settings_theme')}>
              <Segmented<Theme>
                label={t('settings_theme')}
                value={theme}
                onChange={setTheme}
                options={[
                  { value: 'dark', label: t('settings_theme_dark'), icon: <Moon className="size-4" aria-hidden /> },
                  { value: 'light', label: t('settings_theme_light'), icon: <Sun className="size-4" aria-hidden /> },
                ]}
              />
            </Field>

            <Field label={t('settings_language')}>
              <Segmented<Locale>
                label={t('settings_language')}
                value={locale}
                onChange={setLocale}
                options={[
                  { value: 'pt', label: 'Português', icon: <LangCode code="PT" /> },
                  { value: 'en', label: 'English', icon: <LangCode code="EN" /> },
                ]}
              />
            </Field>

            <Field label={t('settings_font')}>
              <Segmented<FontScale>
                label={t('settings_font')}
                value={fontScale}
                onChange={setFontScale}
                options={[
                  { value: 'min', label: t('settings_font_min'), icon: <FontGlyph className="text-xs" /> },
                  { value: 'med', label: t('settings_font_med'), icon: <FontGlyph className="text-base" /> },
                  { value: 'max', label: t('settings_font_max'), icon: <FontGlyph className="text-xl" /> },
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
                    setOpen(false);
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
        </div>
      )}

      {aiMounted && (
        <Suspense fallback={null}>
          <AiSettingsModal open={aiOpen} onOpenChange={setAiOpen} />
        </Suspense>
      )}
    </div>
  );
}
