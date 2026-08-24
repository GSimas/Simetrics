import { useCallback, useMemo } from 'react';
import { create } from 'zustand';
import { TRANSLATIONS, type Locale, type TranslationKey } from '@/lib/i18n/translations';

interface LocaleStoreState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
}

export interface LocaleHookState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
  t: (key: TranslationKey) => string;
}

function getInitialLocale(): Locale {
  if (typeof window === 'undefined') return 'pt';
  const saved = localStorage.getItem('simetrics-locale') as Locale | null;
  if (saved === 'pt' || saved === 'en') return saved;
  const browserLang = (navigator.language || (navigator.languages && navigator.languages[0]) || '').toLowerCase();
  return browserLang.startsWith('pt') ? 'pt' : 'en';
}

export const useLocaleStore = create<LocaleStoreState>((set, get) => ({
  locale: getInitialLocale(),
  setLocale: (locale: Locale) => {
    localStorage.setItem('simetrics-locale', locale);
    set({ locale });
  },
  toggleLocale: () => {
    const next: Locale = get().locale === 'pt' ? 'en' : 'pt';
    localStorage.setItem('simetrics-locale', next);
    set({ locale: next });
  },
}));

/**
 * Hook reativo para internacionalização.
 * Sempre que `locale` mudar, todos os componentes que utilizam `useLocale`
 * receberão um novo `t` e re-renderizarão instantaneamente.
 */
export function useLocale<T = LocaleHookState>(
  selector?: (state: LocaleHookState) => T,
): T {
  const locale = useLocaleStore((state) => state.locale);
  const setLocale = useLocaleStore((state) => state.setLocale);
  const toggleLocale = useLocaleStore((state) => state.toggleLocale);

  const t = useCallback(
    (key: TranslationKey): string => {
      return TRANSLATIONS[locale]?.[key] ?? TRANSLATIONS.pt[key] ?? key;
    },
    [locale],
  );

  const state = useMemo(
    () => ({ locale, setLocale, toggleLocale, t }),
    [locale, setLocale, toggleLocale, t],
  );

  return selector ? selector(state) : (state as unknown as T);
}

useLocale.getState = (): LocaleHookState => {
  const { locale, setLocale, toggleLocale } = useLocaleStore.getState();
  const t = (key: TranslationKey) => TRANSLATIONS[locale]?.[key] ?? TRANSLATIONS.pt[key] ?? key;
  return { locale, setLocale, toggleLocale, t };
};
