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

/** Sem escolha salva, segue o idioma do sistema: o primeiro suportado na lista de preferência. */
function getInitialLocale(): Locale {
  if (typeof window === 'undefined') return 'pt';
  try {
    const saved = localStorage.getItem('simetrics-locale');
    if (saved === 'pt' || saved === 'en') return saved;
  } catch {
    // Armazenamento bloqueado (aba privada): cai no idioma do sistema.
  }
  const preferred = navigator.languages?.length ? navigator.languages : [navigator.language];
  for (const lang of preferred) {
    const code = lang?.toLowerCase().slice(0, 2);
    if (code === 'pt' || code === 'en') return code;
  }
  return 'en';
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

// Título da aba e `lang` do documento acompanham o idioma (o index.html vem em português).
const PAGE_TITLE: Record<Locale, string> = {
  pt: 'Simetrics — Análise Bibliométrica e Cientométrica',
  en: 'Simetrics — Bibliometric and Scientometric Analysis',
};

function syncDocument(locale: Locale): void {
  if (typeof document === 'undefined') return;
  document.title = PAGE_TITLE[locale];
  document.documentElement.lang = locale === 'en' ? 'en' : 'pt-BR';
}

syncDocument(useLocaleStore.getState().locale);
useLocaleStore.subscribe((state) => syncDocument(state.locale));
