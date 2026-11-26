import { create } from 'zustand';

import { viewTransition } from '@/lib/view-transition';

/**
 * Preferências de exibição: tema, tamanho da letra, alto contraste, rolagem da faixa
 * de indicadores e redução de movimento.
 *
 * Aplicadas como classe/atributos no `<html>` (o CSS reage a eles) e guardadas no
 * `localStorage`. O script inline do `index.html` lê as mesmas chaves antes do primeiro
 * paint, para a página não abrir num tema e trocar para outro.
 */
export type Theme = 'dark' | 'light';
/** O que o usuário escolheu; `system` segue o tema do sistema operacional. */
export type ThemeMode = Theme | 'system';
export type FontScale = 'min' | 'med' | 'max';

interface PreferencesState {
  themeMode: ThemeMode;
  /** Tema em vigor (o do sistema, quando `themeMode` é `system`). */
  theme: Theme;
  fontScale: FontScale;
  highContrast: boolean;
  /** Faixa de indicadores sob o cabeçalho; oculta por padrão. */
  tickerVisible: boolean;
  /** Faixa de indicadores rolando sozinha; desligada, fica parada e rolável à mão. */
  tickerScroll: boolean;
  /** Menos movimento: animações, transições e o fundo animado param (como pede o sistema). */
  reduceMotion: boolean;
  setTheme: (mode: ThemeMode) => void;
  setFontScale: (scale: FontScale) => void;
  setHighContrast: (enabled: boolean) => void;
  setTickerScroll: (enabled: boolean) => void;
  setTickerVisible: (visible: boolean) => void;
  setReduceMotion: (enabled: boolean) => void;
}

const KEYS = {
  theme: 'simetrics-theme',
  font: 'simetrics-font',
  contrast: 'simetrics-contrast',
  ticker: 'simetrics-ticker',
  tickerVisible: 'simetrics-ticker-visible',
  motion: 'simetrics-motion',
} as const;

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Navegação privada ou armazenamento bloqueado: a preferência vale só nesta sessão.
  }
}

const systemLight = window.matchMedia('(prefers-color-scheme: light)');

function resolveTheme(mode: ThemeMode): Theme {
  if (mode === 'system') return systemLight.matches ? 'light' : 'dark';
  return mode;
}

function initial(): Pick<PreferencesState, 'themeMode' | 'theme' | 'fontScale' | 'highContrast' | 'tickerScroll' | 'tickerVisible' | 'reduceMotion'> {
  const font = read(KEYS.font);
  const storedTheme = read(KEYS.theme);
  // Escuro por padrão, como o Scientata; claro e sistema são escolha explícita.
  const themeMode: ThemeMode = storedTheme === 'light' || storedTheme === 'system' ? storedTheme : 'dark';
  return {
    themeMode,
    theme: resolveTheme(themeMode),
    fontScale: font === 'min' || font === 'max' ? font : 'med',
    highContrast: read(KEYS.contrast) === 'high',
    tickerScroll: read(KEYS.ticker) !== 'static',
    tickerVisible: read(KEYS.tickerVisible) === 'show',
    reduceMotion: read(KEYS.motion) === 'reduce',
  };
}

function apply({
  theme,
  fontScale,
  highContrast,
  reduceMotion,
}: Pick<PreferencesState, 'theme' | 'fontScale' | 'highContrast' | 'reduceMotion'>): void {
  const root = document.documentElement;
  root.classList.toggle('dark', theme === 'dark');
  root.dataset.font = fontScale;
  if (highContrast) root.dataset.contrast = 'high';
  else delete root.dataset.contrast;
  // O CSS trata `data-motion="reduce"` como `prefers-reduced-motion` (ver index.css).
  if (reduceMotion) root.dataset.motion = 'reduce';
  else delete root.dataset.motion;
}

/** O sistema pede menos movimento — então vale sempre, e a opção do app fica travada. */
export function systemReducesMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Menos movimento, pelo sistema ou pela opção do app — para o que o JS anima. */
export function prefersReducedMotion(): boolean {
  return usePreferences.getState().reduceMotion || systemReducesMotion();
}

export const usePreferences = create<PreferencesState>()((set, get) => ({
  ...initial(),

  setTheme(themeMode) {
    if (themeMode === get().themeMode) return;
    write(KEYS.theme, themeMode);
    const theme = resolveTheme(themeMode);
    if (theme === get().theme) {
      set({ themeMode });
      return;
    }
    // Cruzamento suave entre as paletas.
    viewTransition(() => {
      set({ themeMode, theme });
      apply(get());
    });
  },

  setFontScale(fontScale) {
    write(KEYS.font, fontScale);
    set({ fontScale });
    apply(get());
  },

  setHighContrast(highContrast) {
    write(KEYS.contrast, highContrast ? 'high' : 'normal');
    set({ highContrast });
    apply(get());
  },

  setTickerScroll(tickerScroll) {
    write(KEYS.ticker, tickerScroll ? 'scroll' : 'static');
    set({ tickerScroll });
  },

  setTickerVisible(tickerVisible) {
    write(KEYS.tickerVisible, tickerVisible ? 'show' : 'hide');
    set({ tickerVisible });
  },

  setReduceMotion(reduceMotion) {
    write(KEYS.motion, reduceMotion ? 'reduce' : 'full');
    set({ reduceMotion });
    apply(get());
  },
}));

apply(usePreferences.getState());

// No modo sistema, acompanha a troca de tema do sistema operacional com o app aberto.
systemLight.addEventListener('change', () => {
  const { themeMode, theme } = usePreferences.getState();
  const next = resolveTheme(themeMode);
  if (next === theme) return;
  usePreferences.setState({ theme: next });
  apply(usePreferences.getState());
});
