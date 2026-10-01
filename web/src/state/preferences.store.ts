import { flushSync } from 'react-dom';
import { create } from 'zustand';

/**
 * Preferências de exibição: tema, tamanho da letra, alto contraste, rolagem da faixa
 * de indicadores e redução de movimento.
 *
 * Aplicadas como classe/atributos no `<html>` (o CSS reage a eles) e guardadas no
 * `localStorage`. O script inline do `index.html` lê as mesmas chaves antes do primeiro
 * paint, para a página não abrir num tema e trocar para outro.
 */
export type Theme = 'dark' | 'light';
export type FontScale = 'min' | 'med' | 'max';

interface PreferencesState {
  theme: Theme;
  fontScale: FontScale;
  highContrast: boolean;
  /** Faixa de indicadores rolando sozinha; desligada, fica parada e rolável à mão. */
  tickerScroll: boolean;
  /** Menos movimento: animações, transições e o fundo animado param (como pede o sistema). */
  reduceMotion: boolean;
  setTheme: (theme: Theme) => void;
  setFontScale: (scale: FontScale) => void;
  setHighContrast: (enabled: boolean) => void;
  setTickerScroll: (enabled: boolean) => void;
  setReduceMotion: (enabled: boolean) => void;
}

const KEYS = {
  theme: 'simetrics-theme',
  font: 'simetrics-font',
  contrast: 'simetrics-contrast',
  ticker: 'simetrics-ticker',
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

function initial(): Pick<PreferencesState, 'theme' | 'fontScale' | 'highContrast' | 'tickerScroll' | 'reduceMotion'> {
  const font = read(KEYS.font);
  return {
    // Escuro por padrão, como o Scientata; o claro ("papel") é escolha explícita.
    theme: read(KEYS.theme) === 'light' ? 'light' : 'dark',
    fontScale: font === 'min' || font === 'max' ? font : 'med',
    highContrast: read(KEYS.contrast) === 'high',
    tickerScroll: read(KEYS.ticker) !== 'static',
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

  setTheme(theme) {
    if (theme === get().theme) return;
    write(KEYS.theme, theme);
    const update = (): void => {
      // `flushSync` garante que o "depois" capturado pela View Transition já tem a
      // paleta nova — inclusive nos componentes que leem o tema do store.
      flushSync(() => set({ theme }));
      apply(get());
    };
    // Cruzamento suave entre as paletas; sem suporte (ou com movimento reduzido), troca
    // direto.
    if (document.startViewTransition && !prefersReducedMotion()) document.startViewTransition(update);
    else update();
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

  setReduceMotion(reduceMotion) {
    write(KEYS.motion, reduceMotion ? 'reduce' : 'full');
    set({ reduceMotion });
    apply(get());
  },
}));

apply(usePreferences.getState());
