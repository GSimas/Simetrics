/** Fontes e medida de texto dos gráficos SVG. */

import { usePreferences, type FontScale } from '@/state/preferences.store';

/** Raiz de 14/16/18px do `index.css` (`html[data-font]`), relativa aos 16px de base. */
const FONT_SCALE: Record<FontScale, number> = { min: 14 / 16, med: 1, max: 18 / 16 };

/**
 * Fator do tamanho da letra das Configurações para os textos dos gráficos. Eles medem em px
 * (cortes, colisões, legendas), então o fator entra no tamanho e na medida.
 */
export function useFontScale(): number {
  return FONT_SCALE[usePreferences((state) => state.fontScale)];
}

export const FONT_SANS = 'Manrope, system-ui, sans-serif';
export const FONT_MONO = '"DM Mono", ui-monospace, monospace';

let measureContext: CanvasRenderingContext2D | null = null;

/** Largura do texto em pixels, medida num canvas com a fonte do gráfico. */
export function measureText(text: string, fontSize: number, font = FONT_SANS, weight = 500): number {
  if (typeof document === 'undefined') return text.length * fontSize * 0.56;
  measureContext ??= document.createElement('canvas').getContext('2d');
  if (!measureContext) return text.length * fontSize * 0.56;
  measureContext.font = `${weight} ${fontSize}px ${font}`;
  return measureContext.measureText(text).width;
}

/** Corta o texto com reticências para caber em `maxWidth` pixels. */
export function fitText(text: string, maxWidth: number, fontSize: number, font = FONT_SANS): string {
  if (measureText(text, fontSize, font) <= maxWidth) return text;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (measureText(`${text.slice(0, mid)}…`, fontSize, font) <= maxWidth) low = mid;
    else high = mid - 1;
  }
  return low > 0 ? `${text.slice(0, low).trimEnd()}…` : '…';
}
