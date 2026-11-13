import { fitText, measureText } from './text';

/** Legenda desenhada dentro do SVG — assim ela também sai na imagem exportada. */
export interface LegendItem {
  key: string;
  label: string;
  color: string;
  /** Traço (linha) em vez de quadrado. */
  dash?: string;
  line?: boolean;
  muted?: boolean;
}

export const LEGEND_FONT = 12;
const LEGEND_ROW = 20;

export interface LegendLayout {
  height: number;
  /** Tamanho da letra já com o fator das Configurações. */
  font: number;
  items: { item: LegendItem; x: number; y: number; width: number; text: string }[];
}

/** Distribui a legenda em linhas que cabem em `width`. */
export function layoutLegend(items: readonly LegendItem[], width: number, scale = 1): LegendLayout {
  const font = LEGEND_FONT * scale;
  const rowHeight = LEGEND_ROW * scale;
  if (items.length === 0) return { height: 0, font, items: [] };
  const placed: LegendLayout['items'] = [];
  let x = 0;
  let row = 0;
  const maxLabel = Math.max(80, width * 0.45);
  for (const item of items) {
    const text = fitText(item.label, maxLabel, font);
    const itemWidth = 18 + measureText(text, font) + 16;
    if (x > 0 && x + itemWidth > width) {
      x = 0;
      row += 1;
    }
    placed.push({ item, x, y: row * rowHeight, width: itemWidth - 16, text });
    x += itemWidth;
  }
  return { height: (row + 1) * rowHeight + 6, font, items: placed };
}
