import { FONT_SANS, measureText } from '@/components/charts/svg/text';
import { chartToPngDataUrl, imageFromSvgElement, type ChartImage } from '@/lib/export-image';

import type { ReportChartId, ReportImages } from './report-catalog';

/**
 * Captura dos gráficos da prévia para o documento.
 *
 * A prévia desenha os gráficos com os mesmos componentes do app, e o arquivo leva
 * exatamente o que está na tela: o SVG de cada figura, com as cores da folha clara,
 * rasterizado em PNG. Cada figura é um `[data-report-chart]` com `data-state`
 * (`loading` | `ready` | `empty`) e, quando a legenda é HTML fora do SVG, `data-legend`.
 */

export interface LegendItem {
  label: string;
  color: string;
}

const CHART_SVG = 'svg[role="img"], svg[role="figure"]';
const SHAPES = 'path, circle, rect, line, text, polygon, polyline';
const escapeXml = (text: string): string =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Acrescenta a legenda embaixo do gráfico, em linhas que quebram pela largura. */
function withLegend(image: ChartImage, items: readonly LegendItem[], textColor: string): ChartImage {
  if (items.length === 0) return image;
  const pad = 16;
  const rowHeight = 22;
  const rows: { x: number; item: LegendItem }[][] = [[]];
  let x = pad;
  for (const item of items) {
    const width = 18 + measureText(item.label, 12, FONT_SANS) + 18;
    if (x + width > image.width - pad && rows.at(-1)!.length > 0) {
      rows.push([]);
      x = pad;
    }
    rows.at(-1)!.push({ x, item });
    x += width;
  }
  const legendHeight = rows.length * rowHeight + pad;
  const parts = rows.flatMap((row, index) =>
    row.map(({ x: left, item }) => {
      const y = image.height + pad / 2 + index * rowHeight + rowHeight / 2;
      return (
        `<circle cx="${left + 5}" cy="${y}" r="5" fill="${item.color}"/>` +
        `<text x="${left + 16}" y="${y + 4}" font-size="12" fill="${textColor}">${escapeXml(item.label)}</text>`
      );
    }),
  );
  const height = image.height + legendHeight;
  return {
    svg:
      `<svg xmlns="http://www.w3.org/2000/svg" width="${image.width}" height="${height}" viewBox="0 0 ${image.width} ${height}" font-family="${FONT_SANS}">` +
      image.svg.replace(/^<\?xml[^>]*>/, '') +
      parts.join('') +
      '</svg>',
    width: image.width,
    height,
  };
}

function isDrawn(figure: HTMLElement): boolean {
  const state = figure.dataset['state'];
  if (state === 'empty') return true;
  if (state !== 'ready') return false;
  const img = figure.querySelector<HTMLImageElement>('img[data-report-img]');
  if (img) return img.complete && img.naturalWidth > 0;
  // A nuvem de palavras e o grafo desenham depois do layout assíncrono: pronto é ter forma.
  return figure.querySelector(CHART_SVG)?.querySelector(SHAPES) != null;
}

async function waitDrawn(figures: HTMLElement[], timeoutMs: number): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (figures.some((figure) => !isDrawn(figure)) && performance.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
}

/**
 * Imagens dos gráficos pedidos, na folha `root`. Gráfico sem dados fica de fora; o que
 * não terminar de desenhar a tempo volta em `missing`.
 */
export async function captureReportCharts(
  root: HTMLElement,
  ids: readonly ReportChartId[],
  timeoutMs = 20_000,
): Promise<{ images: ReportImages; missing: ReportChartId[] }> {
  const figures = ids
    .map((id) => root.querySelector<HTMLElement>(`[data-report-chart="${id}"]`))
    .filter((figure): figure is HTMLElement => figure !== null);
  await waitDrawn(figures, timeoutMs);

  const textColor = getComputedStyle(root).getPropertyValue('--foreground').trim() || '#07110f';
  const images: ReportImages = {};
  const missing: ReportChartId[] = [];

  for (const figure of figures) {
    const id = figure.dataset['reportChart'] as ReportChartId;
    if (figure.dataset['state'] === 'empty') continue;
    if (!isDrawn(figure)) {
      missing.push(id);
      continue;
    }
    const img = figure.querySelector<HTMLImageElement>('img[data-report-img]');
    if (img) {
      images[id] = { dataUrl: img.src, width: img.naturalWidth, height: img.naturalHeight };
      continue;
    }
    const svg = figure.querySelector<SVGSVGElement>(CHART_SVG);
    if (!svg) {
      missing.push(id);
      continue;
    }
    const legend = figure.dataset['legend'] ? (JSON.parse(figure.dataset['legend']) as LegendItem[]) : [];
    const image = withLegend(imageFromSvgElement(svg), legend, textColor);
    images[id] = { dataUrl: await chartToPngDataUrl(image), width: image.width, height: image.height };
  }

  return { images, missing };
}
