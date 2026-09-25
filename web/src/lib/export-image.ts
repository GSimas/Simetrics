import { downloadBlob, timestampedFilename } from '@/core/export';
import { useLocale } from '@/state/locale.store';

function fail(pt: string, en: string): Error {
  return new Error(useLocale.getState().locale === 'en' ? en : pt);
}

/**
 * Exportação de gráficos como imagem: SVG (vetorial), JPG (com o fundo do tema) ou PNG
 * (fundo transparente). Todo gráfico entrega um SVG — desenhado pelo próprio Simetrics
 * ou montado a partir do grafo (Sigma) — e os formatos raster saem dele por um canvas.
 */
export type ImageFormat = 'svg' | 'jpg' | 'png';

export interface ChartImage {
  svg: string;
  width: number;
  height: number;
}

/** Escala do raster: 2× para a imagem não sair borrada em telas de alta densidade. */
const RASTER_SCALE = 2;

function cssVariable(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Troca `var(--token)` pela cor resolvida: fora da página as variáveis não existem. */
export function resolveCssVariables(svg: string): string {
  return svg.replace(/var\((--[\w-]+)\)/g, (match, name: string) => cssVariable(name) || match);
}

/** SVG de um elemento da página, com tamanho explícito e cores resolvidas. */
export function imageFromSvgElement(element: SVGSVGElement): ChartImage {
  const box = element.getBoundingClientRect();
  const width = Math.max(1, Math.round(box.width));
  const height = Math.max(1, Math.round(box.height));
  const clone = element.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  // "100%" não significa nada num arquivo isolado; o raster precisa de pixels.
  clone.setAttribute('width', String(width));
  clone.setAttribute('height', String(height));
  return { svg: resolveCssVariables(new XMLSerializer().serializeToString(clone)), width, height };
}

export async function exportChartImage(
  image: ChartImage,
  format: ImageFormat,
  filename: string,
): Promise<void> {
  if (format === 'svg') {
    downloadBlob(
      timestampedFilename(filename, 'svg'),
      new Blob([image.svg], { type: 'image/svg+xml;charset=utf-8' }),
    );
    return;
  }

  const url = URL.createObjectURL(new Blob([image.svg], { type: 'image/svg+xml;charset=utf-8' }));
  try {
    const element = new Image();
    await new Promise<void>((resolve, reject) => {
      element.onload = () => resolve();
      element.onerror = () => reject(fail('Não foi possível converter o gráfico em imagem.', 'Could not convert the chart to an image.'));
      element.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = image.width * RASTER_SCALE;
    canvas.height = image.height * RASTER_SCALE;
    const context = canvas.getContext('2d');
    if (!context) throw fail('Canvas indisponível neste navegador.', 'Canvas is not available in this browser.');

    // JPG não tem transparência: leva o fundo do tema, como o gráfico aparece na tela.
    if (format === 'jpg') {
      context.fillStyle = cssVariable('--background') || '#07110f';
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.drawImage(element, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, format === 'jpg' ? 'image/jpeg' : 'image/png', 0.92),
    );
    if (!blob) throw fail('Falha ao gerar a imagem.', 'Failed to generate the image.');
    downloadBlob(timestampedFilename(filename, format), blob);
  } finally {
    URL.revokeObjectURL(url);
  }
}
