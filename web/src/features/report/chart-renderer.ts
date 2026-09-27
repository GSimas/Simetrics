/**
 * Gráfico de distribuição dos temas (Canvas 2D / Retina 2x) e tokens do tema claro
 * Scientata, compartilhados pelos geradores de PDF e Word.
 * Totalmente bilíngue (Português / Inglês).
 */
import { PALETTE } from '@/features/overview/viz-shared';

export { PALETTE };

/** Tokens do tema claro Scientata — compartilhados pelos geradores de PDF e Word. */
export const BRAND = {
  paper: '#f0eee6',
  card: '#f7f6f1',
  muted: '#e6e3d8',
  border: '#d5d2c6',
  ink: '#07110f',
  inkMuted: '#56625d',
  pine: '#236e5e',
  lime: '#b8ff4a',
  cyan: '#53d7d0',
  destructive: '#c2412d',
} as const;

const SANS = 'Manrope, system-ui, sans-serif';
const MONO = '"DM Mono", ui-monospace, monospace';

const pal = (i: number): string => PALETTE[i % PALETTE.length] ?? BRAND.pine;

export interface ChartRenderOptions {
  width?: number;
  height?: number;
  locale?: 'pt' | 'en';
  isDark?: boolean;
}

/**
 * Moldura comum: fundo de cartão, contorno fino, marca lima, título em tinta e,
 * opcionalmente, uma legenda em mono maiúsculo.
 */
function drawFrame(ctx: CanvasRenderingContext2D, width: number, height: number, title: string, subtitle?: string): void {
  ctx.fillStyle = BRAND.card;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = BRAND.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  // Marca lima curta ao lado do título
  ctx.fillStyle = BRAND.lime;
  ctx.fillRect(40, 18, 18, 4);

  ctx.textAlign = 'left';
  ctx.fillStyle = BRAND.ink;
  ctx.font = `700 22px ${SANS}`;
  ctx.fillText(title, 40, 44);

  if (subtitle) monoLabel(ctx, subtitle.toUpperCase(), 40, 64, BRAND.inkMuted, 11);
}

/** Rótulo mono espaçado (eixos, legendas). */
function monoLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string = BRAND.inkMuted,
  size = 11,
  align: CanvasTextAlign = 'left',
): void {
  ctx.save();
  ctx.font = `400 ${size}px ${MONO}`;
  ctx.letterSpacing = '1px';
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function createCanvas(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D | null } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return { canvas, ctx: canvas.getContext('2d') };
}

// ---------------------------------------------------------------------------------------
// Imagens já geradas
//
// A prévia do relatório e a exportação (PDF e DOCX) pedem os mesmos gráficos. Cada imagem
// fica guardada pela entrada que a gerou — os dados (pela identidade do objeto, que o app
// nunca altera no lugar; ou pelo conteúdo, nas listas curtas montadas na hora) e as
// opções. A exportação reaproveita a imagem da prévia quando tudo coincide e desenha
// normalmente quando não: o resultado é sempre o mesmo que seria desenhado.

/** Os 7 gráficos da prévia, com folga para trocar de idioma ou de base. */
const IMAGE_CACHE_SIZE = 12;
const imageCache = new Map<string, string>();

function cached(key: string, draw: () => string): string {
  const hit = imageCache.get(key);
  if (hit !== undefined) {
    // Reinsere no fim: a ordem do Map vira a ordem de uso (LRU).
    imageCache.delete(key);
    imageCache.set(key, hit);
    return hit;
  }
  const image = draw();
  // Com alguma fonte ainda carregando, o canvas escreve na fonte de reserva: essa imagem
  // serve à prévia, mas não fica guardada para a exportação.
  if (image && document.fonts.status === 'loaded') {
    imageCache.set(key, image);
    if (imageCache.size > IMAGE_CACHE_SIZE) {
      const oldest = imageCache.keys().next().value;
      if (oldest !== undefined) imageCache.delete(oldest);
    }
  }
  return image;
}

function optionsKey(options: ChartRenderOptions): string {
  return `${options.width ?? ''}x${options.height ?? ''} ${options.locale ?? ''} ${options.isDark ? 'dark' : ''}`;
}

function renderThemesPieChart(
  clusters: { clusterId: number; name: string; docCount: number; share: number }[],
  options: ChartRenderOptions = {},
): string {
  return cached(`themes ${JSON.stringify(clusters)} ${optionsKey(options)}`, () =>
    drawThemesPieChart(clusters, options),
  );
}

function drawThemesPieChart(
  clusters: { clusterId: number; name: string; docCount: number; share: number }[],
  options: ChartRenderOptions,
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 460;
  const { canvas, ctx } = createCanvas(width, height);
  if (!ctx) return '';

  drawFrame(
    ctx,
    width,
    height,
    isEn ? 'AI Thematic Distribution Across Articles' : 'Distribuição Temática dos Artigos por IA',
  );

  if (clusters.length === 0) return canvas.toDataURL('image/png');

  const centerX = 260;
  const centerY = height / 2 + 20;
  const outerRadius = 140;
  const innerRadius = 75;

  const total = clusters.reduce((acc, c) => acc + c.docCount, 0) || 1;
  let currentAngle = -Math.PI / 2;

  clusters.forEach((c, idx) => {
    const sliceAngle = (c.docCount / total) * 2 * Math.PI;
    const endAngle = currentAngle + sliceAngle;

    ctx.beginPath();
    ctx.arc(centerX, centerY, outerRadius, currentAngle, endAngle);
    ctx.arc(centerX, centerY, innerRadius, endAngle, currentAngle, true);
    ctx.closePath();
    ctx.fillStyle = pal(idx);
    ctx.fill();
    // Separador fino em papel entre as fatias
    ctx.strokeStyle = BRAND.card;
    ctx.lineWidth = 2;
    ctx.stroke();

    currentAngle = endAngle;
  });

  // Total no centro do anel
  ctx.fillStyle = BRAND.ink;
  ctx.font = `700 28px ${SANS}`;
  ctx.textAlign = 'center';
  ctx.fillText(total.toLocaleString(isEn ? 'en-US' : 'pt-BR'), centerX, centerY + 6);
  monoLabel(ctx, 'DOCS', centerX, centerY + 26, BRAND.inkMuted, 10, 'center');

  const legendX = 480;
  let legendY = 84;
  const rowH = Math.min(36, (height - 100) / clusters.length);

  clusters.forEach((c, idx) => {
    ctx.fillStyle = pal(idx);
    ctx.fillRect(legendX, legendY + 4, 14, 14);

    ctx.fillStyle = BRAND.ink;
    ctx.font = `600 13px ${SANS}`;
    ctx.textAlign = 'left';
    ctx.fillText(c.name, legendX + 24, legendY + 16);
    monoLabel(
      ctx,
      `${c.docCount} DOCS · ${((c.docCount / total) * 100).toFixed(1)}%`,
      legendX + 24 + ctx.measureText(c.name).width + 12,
      legendY + 16,
      BRAND.inkMuted,
      11,
    );

    legendY += rowH;
  });

  return canvas.toDataURL('image/png');
}

// ---------------------------------------------------------------------------------------
// Gráficos do relatório
//
// Só a distribuição de temas segue em canvas: é o único gráfico do relatório sem
// equivalente nas abas. Os demais são os próprios componentes do app, capturados da
// prévia (ver `capture.ts`).

type ReportLocale = 'pt' | 'en';

const THEMES_SIZE = { width: 1000, height: 420 } as const;

export function reportThemesChart(
  clusters: readonly { clusterId: number; size: number }[],
  totalDocs: number,
  locale: ReportLocale,
): string {
  const items = clusters.map((c) => ({
    clusterId: c.clusterId,
    name: `${locale === 'en' ? 'Theme' : 'Tema'} ${c.clusterId + 1}`,
    docCount: c.size,
    share: totalDocs > 0 ? (c.size / totalDocs) * 100 : 0,
  }));
  return renderThemesPieChart(items, { ...THEMES_SIZE, locale });
}

/** Pizza de temas já nomeados — usada pela classificação híbrida, cujas categorias têm nome. */
export function reportNamedThemesChart(
  themes: readonly { name: string; documents: number }[],
  totalDocs: number,
  locale: ReportLocale,
): string {
  const items = themes
    .filter((theme) => theme.documents > 0)
    .map((theme, index) => ({
      clusterId: index,
      name: theme.name,
      docCount: theme.documents,
      share: totalDocs > 0 ? (theme.documents / totalDocs) * 100 : 0,
    }));
  return renderThemesPieChart(items, { ...THEMES_SIZE, locale });
}
