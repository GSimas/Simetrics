import { fitText, FONT_MONO, FONT_SANS, measureText } from '@/components/charts/svg/text';
import { resolveCssVariables, type ChartImage } from '@/lib/export-image';

/**
 * Imagem de um ranking do Top 10 para a exportação.
 *
 * Na tela o ranking é HTML — lista clicável, que se ajusta a qualquer largura. Para
 * exportar, o mesmo conteúdo vira um gráfico de barras em SVG, com título e métrica, e
 * segue o caminho comum dos gráficos (SVG direto, ou JPG/PNG por canvas).
 */
export interface RankingExportItem {
  label: string;
  value: string;
  ratio: number;
  detail?: string | undefined;
}

const WIDTH = 760;
const PAD = 28;
const ROW = 38;
const HEADER = 74;

const escapeXml = (text: string): string =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function rankingImage(title: string, metric: string, items: readonly RankingExportItem[]): ChartImage {
  const height = HEADER + items.length * ROW + PAD;
  const valueWidth = Math.max(...items.map((item) => measureText(item.value, 13, FONT_SANS, 700)), 30);
  const labelLeft = PAD + 30;
  const labelWidth = WIDTH - labelLeft - PAD - valueWidth - 16;

  const rows = items
    .map((item, index) => {
      const y = HEADER + index * ROW;
      const text = fitText(item.detail ? `${item.label} · ${item.detail}` : item.label, labelWidth, 13);
      const barWidth = Math.max(2, labelWidth * item.ratio);
      return [
        `<text x="${PAD}" y="${y + 14}" font-family='${FONT_MONO}' font-size="12" fill="var(--muted-foreground)">${index + 1}</text>`,
        `<text x="${labelLeft}" y="${y + 14}" font-size="13" fill="var(--foreground)">${escapeXml(text)}</text>`,
        `<rect x="${labelLeft}" y="${y + 21}" width="${labelWidth}" height="6" rx="3" fill="var(--muted)"/>`,
        `<rect x="${labelLeft}" y="${y + 21}" width="${barWidth.toFixed(1)}" height="6" rx="3" fill="var(--highlight)"/>`,
        `<text x="${WIDTH - PAD}" y="${y + 20}" text-anchor="end" font-size="13" font-weight="700" fill="var(--foreground)">${escapeXml(item.value)}</text>`,
      ].join('');
    })
    .join('');

  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${height}" viewBox="0 0 ${WIDTH} ${height}" font-family='${FONT_SANS}'>` +
    `<rect width="${WIDTH}" height="${height}" fill="var(--card)"/>` +
    `<text x="${PAD}" y="${PAD + 12}" font-size="18" font-weight="700" fill="var(--foreground)">${escapeXml(title)}</text>` +
    `<text x="${PAD}" y="${PAD + 32}" font-family='${FONT_MONO}' font-size="11" letter-spacing="1.2" fill="var(--highlight)">${escapeXml(metric.toUpperCase())}</text>` +
    rows +
    `</svg>`;

  return { svg: resolveCssVariables(svg), width: WIDTH, height };
}
