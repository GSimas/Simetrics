/**
 * Gerador de gráficos em alta resolução (Canvas 2D / Retina 2x) para relatórios.
 * Produz imagens PNG nítidas na identidade Simetrics/Scientata (tema claro, papel e tinta)
 * para PDF, Word (.docx) e Live Preview.
 * Totalmente bilíngue (Português / Inglês).
 */
import { geoGraticule10, geoNaturalEarth1, geoPath, type GeoPermissibleObjects } from 'd3-geo';
import { feature } from 'topojson-client';
import type { FeatureCollection, Geometry } from 'geojson';
import type { GeometryCollection, Topology } from 'topojson-specification';
import world from 'world-atlas/countries-110m.json';

import { PALETTE } from '@/features/overview/viz-shared';
import { identityKey } from '@/lib/use-async-result';

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

function renderProductionTimelineCanvas(
  data: { year: number; count: number }[],
  options: ChartRenderOptions = {},
): string {
  return cached(`production ${identityKey(data)} ${optionsKey(options)}`, () =>
    drawProductionTimelineCanvas(data, options),
  );
}

function drawProductionTimelineCanvas(
  data: { year: number; count: number }[],
  options: ChartRenderOptions,
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 460;
  const { canvas, ctx } = createCanvas(width, height);
  if (!ctx) return '';

  const padding = { top: 70, right: 40, bottom: 60, left: 60 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  drawFrame(
    ctx,
    width,
    height,
    isEn ? 'Annual Scientific Production Evolution (Articles / Year)' : 'Evolução da Produção Científica Anual (Artigos / Ano)',
  );

  if (!data || data.length === 0) {
    ctx.fillStyle = BRAND.inkMuted;
    ctx.font = `400 16px ${SANS}`;
    ctx.fillText(isEn ? 'No year data available.' : 'Sem dados de anos disponíveis.', padding.left, height / 2);
    return canvas.toDataURL('image/png');
  }

  const sorted = [...data].sort((a, b) => a.year - b.year);
  const maxCount = Math.max(...sorted.map((d) => d.count), 1);
  const countStep = Math.ceil(maxCount / 5);

  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const val = i * countStep;
    const y = Math.round(padding.top + chartH - (val / (countStep * 5)) * chartH) + 0.5;
    ctx.strokeStyle = i === 0 ? BRAND.ink : BRAND.border;
    ctx.globalAlpha = i === 0 ? 0.6 : 0.7;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.globalAlpha = 1;

    monoLabel(ctx, String(val), padding.left - 10, y + 4, BRAND.inkMuted, 11, 'right');
  }

  const barWidth = Math.max(8, Math.min(36, (chartW / sorted.length) * 0.7));
  const stepX = chartW / sorted.length;

  sorted.forEach((d, idx) => {
    const x = padding.left + idx * stepX + (stepX - barWidth) / 2;
    const barH = (d.count / (countStep * 5)) * chartH;
    const y = padding.top + chartH - barH;

    // Barras chapadas em pinho; a última recebe o destaque lima no topo
    ctx.fillStyle = BRAND.pine;
    ctx.fillRect(x, y, barWidth, barH);
    if (idx === sorted.length - 1 && barH > 3) {
      ctx.fillStyle = BRAND.lime;
      ctx.fillRect(x, y, barWidth, 3);
    }

    if (sorted.length <= 15 || idx % Math.ceil(sorted.length / 12) === 0 || idx === sorted.length - 1) {
      monoLabel(ctx, String(d.year), x + barWidth / 2, height - padding.bottom + 22, BRAND.inkMuted, 11, 'center');
    }
  });

  return canvas.toDataURL('image/png');
}

function renderHorizontalBarChart(
  title: string,
  items: { label: string; value: number; sub?: string }[],
  options: ChartRenderOptions = {},
): string {
  return cached(`bars ${JSON.stringify([title, items])} ${optionsKey(options)}`, () =>
    drawHorizontalBarChart(title, items, options),
  );
}

function drawHorizontalBarChart(
  title: string,
  items: { label: string; value: number; sub?: string }[],
  options: ChartRenderOptions,
): string {
  const width = options.width ?? 1000;
  const height = options.height ?? Math.max(400, items.length * 36 + 100);
  const { canvas, ctx } = createCanvas(width, height);
  if (!ctx) return '';

  const padding = { top: 76, right: 70, bottom: 30, left: 240 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  drawFrame(ctx, width, height, title);

  if (items.length === 0) return canvas.toDataURL('image/png');

  const maxVal = Math.max(...items.map((i) => i.value), 1);
  const rowHeight = chartH / items.length;
  const barH = Math.min(22, rowHeight * 0.65);

  items.forEach((item, idx) => {
    const y = padding.top + idx * rowHeight + (rowHeight - barH) / 2;
    const barW = (item.value / maxVal) * chartW;

    ctx.fillStyle = BRAND.ink;
    ctx.font = `600 13px ${SANS}`;
    ctx.textAlign = 'right';
    const truncatedLabel = item.label.length > 28 ? item.label.slice(0, 26) + '…' : item.label;
    ctx.fillText(truncatedLabel, padding.left - 15, y + barH / 2 + 5);

    // Trilho em tom neutro e barra chapada em pinho
    ctx.fillStyle = BRAND.muted;
    ctx.fillRect(padding.left, y, chartW, barH);
    ctx.fillStyle = BRAND.pine;
    ctx.fillRect(padding.left, y, Math.max(3, barW), barH);

    // Barras longas levam o valor por dentro (papel sobre pinho); curtas, por fora
    const inside = barW > chartW * 0.6;
    monoLabel(
      ctx,
      `${item.value.toLocaleString(options.locale === 'en' ? 'en-US' : 'pt-BR')}${item.sub ? ` · ${item.sub}` : ''}`,
      inside ? padding.left + barW - 10 : padding.left + barW + 10,
      y + barH / 2 + 4,
      inside ? BRAND.card : BRAND.ink,
      11,
      inside ? 'right' : 'left',
    );
  });

  return canvas.toDataURL('image/png');
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

function renderWordCloudCanvas(
  terms: { entity: string; docCount: number; citations: number }[],
  options: ChartRenderOptions = {},
): string {
  return cached(`wordcloud ${identityKey(terms)} ${optionsKey(options)}`, () =>
    drawWordCloudCanvas(terms, options),
  );
}

function drawWordCloudCanvas(
  terms: { entity: string; docCount: number; citations: number }[],
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
    isEn ? 'Keyword Cloud & Scientific Lexicometrics' : 'Nuvem de Termos & Lexicometria Científica',
  );

  if (!terms || terms.length === 0) return canvas.toDataURL('image/png');

  const topTerms = terms.slice(0, 30);
  const maxCount = Math.max(...topTerms.map((t) => t.docCount), 1);
  const minCount = Math.min(...topTerms.map((t) => t.docCount), 1);

  const cols = 5;
  const cellW = (width - 80) / cols;
  const rows = Math.ceil(topTerms.length / cols);
  const cellH = (height - 90) / rows;

  topTerms.forEach((t, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const x = 50 + col * cellW + cellW / 2;
    const y = 84 + row * cellH + cellH / 2;

    const normalized = (t.docCount - minCount) / (maxCount - minCount || 1);
    // Reduz a fonte até o termo caber na célula
    let fontSize = Math.floor(13 + normalized * 18);
    ctx.font = `700 ${fontSize}px ${SANS}`;
    while (fontSize > 10 && ctx.measureText(t.entity).width > cellW - 16) {
      fontSize -= 1;
      ctx.font = `700 ${fontSize}px ${SANS}`;
    }

    ctx.fillStyle = pal(idx);
    ctx.textAlign = 'center';
    ctx.fillText(t.entity, x, y);
    monoLabel(ctx, String(t.docCount), x, y + 14, BRAND.inkMuted, 10, 'center');
  });

  return canvas.toDataURL('image/png');
}

/**
 * Gráfico da Rede de Coocorrência & Grafos de Conhecimento (SNA).
 */
function renderNetworkGraphCanvas(
  nodes: { label: string; count?: number; community?: number; degreeAbsolute?: number }[],
  edges: { source: string; target: string; weight?: number }[],
  options: ChartRenderOptions = {},
): string {
  return cached(`network ${identityKey(nodes)} ${identityKey(edges)} ${optionsKey(options)}`, () =>
    drawNetworkGraphCanvas(nodes, edges, options),
  );
}

function drawNetworkGraphCanvas(
  nodes: { label: string; count?: number; community?: number; degreeAbsolute?: number }[],
  edges: { source: string; target: string; weight?: number }[],
  options: ChartRenderOptions,
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 540;
  const { canvas, ctx } = createCanvas(width, height);
  if (!ctx) return '';

  drawFrame(
    ctx,
    width,
    height,
    isEn ? 'Co-occurrence Network & Scientific Communities (Louvain)' : 'Rede de Coocorrência & Comunidades Científicas (Louvain)',
    isEn ? `${nodes.length} nodes · ${edges.length} edges · Topological clustering` : `${nodes.length} nós · ${edges.length} arestas · Agrupamento topológico`,
  );

  if (nodes.length === 0) return canvas.toDataURL('image/png');

  const topNodes = nodes.slice(0, 35);
  const nodeMap = new Map<string, { x: number; y: number; label: string; radius: number; color: string; comm: number }>();

  const centerX = width / 2;
  const centerY = height / 2 + 25;
  const radius = Math.min(centerX - 100, centerY - 80);

  // Distribuição dos nós
  topNodes.forEach((n, idx) => {
    const angle = (idx / topNodes.length) * 2 * Math.PI - Math.PI / 2;
    // Variação leve no raio para efeito de nebulosa orgânica
    const rVar = radius * (0.65 + 0.35 * Math.sin(idx * 3.7));
    const x = centerX + rVar * Math.cos(angle);
    const y = centerY + rVar * Math.sin(angle);
    const comm = n.community ?? (idx % 5);
    const nodeR = Math.max(6, Math.min(18, 6 + (n.count ? Math.sqrt(n.count) * 1.5 : 4)));

    nodeMap.set(n.label, { x, y, label: n.label, radius: nodeR, color: pal(comm), comm });
  });

  // Desenha Arestas em tinta translúcida
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(7, 17, 15, 0.15)';
  edges.slice(0, 100).forEach((edge) => {
    const s = nodeMap.get(edge.source);
    const t = nodeMap.get(edge.target);
    if (s && t) {
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }
  });

  // Desenha Nós chapados com contorno em papel e rótulos com halo
  nodeMap.forEach((n, label) => {
    ctx.fillStyle = n.color;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = BRAND.card;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const cleanLabel = label.length > 20 ? label.slice(0, 18) + '…' : label;
    ctx.font = `600 11px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = BRAND.card;
    ctx.lineWidth = 3;
    ctx.strokeText(cleanLabel, n.x, n.y + n.radius + 14);
    ctx.fillStyle = BRAND.ink;
    ctx.fillText(cleanLabel, n.x, n.y + n.radius + 14);
  });

  return canvas.toDataURL('image/png');
}

// Geometria dos países (world-atlas 1:110m): decodificada uma única vez por sessão, na
// primeira vez que o mapa é desenhado — não ao carregar o módulo (a aba Relatório é
// pré-carregada no ócio, e isso gastaria CPU sem ninguém ter pedido o mapa).
let countries: FeatureCollection<Geometry> | null = null;
function worldCountries(): FeatureCollection<Geometry> {
  countries ??= feature(
    world as unknown as Topology,
    (world as unknown as Topology).objects.countries as GeometryCollection,
  ) as FeatureCollection<Geometry>;
  return countries;
}

/**
 * Gráfico de Mapa Global de Colaboração Internacional.
 */
function renderWorldCollaborationMapCanvas(
  network: {
    nodes: { country: string; label: string; documents: number; latitude: number | null; longitude: number | null }[];
    edges: { source: string; target: string; documents: number }[];
  },
  options: ChartRenderOptions = {},
): string {
  return cached(`map ${identityKey(network)} ${optionsKey(options)}`, () =>
    drawWorldCollaborationMapCanvas(network, options),
  );
}

function drawWorldCollaborationMapCanvas(
  network: {
    nodes: { country: string; label: string; documents: number; latitude: number | null; longitude: number | null }[];
    edges: { source: string; target: string; documents: number }[];
  },
  options: ChartRenderOptions,
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 500;
  const { canvas, ctx } = createCanvas(width, height);
  if (!ctx) return '';

  drawFrame(
    ctx,
    width,
    height,
    isEn ? 'Global International Scientific Collaboration Map' : 'Mapa Global de Colaboração Científica Internacional',
    isEn ? 'Cross-border partnerships, international co-authorship arcs & output hubs' : 'Parcerias transfronteiriças, arcos de coautoria e centros de produção',
  );

  const padding = { top: 80, right: 30, bottom: 20, left: 30 };
  const projection = geoNaturalEarth1().fitExtent(
    [
      [padding.left, padding.top],
      [width - padding.right, height - padding.bottom],
    ],
    { type: 'Sphere' },
  );
  const path = geoPath(projection, ctx);

  // Oceano em papel, graticulado e países em tom neutro com fronteiras finas
  ctx.beginPath();
  path({ type: 'Sphere' });
  ctx.fillStyle = BRAND.paper;
  ctx.fill();

  ctx.beginPath();
  path(geoGraticule10());
  ctx.strokeStyle = BRAND.border;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 0.5;
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.beginPath();
  path(worldCountries() as GeoPermissibleObjects);
  ctx.fillStyle = BRAND.muted;
  ctx.fill();
  ctx.strokeStyle = BRAND.border;
  ctx.lineWidth = 0.6;
  ctx.stroke();

  ctx.beginPath();
  path({ type: 'Sphere' });
  ctx.strokeStyle = BRAND.border;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Projeção dos Países
  const countryCoords = new Map<string, { x: number; y: number; label: string; docs: number }>();
  network.nodes.forEach((n) => {
    if (n.latitude !== null && n.longitude !== null) {
      const p = projection([n.longitude, n.latitude]);
      if (p) countryCoords.set(n.country, { x: p[0], y: p[1], label: n.label, docs: n.documents });
    }
  });

  // Arcos de Colaboração (Curvas de Bézier) em pinho, alternando com ciano
  ctx.lineWidth = 1.4;
  network.edges.slice(0, 40).forEach((edge, idx) => {
    const s = countryCoords.get(edge.source);
    const t = countryCoords.get(edge.target);
    if (s && t) {
      const midX = (s.x + t.x) / 2;
      const midY = Math.min(s.y, t.y) - Math.abs(s.x - t.x) * 0.18;

      ctx.strokeStyle = idx % 3 === 2 ? 'rgba(83, 215, 208, 0.85)' : 'rgba(35, 110, 94, 0.55)';
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.quadraticCurveTo(midX, midY, t.x, t.y);
      ctx.stroke();
    }
  });

  // Marcadores dos Países
  countryCoords.forEach((node) => {
    const radius = Math.max(4, Math.min(14, 3 + Math.sqrt(node.docs) * 1.1));

    ctx.fillStyle = BRAND.pine;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = BRAND.card;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Nome do País com halo em papel
    const text = `${node.label} (${node.docs})`;
    ctx.font = `600 10px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = BRAND.card;
    ctx.lineWidth = 3;
    ctx.strokeText(text, node.x, node.y - radius - 4);
    ctx.fillStyle = BRAND.ink;
    ctx.fillText(text, node.x, node.y - radius - 4);
  });

  return canvas.toDataURL('image/png');
}

// ---------------------------------------------------------------------------------------
// Gráficos do relatório
//
// A prévia, o PDF e o DOCX pedem cada gráfico por aqui, com os mesmos dados, textos e
// dimensões: as três saídas não divergem, e a exportação sempre reaproveita as imagens da
// prévia (ver `cached`). As dimensões também reservam o espaço da prévia.

type ReportLocale = 'pt' | 'en';

export const REPORT_CHART_SIZE = {
  production: { width: 1000, height: 420 },
  bars: { width: 1000, height: 420 },
  worldMap: { width: 1000, height: 500 },
  wordCloud: { width: 1000, height: 420 },
  themes: { width: 1000, height: 420 },
  network: { width: 1000, height: 520 },
} as const;

export function reportProductionChart(docsPerYear: { year: number; count: number }[], locale: ReportLocale): string {
  return renderProductionTimelineCanvas(docsPerYear, { ...REPORT_CHART_SIZE.production, locale });
}

export function reportAuthorsChart(
  authors: readonly { entity: string; docCount: number; citations: number; h: number }[],
  locale: ReportLocale,
): string {
  const items = authors.slice(0, 10).map((a) => ({ label: a.entity, value: a.docCount, sub: `${a.citations} cit. | h=${a.h}` }));
  const title = locale === 'en' ? 'Top 10 Most Prolific Authors (Published Papers)' : 'Top 10 Autores Mais Produtivos (Artigos Publicados)';
  return renderHorizontalBarChart(title, items, { ...REPORT_CHART_SIZE.bars, locale });
}

export function reportCountriesChart(
  countries: readonly { entity: string; docCount: number; citations: number }[],
  locale: ReportLocale,
): string {
  const items = countries.slice(0, 10).map((c) => ({ label: c.entity, value: c.docCount, sub: `${c.citations} cit.` }));
  const title = locale === 'en' ? 'Top 10 Leading Countries by Scientific Output' : 'Top 10 Países com Maior Produção Científica';
  return renderHorizontalBarChart(title, items, { ...REPORT_CHART_SIZE.bars, locale });
}

export function reportWorldMapChart(
  collaboration: Parameters<typeof renderWorldCollaborationMapCanvas>[0],
  locale: ReportLocale,
): string {
  return renderWorldCollaborationMapCanvas(collaboration, { ...REPORT_CHART_SIZE.worldMap, locale });
}

export function reportWordCloudChart(keywords: Parameters<typeof renderWordCloudCanvas>[0], locale: ReportLocale): string {
  return renderWordCloudCanvas(keywords, { ...REPORT_CHART_SIZE.wordCloud, locale });
}

export function reportThemesChart(
  clusters: readonly { clusterId: number; size: number }[],
  totalDocs: number,
  locale: ReportLocale,
): string {
  const items = clusters.map((c) => ({
    clusterId: c.clusterId,
    name: `Tema ${c.clusterId + 1}`,
    docCount: c.size,
    share: totalDocs > 0 ? (c.size / totalDocs) * 100 : 0,
  }));
  return renderThemesPieChart(items, { ...REPORT_CHART_SIZE.themes, locale });
}

export function reportNetworkChart(
  nodes: Parameters<typeof renderNetworkGraphCanvas>[0],
  edges: Parameters<typeof renderNetworkGraphCanvas>[1],
  locale: ReportLocale,
): string {
  return renderNetworkGraphCanvas(nodes, edges, { ...REPORT_CHART_SIZE.network, locale });
}
