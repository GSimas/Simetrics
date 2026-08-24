/**
 * Gerador de gráficos em alta resolução (Canvas 2D / Retina 2x) para relatórios.
 * Produz imagens PNG nítidas e estilizadas para PDF, Word (.docx) e Live Preview.
 * Totalmente bilíngue (Português / Inglês).
 */

export interface ChartRenderOptions {
  width?: number;
  height?: number;
  locale?: 'pt' | 'en';
  isDark?: boolean;
}

export function renderProductionTimelineCanvas(
  data: { year: number; count: number }[],
  options: ChartRenderOptions = {},
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 460;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const padding = { top: 60, right: 40, bottom: 60, left: 60 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'Annual Scientific Production Evolution (Articles / Year)' : 'Evolução da Produção Científica Anual (Artigos / Ano)',
    padding.left,
    36,
  );

  if (!data || data.length === 0) {
    ctx.fillStyle = '#64748B';
    ctx.font = '16px Inter, system-ui, sans-serif';
    ctx.fillText(isEn ? 'No year data available.' : 'Sem dados de anos disponíveis.', padding.left, height / 2);
    return canvas.toDataURL('image/png');
  }

  const sorted = [...data].sort((a, b) => a.year - b.year);
  const maxCount = Math.max(...sorted.map((d) => d.count), 1);
  const countStep = Math.ceil(maxCount / 5);

  ctx.strokeStyle = '#F1F5F9';
  ctx.lineWidth = 1.5;
  for (let i = 0; i <= 5; i++) {
    const val = i * countStep;
    const y = padding.top + chartH - (val / (countStep * 5)) * chartH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();

    ctx.fillStyle = '#94A3B8';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(val), padding.left - 10, y + 4);
  }

  const barWidth = Math.max(8, Math.min(36, (chartW / sorted.length) * 0.7));
  const stepX = chartW / sorted.length;

  sorted.forEach((d, idx) => {
    const x = padding.left + idx * stepX + (stepX - barWidth) / 2;
    const barH = (d.count / (countStep * 5)) * chartH;
    const y = padding.top + chartH - barH;

    const grad = ctx.createLinearGradient(x, y, x, y + barH);
    grad.addColorStop(0, '#2563EB');
    grad.addColorStop(1, '#3B82F6');
    ctx.fillStyle = grad;

    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barH, [4, 4, 0, 0]);
    ctx.fill();

    if (sorted.length <= 15 || idx % Math.ceil(sorted.length / 12) === 0 || idx === sorted.length - 1) {
      ctx.fillStyle = '#64748B';
      ctx.font = '12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(String(d.year), x + barWidth / 2, height - padding.bottom + 22);
    }
  });

  return canvas.toDataURL('image/png');
}

export function renderHorizontalBarChart(
  title: string,
  items: { label: string; value: number; sub?: string }[],
  options: ChartRenderOptions = {},
): string {
  const width = options.width ?? 1000;
  const height = options.height ?? Math.max(400, items.length * 36 + 100);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  const padding = { top: 60, right: 70, bottom: 40, left: 240 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(title, 40, 36);

  if (items.length === 0) return canvas.toDataURL('image/png');

  const maxVal = Math.max(...items.map((i) => i.value), 1);
  const rowHeight = chartH / items.length;
  const barH = Math.min(22, rowHeight * 0.65);

  items.forEach((item, idx) => {
    const y = padding.top + idx * rowHeight + (rowHeight - barH) / 2;
    const barW = (item.value / maxVal) * chartW;

    ctx.fillStyle = '#1E293B';
    ctx.font = 'bold 13px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    const truncatedLabel = item.label.length > 28 ? item.label.slice(0, 26) + '…' : item.label;
    ctx.fillText(truncatedLabel, padding.left - 15, y + barH / 2 + 5);

    ctx.fillStyle = '#F1F5F9';
    ctx.beginPath();
    ctx.roundRect(padding.left, y, chartW, barH, 4);
    ctx.fill();

    const grad = ctx.createLinearGradient(padding.left, y, padding.left + barW, y);
    grad.addColorStop(0, '#4F46E5');
    grad.addColorStop(1, '#7C3AED');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(padding.left, y, Math.max(8, barW), barH, 4);
    ctx.fill();

    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${item.value.toLocaleString(options.locale === 'en' ? 'en-US' : 'pt-BR')}${item.sub ? ` (${item.sub})` : ''}`, padding.left + barW + 10, y + barH / 2 + 4);
  });

  return canvas.toDataURL('image/png');
}

export function renderThemesPieChart(
  clusters: { clusterId: number; name: string; docCount: number; share: number }[],
  options: ChartRenderOptions = {},
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 460;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'AI Thematic Distribution Across Articles' : 'Distribuição Temática dos Artigos por IA',
    40,
    36,
  );

  if (clusters.length === 0) return canvas.toDataURL('image/png');

  const centerX = 260;
  const centerY = height / 2 + 10;
  const outerRadius = 140;
  const innerRadius = 75;

  const colors = [
    '#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EC4899',
    '#06B6D4', '#6366F1', '#14B8A6', '#F97316', '#84CC16',
  ];

  const total = clusters.reduce((acc, c) => acc + c.docCount, 0) || 1;
  let currentAngle = -Math.PI / 2;

  clusters.forEach((c, idx) => {
    const sliceAngle = (c.docCount / total) * 2 * Math.PI;
    const endAngle = currentAngle + sliceAngle;

    ctx.beginPath();
    ctx.arc(centerX, centerY, outerRadius, currentAngle, endAngle);
    ctx.arc(centerX, centerY, innerRadius, endAngle, currentAngle, true);
    ctx.closePath();
    ctx.fillStyle = colors[idx % colors.length] ?? '#3B82F6';
    ctx.fill();

    currentAngle = endAngle;
  });

  const legendX = 480;
  let legendY = 80;
  const rowH = Math.min(36, (height - 100) / clusters.length);

  clusters.forEach((c, idx) => {
    const color = colors[idx % colors.length] ?? '#3B82F6';

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(legendX, legendY + 4, 14, 14, 3);
    ctx.fill();

    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 13px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    const label = `${c.name} — ${c.docCount} docs (${((c.docCount / total) * 100).toFixed(1)}%)`;
    ctx.fillText(label, legendX + 24, legendY + 16);

    legendY += rowH;
  });

  return canvas.toDataURL('image/png');
}

export function renderWordCloudCanvas(
  terms: { entity: string; docCount: number; citations: number }[],
  options: ChartRenderOptions = {},
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 460;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'Keyword Cloud & Scientific Lexicometrics' : 'Nuvem de Termos & Lexicometria Científica',
    40,
    36,
  );

  if (!terms || terms.length === 0) return canvas.toDataURL('image/png');

  const topTerms = terms.slice(0, 30);
  const maxCount = Math.max(...topTerms.map((t) => t.docCount), 1);
  const minCount = Math.min(...topTerms.map((t) => t.docCount), 1);

  const colors = [
    '#2563EB', '#7C3AED', '#0D9488', '#0284C7', '#4F46E5',
    '#059669', '#D97706', '#DC2626', '#4338CA', '#0891B2',
  ];

  const cols = 5;
  const cellW = (width - 80) / cols;
  const rows = Math.ceil(topTerms.length / cols);
  const cellH = (height - 90) / rows;

  topTerms.forEach((t, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const x = 50 + col * cellW + cellW / 2;
    const y = 80 + row * cellH + cellH / 2;

    const normalized = (t.docCount - minCount) / (maxCount - minCount || 1);
    const fontSize = Math.floor(13 + normalized * 18);

    ctx.fillStyle = colors[idx % colors.length] ?? '#2563EB';
    ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`${t.entity} (${t.docCount})`, x, y);
  });

  return canvas.toDataURL('image/png');
}

/**
 * Gráfico da Rede de Coocorrência & Grafos de Conhecimento (SNA).
 */
export function renderNetworkGraphCanvas(
  nodes: { label: string; count?: number; community?: number; degreeAbsolute?: number }[],
  edges: { source: string; target: string; weight?: number }[],
  options: ChartRenderOptions = {},
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 540;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#0F172A'; // Fundo Dark Slate executivo para contraste de grafos
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'Co-occurrence Network & Scientific Communities (Louvain)' : 'Rede de Coocorrência & Comunidades Científicas (Louvain)',
    40,
    38,
  );

  ctx.fillStyle = '#94A3B8';
  ctx.font = '12px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? `${nodes.length} nodes · ${edges.length} edges · Topological clustering` : `${nodes.length} nós · ${edges.length} arestas · Agrupamento topológico`,
    40,
    58,
  );

  if (nodes.length === 0) return canvas.toDataURL('image/png');

  const topNodes = nodes.slice(0, 35);
  const nodeMap = new Map<string, { x: number; y: number; label: string; radius: number; color: string; comm: number }>();

  const communityColors = [
    '#38BDF8', '#A855F7', '#34D399', '#FBBF24', '#F43F5E',
    '#818CF8', '#2DD4BF', '#FB923C', '#A3E635', '#E879F9',
  ];

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
    const color = communityColors[comm % communityColors.length] ?? '#38BDF8';
    const nodeR = Math.max(6, Math.min(18, 6 + (n.count ? Math.sqrt(n.count) * 1.5 : 4)));

    nodeMap.set(n.label, { x, y, label: n.label, radius: nodeR, color, comm });
  });

  // Desenha Arestas
  ctx.lineWidth = 1.2;
  edges.slice(0, 100).forEach((edge) => {
    const s = nodeMap.get(edge.source);
    const t = nodeMap.get(edge.target);
    if (s && t) {
      ctx.strokeStyle = s.comm === t.comm ? `${s.color}55` : 'rgba(148, 163, 184, 0.25)';
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }
  });

  // Desenha Nós com Glow e Rótulos
  nodeMap.forEach((n, label) => {
    // Glow do nó
    const glow = ctx.createRadialGradient(n.x, n.y, n.radius * 0.2, n.x, n.y, n.radius * 2);
    glow.addColorStop(0, `${n.color}DD`);
    glow.addColorStop(1, `${n.color}00`);
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius * 2, 0, Math.PI * 2);
    ctx.fill();

    // Nó central
    ctx.fillStyle = n.color;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Rótulo do nó
    ctx.font = 'bold 11px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#F8FAFC';
    ctx.textAlign = 'center';
    const cleanLabel = label.length > 20 ? label.slice(0, 18) + '…' : label;
    ctx.fillText(cleanLabel, n.x, n.y + n.radius + 14);
  });

  return canvas.toDataURL('image/png');
}

/**
 * Gráfico de Mapa Global de Colaboração Internacional.
 */
export function renderWorldCollaborationMapCanvas(
  network: {
    nodes: { country: string; label: string; documents: number; latitude: number | null; longitude: number | null }[];
    edges: { source: string; target: string; documents: number }[];
  },
  options: ChartRenderOptions = {},
): string {
  const isEn = options.locale === 'en';
  const width = options.width ?? 1000;
  const height = options.height ?? 500;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#090D16'; // Fundo Ocean Navy
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = 'bold 22px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'Global International Scientific Collaboration Map' : 'Mapa Global de Colaboração Científica Internacional',
    40,
    38,
  );

  ctx.fillStyle = '#94A3B8';
  ctx.font = '12px Inter, system-ui, sans-serif';
  ctx.fillText(
    isEn ? 'Cross-border partnerships, international co-authorship arcs & output hubs' : 'Parcerias transfronteiriças, arcos de coautoria e centros de produção',
    40,
    58,
  );

  const padding = { top: 80, right: 50, bottom: 40, left: 50 };
  const mapW = width - padding.left - padding.right;
  const mapH = height - padding.top - padding.bottom;

  // Grade Cartográfica de Fundo (Meridianos & Paralelos)
  ctx.strokeStyle = '#1E293B';
  ctx.lineWidth = 0.8;
  for (let lon = -180; lon <= 180; lon += 60) {
    const x = padding.left + ((lon + 180) / 360) * mapW;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
  }
  for (let lat = -60; lat <= 80; lat += 30) {
    const y = padding.top + ((90 - lat) / 180) * mapH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  // Linha do Equador
  const equatorY = padding.top + (90 / 180) * mapH;
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, equatorY);
  ctx.lineTo(width - padding.right, equatorY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Projeção dos Países
  const countryCoords = new Map<string, { x: number; y: number; label: string; docs: number }>();
  network.nodes.forEach((n) => {
    if (n.latitude !== null && n.longitude !== null) {
      const x = padding.left + ((n.longitude + 180) / 360) * mapW;
      const y = padding.top + ((90 - n.latitude) / 180) * mapH;
      countryCoords.set(n.country, { x, y, label: n.label, docs: n.documents });
    }
  });

  // Arcos de Colaboração (Curvas de Bézier)
  ctx.lineWidth = 1.5;
  network.edges.slice(0, 40).forEach((edge) => {
    const s = countryCoords.get(edge.source);
    const t = countryCoords.get(edge.target);
    if (s && t) {
      const midX = (s.x + t.x) / 2;
      const midY = Math.min(s.y, t.y) - Math.abs(s.x - t.x) * 0.18;

      const grad = ctx.createLinearGradient(s.x, s.y, t.x, t.y);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.7)');
      grad.addColorStop(1, 'rgba(168, 85, 247, 0.7)');
      ctx.strokeStyle = grad;

      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.quadraticCurveTo(midX, midY, t.x, t.y);
      ctx.stroke();
    }
  });

  // Nós dos Países
  countryCoords.forEach((node) => {
    const radius = Math.max(5, Math.min(16, 4 + Math.sqrt(node.docs) * 1.2));

    // Glow
    const glow = ctx.createRadialGradient(node.x, node.y, radius * 0.3, node.x, node.y, radius * 2.5);
    glow.addColorStop(0, 'rgba(52, 211, 153, 0.8)');
    glow.addColorStop(1, 'rgba(52, 211, 153, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Ponto Central
    ctx.fillStyle = '#34D399';
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Nome do País
    ctx.font = 'bold 10px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#F1F5F9';
    ctx.textAlign = 'center';
    ctx.fillText(`${node.label} (${node.docs})`, node.x, node.y - radius - 4);
  });

  return canvas.toDataURL('image/png');
}
