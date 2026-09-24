import jsPDF from 'jspdf';
import autoTable, { type UserOptions } from 'jspdf-autotable';

import type { AnalyticsBundle, EntityTables } from '@/workers/analytics.worker';
import type { CooccurrenceReport, SnaReport } from '@/core/graph';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import type { ClusteringResult } from '@/core/clustering';
import type { Dataset } from '@/lib/types';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import {
  BRAND,
  reportAuthorsChart,
  reportCountriesChart,
  reportNetworkChart,
  reportProductionChart,
  reportThemesChart,
  reportWordCloudChart,
  reportWorldMapChart,
} from './chart-renderer';

export interface ReportSectionsSelection {
  summary: boolean;
  kpis: boolean;
  chartProduction: boolean;
  authors: boolean;
  chartAuthors: boolean;
  countries: boolean;
  chartCountries: boolean;
  chartWorldMap: boolean;
  venues: boolean;
  keywords: boolean;
  chartKeywords: boolean;
  themes: boolean;
  chartThemes: boolean;
  networkTopology: boolean;
  chartNetwork: boolean;
  topDocuments: boolean;
}

export interface PdfReportData {
  dataset: Dataset;
  overview: AnalyticsBundle | null;
  tables: EntityTables | null;
  sna: SnaReport | null;
  network: CooccurrenceReport | null;
  collaboration: CollaborationNetwork | null;
  clustering: ClusteringResult | null;
  selection: ReportSectionsSelection;
  topN: number;
  locale: 'pt' | 'en';
}

type Rgb = [number, number, number];

/** Converte '#rrggbb' em tupla RGB para o jsPDF. */
function rgb(hex: string): Rgb {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

// Paleta Scientata (tema claro) em RGB
const C = {
  paper: rgb(BRAND.paper),
  card: rgb(BRAND.card),
  muted: rgb(BRAND.muted),
  border: rgb(BRAND.border),
  ink: rgb(BRAND.ink),
  inkMuted: rgb(BRAND.inkMuted),
  pine: rgb(BRAND.pine),
  lime: rgb(BRAND.lime),
};

function formatMetricVal(val: number | string, decimals = 4): string {
  if (typeof val === 'string') return val;
  if (!Number.isFinite(val)) return '—';
  return val.toFixed(decimals);
}

export function generatePdfReport({
  dataset,
  overview,
  tables,
  sna,
  network,
  collaboration,
  clustering,
  selection,
  topN = 15,
  locale = 'pt',
}: PdfReportData): void {
  const isEn = locale === 'en';
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'pt',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 40;
  const contentWidth = pageWidth - margin * 2;
  let cursorY = margin;

  // Pinta o fundo papel de cada página uma única vez (antes do conteúdo)
  const paintedPages = new Set<number>();
  const paintPage = () => {
    const page = doc.getCurrentPageInfo().pageNumber;
    if (paintedPages.has(page)) return;
    paintedPages.add(page);
    doc.setFillColor(...C.paper);
    doc.rect(0, 0, pageWidth, pageHeight, 'F');
  };
  paintPage();

  const checkPageBreak = (neededHeight: number) => {
    if (cursorY + neededHeight > pageHeight - margin - 30) {
      doc.addPage();
      paintPage();
      cursorY = margin + 20;
    }
  };

  // Rótulo "eyebrow": mono, maiúsculo, espaçado
  const eyebrow = (text: string, x: number, y: number, color: Rgb = C.pine, size = 7.5) => {
    doc.setFont('courier', 'bold');
    doc.setFontSize(size);
    doc.setTextColor(...color);
    doc.text(text.toUpperCase(), x, y, { charSpace: 0.8 });
  };

  // Cabeçalho de seção: eyebrow numerado em pinho + título em tinta
  const sectionHeading = (num: number, title: string) => {
    eyebrow(`— ${String(num).padStart(2, '0')}`, margin, cursorY);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(...C.ink);
    doc.text(title, margin, cursorY + 16);
    cursorY += 26;
  };

  // Estilo comum das tabelas: cabeçalho em tinta, corpo alternando cartão/papel, bordas finas
  const tableBase: Partial<UserOptions> = {
    margin: { left: margin, right: margin },
    theme: 'grid',
    willDrawPage: paintPage,
    headStyles: { fillColor: C.ink, textColor: C.paper, fontSize: 8, fontStyle: 'bold' },
    bodyStyles: { fillColor: C.card },
    alternateRowStyles: { fillColor: C.paper },
  };
  const cellStyles = (fontSize: number, cellPadding: number) => ({
    font: 'helvetica',
    fontSize,
    cellPadding,
    textColor: C.ink,
    lineColor: C.border,
    lineWidth: 0.5,
  });

  // Cartões de KPI chapados: rótulo mono + valor grande
  const kpiCards = (items: [string, string][]) => {
    const cols = 4;
    const gap = 8;
    const cardW = (contentWidth - gap * (cols - 1)) / cols;
    const cardH = 50;
    items.forEach(([label, value], idx) => {
      const x = margin + (idx % cols) * (cardW + gap);
      const y = cursorY + Math.floor(idx / cols) * (cardH + gap);
      doc.setFillColor(...C.card);
      doc.setDrawColor(...C.border);
      doc.setLineWidth(0.5);
      doc.rect(x, y, cardW, cardH, 'FD');
      eyebrow(label, x + 10, y + 16, C.inkMuted, 6.5);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(17);
      doc.setTextColor(...C.ink);
      doc.text(value, x + 10, y + 39);
    });
    cursorY += Math.ceil(items.length / cols) * (cardH + gap) + 8;
  };

  const totalCitations = dataset.reduce((acc, d) => acc + (toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0), 0);
  const meanCitations = dataset.length > 0 ? totalCitations / dataset.length : 0;

  // --- CABEÇALHO DO RELATÓRIO ---
  eyebrow(isEn ? 'Simetrics · Scientometric Report' : 'Simetrics · Relatório Cientométrico', margin, cursorY + 6);
  cursorY += 32;

  // Título com palavra de destaque em serifa itálica
  const titleMain = isEn ? 'Scientometric Intelligence ' : 'Relatório Cientométrico & ';
  const titleAccent = isEn ? 'Report' : 'Bibliométrico';
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(22);
  doc.setTextColor(...C.ink);
  doc.text(titleMain, margin, cursorY);
  const titleMainW = doc.getTextWidth(titleMain);
  doc.setFont('times', 'italic');
  doc.setFontSize(25);
  doc.setTextColor(...C.pine);
  doc.text(titleAccent, margin + titleMainW, cursorY);
  cursorY += 18;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(...C.inkMuted);
  const dateStr = new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  doc.text(
    `${isEn ? 'Generated on' : 'Emitido em'}: ${dateStr} · ${isEn ? 'Simetrics · A Scientata application' : 'Simetrics · Uma aplicação Scientata'}`,
    margin,
    cursorY,
  );
  cursorY += 12;

  // Régua fina em pinho com bloco curto em lima
  doc.setDrawColor(...C.pine);
  doc.setLineWidth(1);
  doc.line(margin, cursorY, pageWidth - margin, cursorY);
  doc.setFillColor(...C.lime);
  doc.rect(margin, cursorY - 2, 36, 4, 'F');
  cursorY += 22;

  // --- 1. RESUMO EXECUTIVO ---
  if (selection.summary && overview) {
    checkPageBreak(120);

    doc.setFillColor(...C.card);
    doc.setDrawColor(...C.border);
    doc.setLineWidth(0.5);
    doc.rect(margin, cursorY, contentWidth, 68, 'FD');
    doc.setFillColor(...C.pine);
    doc.rect(margin, cursorY, 3, 68, 'F');

    eyebrow(isEn ? 'Executive Summary & Dataset Scope' : 'Resumo Executivo & Escopo da Base', margin + 14, cursorY + 18);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(...C.ink);

    const summaryText = isEn
      ? `This report compiles bibliometric metrics, collaboration graphs, and research themes from a corpus of ${dataset.length.toLocaleString('en-US')} papers published between ${overview.summary.timespan || 'N/A'}. A total of ${overview.summary.authorsCount.toLocaleString('en-US')} authors and ${overview.summary.countriesCount.toLocaleString('en-US')} countries participated in the production.`
      : `Este relatório consolida indicadores cientométricos, redes de colaboração e tópicos de pesquisa a partir de uma base com ${dataset.length.toLocaleString('pt-BR')} documentos indexados no período ${overview.summary.timespan || 'N/A'}. A produção envolveu ${overview.summary.authorsCount.toLocaleString('pt-BR')} autores e ${overview.summary.countriesCount.toLocaleString('pt-BR')} países.`;

    doc.text(doc.splitTextToSize(summaryText, contentWidth - 28), margin + 14, cursorY + 34);
    cursorY += 80;
  }

  // --- 2. INDICADORES CIENTOMÉTRICOS PRINCIPAIS ---
  if (selection.kpis && overview) {
    checkPageBreak(120);

    sectionHeading(1, isEn ? 'Core Scientometric Indicators' : 'Indicadores Cientométricos Globais');

    const s = overview.summary;
    const loc = isEn ? 'en-US' : 'pt-BR';
    kpiCards([
      [isEn ? 'Total Documents' : 'Total de Documentos', s.totalDocs.toLocaleString(loc)],
      [isEn ? 'Total Authors' : 'Total de Autores', s.authorsCount.toLocaleString(loc)],
      [isEn ? 'Total Citations' : 'Total de Citações', totalCitations.toLocaleString(loc)],
      [isEn ? 'Citations / Doc (Mean)' : 'Citações / Doc (Média)', meanCitations.toFixed(2)],
      [isEn ? 'Annual Growth Rate' : 'Crescimento Anual', `${s.bibliometrix.growthRate.toFixed(2)}%`],
      [isEn ? 'Co-authors / Doc' : 'Coautores / Artigo', s.bibliometrix.coauthIndex.toFixed(2)],
      [isEn ? 'Unique Countries' : 'Países Únicos', s.countriesCount.toLocaleString(loc)],
      [isEn ? 'Unique Venues' : 'Periódicos (Venues)', s.venuesCount.toLocaleString(loc)],
    ]);
  }

  // --- GRÁFICO 1: EVOLUÇÃO TEMPORAL DA PRODUÇÃO ---
  if (selection.chartProduction && overview && overview.docsPerYear.length > 0) {
    checkPageBreak(210);
    const chartImg = reportProductionChart(overview.docsPerYear, locale);
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 3. TOP AUTORES ---
  if (selection.authors && tables && tables.authors.length > 0) {
    checkPageBreak(140);

    sectionHeading(2, isEn ? `Top ${topN} Authors by Production & Impact` : `Principais Autores (Top ${topN})`);

    const authorRows = tables.authors.slice(0, topN).map((a, idx) => [
      String(idx + 1),
      a.entity,
      a.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      a.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      String(a.h),
      String(a.g),
      String(a.i10),
      a.m.toFixed(2),
      a.meanCitations.toFixed(1),
    ]);

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        isEn ? 'Author' : 'Autor',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        'g',
        'i10',
        'm',
        isEn ? 'Mean Cit.' : 'Média Cit.',
      ]],
      body: authorRows,
      styles: cellStyles(7.5, 3.5),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 160 },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 2: TOP AUTORES ---
  if (selection.chartAuthors && tables && tables.authors.length > 0) {
    checkPageBreak(210);
    const chartImg = reportAuthorsChart(tables.authors, locale);
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 4. TOP PAÍSES ---
  if (selection.countries && tables && tables.countries.length > 0) {
    checkPageBreak(130);

    sectionHeading(3, isEn ? `Geographic Distribution (Top ${topN} Countries)` : `Distribuição Geográfica (Top ${topN} Países)`);

    const countryRows = tables.countries.slice(0, topN).map((c, idx) => [
      String(idx + 1),
      c.entity,
      c.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      c.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      String(c.h),
      c.meanCitations.toFixed(1),
      c.topDocument ? c.topDocument.slice(0, 50) + '...' : '—',
    ]);

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        isEn ? 'Country' : 'País',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        isEn ? 'Mean' : 'Média',
        isEn ? 'Top Document' : 'Documento Mais Citado',
      ]],
      body: countryRows,
      styles: cellStyles(7.5, 3.5),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 110 },
        6: { cellWidth: 170 },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 3: TOP PAÍSES ---
  if (selection.chartCountries && tables && tables.countries.length > 0) {
    checkPageBreak(210);
    const chartImg = reportCountriesChart(tables.countries, locale);
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- GRÁFICO 4: MAPA-MÚNDI DE COLABORAÇÃO INTERNACIONAL ---
  if (selection.chartWorldMap && collaboration && collaboration.nodes.length > 0) {
    checkPageBreak(250);
    const mapImg = reportWorldMapChart(collaboration, locale);
    if (mapImg) {
      doc.addImage(mapImg, 'PNG', margin, cursorY, contentWidth, 230);
      cursorY += 245;
    }
  }

  // --- 5. TOP VENUES ---
  if (selection.venues && tables && tables.venues.length > 0) {
    checkPageBreak(130);

    sectionHeading(4, isEn ? `Top Publishing Venues (Top ${topN})` : `Principais Veículos de Publicação (Top ${topN})`);

    const venueRows = tables.venues.slice(0, topN).map((v, idx) => [
      String(idx + 1),
      v.entity,
      v.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      v.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      String(v.h),
      v.meanCitations.toFixed(1),
    ]);

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        'Venue / Journal',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        isEn ? 'Mean Citations' : 'Média Citações',
      ]],
      body: venueRows,
      styles: cellStyles(7.5, 3.5),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 260 },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- 6. PALAVRAS-CHAVE ---
  if (selection.keywords && tables && tables.keywords.length > 0) {
    checkPageBreak(130);

    sectionHeading(5, isEn ? `Top Keywords & Lexicometrics (Top ${topN})` : `Palavras-Chave & Lexicometria (Top ${topN})`);

    const kwRows = tables.keywords.slice(0, topN).map((k, idx) => [
      String(idx + 1),
      k.entity,
      k.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      k.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      String(k.h),
      k.meanCitations.toFixed(1),
    ]);

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        isEn ? 'Keyword' : 'Palavra-chave',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        isEn ? 'Mean Citations' : 'Média de Citações',
      ]],
      body: kwRows,
      styles: cellStyles(7.5, 3.5),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 220 },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 5: NUVEM DE PALAVRAS-CHAVE ---
  if (selection.chartKeywords && tables && tables.keywords.length > 0) {
    checkPageBreak(210);
    const chartImg = reportWordCloudChart(tables.keywords, locale);
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 7. MAPEAMENTO TEMÁTICO POR IA ---
  if (selection.themes && clustering && clustering.clusters.length > 0) {
    checkPageBreak(130);

    sectionHeading(
      6,
      isEn
        ? `AI Semantic Thematic Clusters (Silhouette: ${clustering.silhouette.toFixed(3)})`
        : `Agrupamento Temático por IA (Silhouette: ${clustering.silhouette.toFixed(3)})`,
    );

    const themeRows = clustering.clusters.map((c) => {
      const share = dataset.length > 0 ? (c.size / dataset.length) * 100 : 0;
      return [
        String(c.clusterId + 1),
        `Tema ${c.clusterId + 1}`,
        c.size.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        `${share.toFixed(1)}%`,
        c.topTerms.slice(0, 5).join(', '),
      ];
    });

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        isEn ? 'Theme Name' : 'Nome do Tema',
        'Docs',
        '% Share',
        isEn ? 'Key Terms' : 'Termos Característicos',
      ]],
      body: themeRows,
      styles: cellStyles(7.5, 4),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 160 },
        2: { cellWidth: 45 },
        3: { cellWidth: 50 },
        4: { cellWidth: 'auto' },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 6: DISTRIBUIÇÃO DE TEMAS ---
  if (selection.chartThemes && clustering && clustering.clusters.length > 0) {
    checkPageBreak(210);
    const chartImg = reportThemesChart(clustering.clusters, dataset.length, locale);
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 8. TOPOLOGIA DA REDE & ECOLOGIA PROFUNDA ---
  if (selection.networkTopology && sna) {
    checkPageBreak(140);

    sectionHeading(7, isEn ? 'Deep Knowledge Ecology & Network Topology' : 'Topologia da Rede & Ecologia Profunda');

    const g = sna.global;
    const snaRows = [
      [
        { content: isEn ? 'Density' : 'Densidade', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.density, 4),
        { content: isEn ? 'Avg Clustering' : 'Clustering Médio', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.clustering, 4),
      ],
      [
        { content: isEn ? 'Shannon Entropy' : 'Entropia de Shannon', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.entropy, 3),
        { content: isEn ? 'Global Efficiency' : 'Eficiência Global', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.efficiency, 4),
      ],
      [
        { content: isEn ? 'Mean Degree' : 'Grau Médio', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.meanDegree, 2),
        { content: isEn ? 'Degree Std Dev' : 'Desvio do Grau', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.stdDegree, 2),
      ],
      [
        { content: isEn ? 'Mean PageRank' : 'PageRank Médio', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.meanPageRank, 4),
        { content: isEn ? 'Assortativity' : 'Assortatividade', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.assortativity, 3),
      ],
      [
        { content: isEn ? 'Power Law Exponent' : 'Lei de Potência (Expoente)', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.powerLawExponent, 2),
        { content: isEn ? 'Degree×Betweenness Corr' : 'Spearman Grau×Ponte', styles: { fontStyle: 'bold' as const } },
        formatMetricVal(g.spearmanDegreeBetweenness, 3),
      ],
    ];

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      body: snaRows,
      styles: cellStyles(8, 4),
      alternateRowStyles: {},
      columnStyles: {
        0: { fillColor: C.muted, cellWidth: 150 },
        1: { cellWidth: 90 },
        2: { fillColor: C.muted, cellWidth: 150 },
        3: { cellWidth: 'auto' },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 7: REDE DE COOCORRÊNCIA (GRAFOS) ---
  if (selection.chartNetwork && network && network.nodes.length > 0) {
    checkPageBreak(260);
    const netImg = reportNetworkChart(network.nodes, network.edges, locale);
    if (netImg) {
      doc.addImage(netImg, 'PNG', margin, cursorY, contentWidth, 240);
      cursorY += 255;
    }
  }

  // --- 9. TOP DOCUMENTOS MAIS CITADOS ---
  if (selection.topDocuments && dataset.length > 0) {
    checkPageBreak(150);

    sectionHeading(8, isEn ? `Highly Cited Seminal Documents (Top ${topN})` : `Documentos Mais Citados da Base (Top ${topN})`);

    const columns = collectColumns(dataset);
    const titleCol = pickColumn(columns, FIELD_CANDIDATES.title);
    const authCol = pickColumn(columns, FIELD_CANDIDATES.authors);
    const venueCol = pickColumn(columns, FIELD_CANDIDATES.venue);

    const sortedDocs = [...dataset]
      .sort((a, b) => (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0))
      .slice(0, topN);

    const docRows = sortedDocs.map((d, idx) => [
      String(idx + 1),
      titleCol ? String(d[titleCol] ?? '').slice(0, 75) + '...' : '—',
      authCol ? String(d[authCol] ?? '').slice(0, 30) : '—',
      String(toNumeric(d[FIELD.YEAR_CLEAN]) ?? '—'),
      String(toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0),
      venueCol ? String(d[venueCol] ?? '').slice(0, 35) : '—',
    ]);

    autoTable(doc, {
      startY: cursorY,
      ...tableBase,
      head: [[
        '#',
        isEn ? 'Title' : 'Título',
        isEn ? 'Authors' : 'Autores',
        isEn ? 'Year' : 'Ano',
        isEn ? 'Cit.' : 'Cit.',
        'Venue',
      ]],
      body: docRows,
      styles: cellStyles(7.5, 3.5),
      columnStyles: {
        0: { cellWidth: 20 },
        1: { cellWidth: 180 },
        2: { cellWidth: 90 },
        3: { cellWidth: 35 },
        4: { cellWidth: 35 },
        5: { cellWidth: 'auto' },
      },
    });
  }

  // --- NUMERAÇÃO DE PÁGINAS E RODAPÉ ---
  const totalPages = doc.getNumberOfPages();
  const footerText = isEn
    ? 'Simetrics · A Scientata application · scientata.com'
    : 'Simetrics · Uma aplicação Scientata · scientata.com';
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);

    doc.setDrawColor(...C.border);
    doc.setLineWidth(0.5);
    doc.line(margin, pageHeight - 25, pageWidth - margin, pageHeight - 25);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...C.inkMuted);
    doc.text(footerText, margin, pageHeight - 14);

    doc.setFont('courier', 'normal');
    doc.setFontSize(7.5);
    const pageStr = isEn ? `Page ${i} of ${totalPages}` : `Página ${i} de ${totalPages}`;
    doc.text(pageStr, pageWidth - margin - doc.getTextWidth(pageStr), pageHeight - 14);
  }

  const filename = `simetrics-relatorio-${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}
