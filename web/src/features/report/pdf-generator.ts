import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

import type { AnalyticsBundle, EntityTables } from '@/workers/analytics.worker';
import type { CooccurrenceReport, SnaReport } from '@/core/graph';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import type { ClusteringResult } from '@/core/clustering';
import type { Dataset } from '@/lib/types';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import {
  renderHorizontalBarChart,
  renderNetworkGraphCanvas,
  renderProductionTimelineCanvas,
  renderThemesPieChart,
  renderWordCloudCanvas,
  renderWorldCollaborationMapCanvas,
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

  const primaryColor: [number, number, number] = [37, 99, 235]; // Blue 600
  const secondaryColor: [number, number, number] = [15, 23, 42]; // Slate 900
  const mutedColor: [number, number, number] = [100, 116, 139]; // Slate 500
  const accentColor: [number, number, number] = [124, 58, 237]; // Purple 600

  const checkPageBreak = (neededHeight: number) => {
    if (cursorY + neededHeight > pageHeight - margin - 30) {
      doc.addPage();
      cursorY = margin + 20;
    }
  };

  const totalCitations = dataset.reduce((acc, d) => acc + (toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0), 0);
  const meanCitations = dataset.length > 0 ? totalCitations / dataset.length : 0;

  // --- CABEÇALHO DO RELATÓRIO ---
  doc.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.rect(margin, cursorY, contentWidth, 4, 'F');
  cursorY += 16;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
  doc.text(
    isEn ? 'SIMETRICS — Scientometric Intelligence Report' : 'SIMETRICS — Relatório Cientométrico & Bibliométrico',
    margin,
    cursorY,
  );
  cursorY += 16;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);
  const dateStr = new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  doc.text(
    `${isEn ? 'Generated on' : 'Emitido em'}: ${dateStr} · Plataforma Simetrics (gustavosimas.com)`,
    margin,
    cursorY,
  );
  cursorY += 16;

  // --- 1. RESUMO EXECUTIVO ---
  if (selection.summary && overview) {
    checkPageBreak(120);

    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(226, 232, 240);
    doc.roundedRect(margin, cursorY, contentWidth, 68, 6, 6, 'FD');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    doc.text(isEn ? 'Executive Summary & Dataset Scope' : 'Resumo Executivo & Escopo da Base', margin + 12, cursorY + 18);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);

    const summaryText = isEn
      ? `This report compiles bibliometric metrics, collaboration graphs, and research themes from a corpus of ${dataset.length.toLocaleString('en-US')} papers published between ${overview.summary.timespan || 'N/A'}. A total of ${overview.summary.authorsCount.toLocaleString('en-US')} authors and ${overview.summary.countriesCount.toLocaleString('en-US')} countries participated in the production.`
      : `Este relatório consolida indicadores cientométricos, redes de colaboração e tópicos de pesquisa a partir de uma base com ${dataset.length.toLocaleString('pt-BR')} documentos indexados no período ${overview.summary.timespan || 'N/A'}. A produção envolveu ${overview.summary.authorsCount.toLocaleString('pt-BR')} autores e ${overview.summary.countriesCount.toLocaleString('pt-BR')} países.`;

    doc.text(doc.splitTextToSize(summaryText, contentWidth - 24), margin + 12, cursorY + 34);
    cursorY += 80;
  }

  // --- 2. INDICADORES CIENTOMÉTRICOS PRINCIPAIS ---
  if (selection.kpis && overview) {
    checkPageBreak(120);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(isEn ? '1. Core Scientometric Indicators' : '1. Indicadores Cientométricos Globais', margin, cursorY);
    cursorY += 14;

    const s = overview.summary;
    const kpiData = [
      [
        { content: isEn ? 'Total Documents' : 'Total de Documentos', styles: { fontStyle: 'bold' as const } },
        s.totalDocs.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        { content: isEn ? 'Total Authors' : 'Total de Autores', styles: { fontStyle: 'bold' as const } },
        s.authorsCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      ],
      [
        { content: isEn ? 'Total Citations' : 'Total de Citações', styles: { fontStyle: 'bold' as const } },
        totalCitations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        { content: isEn ? 'Citations / Doc (Mean)' : 'Citações / Doc (Média)', styles: { fontStyle: 'bold' as const } },
        meanCitations.toFixed(2),
      ],
      [
        { content: isEn ? 'Annual Growth Rate' : 'Crescimento Anual', styles: { fontStyle: 'bold' as const } },
        `${(s.bibliometrix.growthRate * 100).toFixed(2)}%`,
        { content: isEn ? 'Co-authors / Doc' : 'Coautores / Artigo', styles: { fontStyle: 'bold' as const } },
        s.bibliometrix.coauthIndex.toFixed(2),
      ],
      [
        { content: isEn ? 'Unique Countries' : 'Países Únicos', styles: { fontStyle: 'bold' as const } },
        s.countriesCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        { content: isEn ? 'Unique Venues' : 'Periódicos (Venues)', styles: { fontStyle: 'bold' as const } },
        s.venuesCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
      ],
    ];

    autoTable(doc, {
      startY: cursorY,
      margin: { left: margin, right: margin },
      body: kpiData,
      theme: 'grid',
      styles: { fontSize: 8.5, cellPadding: 4.5, textColor: secondaryColor },
      columnStyles: {
        0: { fillColor: [241, 245, 249], cellWidth: 140 },
        1: { cellWidth: 100 },
        2: { fillColor: [241, 245, 249], cellWidth: 140 },
        3: { cellWidth: 'auto' },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 1: EVOLUÇÃO TEMPORAL DA PRODUÇÃO ---
  if (selection.chartProduction && overview && overview.docsPerYear.length > 0) {
    checkPageBreak(210);
    const chartImg = renderProductionTimelineCanvas(overview.docsPerYear, { width: 1000, height: 420, locale });
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 3. TOP AUTORES ---
  if (selection.authors && tables && tables.authors.length > 0) {
    checkPageBreak(140);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(isEn ? `2. Top ${topN} Authors by Production & Impact` : `2. Principais Autores (Top ${topN})`, margin, cursorY);
    cursorY += 10;

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
      margin: { left: margin, right: margin },
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
      theme: 'striped',
      headStyles: { fillColor: primaryColor, fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 3.5 },
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
    const authorItems = tables.authors.slice(0, 10).map((a) => ({
      label: a.entity,
      value: a.docCount,
      sub: `${a.citations} cit. | h=${a.h}`,
    }));
    const chartTitle = isEn ? 'Top 10 Most Prolific Authors (Published Papers)' : 'Top 10 Autores Mais Produtivos (Artigos Publicados)';
    const chartImg = renderHorizontalBarChart(chartTitle, authorItems, {
      width: 1000,
      height: 420,
      locale,
    });
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 4. TOP PAÍSES ---
  if (selection.countries && tables && tables.countries.length > 0) {
    checkPageBreak(130);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(isEn ? `3. Geographic Distribution (Top ${topN} Countries)` : `3. Distribuição Geográfica (Top ${topN} Países)`, margin, cursorY);
    cursorY += 10;

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
      margin: { left: margin, right: margin },
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
      theme: 'striped',
      headStyles: { fillColor: [30, 41, 59], fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 3.5 },
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
    const countryItems = tables.countries.slice(0, 10).map((c) => ({
      label: c.entity,
      value: c.docCount,
      sub: `${c.citations} cit.`,
    }));
    const chartTitle = isEn ? 'Top 10 Leading Countries by Scientific Output' : 'Top 10 Países com Maior Produção Científica';
    const chartImg = renderHorizontalBarChart(chartTitle, countryItems, {
      width: 1000,
      height: 420,
      locale,
    });
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- GRÁFICO 4: MAPA-MÚNDI DE COLABORAÇÃO INTERNACIONAL ---
  if (selection.chartWorldMap && collaboration && collaboration.nodes.length > 0) {
    checkPageBreak(250);
    const mapImg = renderWorldCollaborationMapCanvas(collaboration, {
      width: 1000,
      height: 500,
      locale,
    });
    if (mapImg) {
      doc.addImage(mapImg, 'PNG', margin, cursorY, contentWidth, 230);
      cursorY += 245;
    }
  }

  // --- 5. TOP VENUES ---
  if (selection.venues && tables && tables.venues.length > 0) {
    checkPageBreak(130);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(isEn ? `4. Top Publishing Venues (Top ${topN})` : `4. Principais Veículos de Publicação (Top ${topN})`, margin, cursorY);
    cursorY += 10;

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
      margin: { left: margin, right: margin },
      head: [[
        '#',
        'Venue / Journal',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        isEn ? 'Mean Citations' : 'Média Citações',
      ]],
      body: venueRows,
      theme: 'striped',
      headStyles: { fillColor: [13, 148, 136], fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 3.5 },
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

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(isEn ? `5. Top Keywords & Lexicometrics (Top ${topN})` : `5. Palavras-Chave & Lexicometria (Top ${topN})`, margin, cursorY);
    cursorY += 10;

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
      margin: { left: margin, right: margin },
      head: [[
        '#',
        isEn ? 'Keyword' : 'Palavra-chave',
        'Docs',
        isEn ? 'Citations' : 'Citações',
        'h',
        isEn ? 'Mean Citations' : 'Média de Citações',
      ]],
      body: kwRows,
      theme: 'striped',
      headStyles: { fillColor: [8, 145, 178], fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 3.5 },
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
    const chartImg = renderWordCloudCanvas(tables.keywords, { width: 1000, height: 420, locale });
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 7. MAPEAMENTO TEMÁTICO POR IA ---
  if (selection.themes && clustering && clustering.clusters.length > 0) {
    checkPageBreak(130);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(
      isEn
        ? `6. AI Semantic Thematic Clusters (Silhouette: ${clustering.silhouette.toFixed(3)})`
        : `6. Agrupamento Temático por IA (Silhouette: ${clustering.silhouette.toFixed(3)})`,
      margin,
      cursorY,
    );
    cursorY += 10;

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
      margin: { left: margin, right: margin },
      head: [[
        '#',
        isEn ? 'Theme Name' : 'Nome do Tema',
        'Docs',
        '% Share',
        isEn ? 'Key Terms' : 'Termos Característicos',
      ]],
      body: themeRows,
      theme: 'striped',
      headStyles: { fillColor: accentColor, fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 4 },
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
    const themeItems = clustering.clusters.map((c) => ({
      clusterId: c.clusterId,
      name: `Tema ${c.clusterId + 1}`,
      docCount: c.size,
      share: dataset.length > 0 ? (c.size / dataset.length) * 100 : 0,
    }));
    const chartImg = renderThemesPieChart(themeItems, { width: 1000, height: 420, locale });
    if (chartImg) {
      doc.addImage(chartImg, 'PNG', margin, cursorY, contentWidth, 200);
      cursorY += 215;
    }
  }

  // --- 8. TOPOLOGIA DA REDE & ECOLOGIA PROFUNDA ---
  if (selection.networkTopology && sna) {
    checkPageBreak(140);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(
      isEn ? '7. Deep Knowledge Ecology & Network Topology' : '7. Topologia da Rede & Ecologia Profunda',
      margin,
      cursorY,
    );
    cursorY += 10;

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
      margin: { left: margin, right: margin },
      body: snaRows,
      theme: 'grid',
      styles: { fontSize: 8, cellPadding: 4, textColor: secondaryColor },
      columnStyles: {
        0: { fillColor: [245, 243, 255], cellWidth: 150 },
        1: { cellWidth: 90 },
        2: { fillColor: [245, 243, 255], cellWidth: 150 },
        3: { cellWidth: 'auto' },
      },
    });

    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  // --- GRÁFICO 7: REDE DE COOCORRÊNCIA (GRAFOS) ---
  if (selection.chartNetwork && network && network.nodes.length > 0) {
    checkPageBreak(260);
    const netImg = renderNetworkGraphCanvas(network.nodes, network.edges, {
      width: 1000,
      height: 520,
      locale,
    });
    if (netImg) {
      doc.addImage(netImg, 'PNG', margin, cursorY, contentWidth, 240);
      cursorY += 255;
    }
  }

  // --- 9. TOP DOCUMENTOS MAIS CITADOS ---
  if (selection.topDocuments && dataset.length > 0) {
    checkPageBreak(150);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
    doc.text(
      isEn ? `8. Highly Cited Seminal Documents (Top ${topN})` : `8. Documentos Mais Citados da Base (Top ${topN})`,
      margin,
      cursorY,
    );
    cursorY += 10;

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
      margin: { left: margin, right: margin },
      head: [[
        '#',
        isEn ? 'Title' : 'Título',
        isEn ? 'Authors' : 'Autores',
        isEn ? 'Year' : 'Ano',
        isEn ? 'Cit.' : 'Cit.',
        'Venue',
      ]],
      body: docRows,
      theme: 'striped',
      headStyles: { fillColor: [30, 41, 59], fontSize: 8, fontStyle: 'bold' },
      styles: { fontSize: 7.5, cellPadding: 3.5 },
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
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(mutedColor[0], mutedColor[1], mutedColor[2]);

    doc.setDrawColor(226, 232, 240);
    doc.line(margin, pageHeight - 25, pageWidth - margin, pageHeight - 25);

    doc.text('Simetrics · Plataforma de Inteligência Bibliométrica (gustavosimas.com)', margin, pageHeight - 14);
    const pageStr = isEn ? `Page ${i} of ${totalPages}` : `Página ${i} de ${totalPages}`;
    doc.text(pageStr, pageWidth - margin - doc.getTextWidth(pageStr), pageHeight - 14);
  }

  const filename = `simetrics-relatorio-${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}
