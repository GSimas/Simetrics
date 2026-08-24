import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  WidthType,
  BorderStyle,
  AlignmentType,
  ShadingType,
  ImageRun,
} from 'docx';

import type { AnalyticsBundle, EntityTables } from '@/workers/analytics.worker';
import type { CooccurrenceReport, SnaReport } from '@/core/graph';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import type { ClusteringResult } from '@/core/clustering';
import type { Dataset } from '@/lib/types';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import type { ReportSectionsSelection } from './pdf-generator';
import {
  renderHorizontalBarChart,
  renderNetworkGraphCanvas,
  renderProductionTimelineCanvas,
  renderThemesPieChart,
  renderWordCloudCanvas,
  renderWorldCollaborationMapCanvas,
} from './chart-renderer';

export interface DocxReportData {
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

function formatVal(val: number | string, decimals = 4): string {
  if (typeof val === 'string') return val;
  if (!Number.isFinite(val)) return '—';
  return val.toFixed(decimals);
}

function dataUrlToUint8Array(dataUrl: string): Uint8Array {
  const base64 = dataUrl.split(',')[1] ?? '';
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export async function generateDocxReport({
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
}: DocxReportData): Promise<void> {
  const isEn = locale === 'en';
  const sectionsChildren: (Paragraph | Table)[] = [];

  const primaryHex = '2563EB'; // Blue
  const darkHex = '0F172A';
  const tableHeaderBg = 'E2E8F0';

  const totalCitations = dataset.reduce((acc, d) => acc + (toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0), 0);
  const meanCitations = dataset.length > 0 ? totalCitations / dataset.length : 0;

  // --- TÍTULO DO RELATÓRIO ---
  sectionsChildren.push(
    new Paragraph({
      heading: HeadingLevel.TITLE,
      children: [
        new TextRun({
          text: isEn
            ? 'SIMETRICS — Scientometric Intelligence Report'
            : 'SIMETRICS — Relatório Cientométrico & Bibliométrico',
          bold: true,
          size: 36,
          color: primaryHex,
        }),
      ],
    }),
    new Paragraph({
      children: [
        new TextRun({
          text: `${isEn ? 'Generated on' : 'Emitido em'}: ${new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', {
            day: '2-digit',
            month: 'long',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })} · Desenvolvido por Gustavo Simas (gustavosimas.com)`,
          italics: true,
          size: 18,
          color: '64748B',
        }),
      ],
      spacing: { after: 300 },
    }),
  );

  // --- 1. RESUMO EXECUTIVO ---
  if (selection.summary && overview) {
    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? 'Executive Summary & Dataset Scope' : 'Resumo Executivo & Escopo da Base',
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 200, after: 100 },
      }),
      new Paragraph({
        children: [
          new TextRun({
            text: isEn
              ? `This document presents a comprehensive bibliometric and scientometric evaluation based on ${dataset.length.toLocaleString('en-US')} publications indexed between ${overview.summary.timespan || 'N/A'}. In total, ${overview.summary.authorsCount.toLocaleString('en-US')} distinct authors and ${overview.summary.countriesCount.toLocaleString('en-US')} countries contributed to the corpus.`
              : `Este documento apresenta uma avaliação bibliométrica e cientométrica detalhada a partir de ${dataset.length.toLocaleString('pt-BR')} publicações indexadas entre ${overview.summary.timespan || 'N/A'}. Ao todo, ${overview.summary.authorsCount.toLocaleString('pt-BR')} autores distintos e ${overview.summary.countriesCount.toLocaleString('pt-BR')} países contribuíram com a produção acadêmica.`,
            size: 20,
          }),
        ],
        spacing: { after: 200 },
      }),
    );
  }

  // --- 2. INDICADORES GLOBAIS ---
  if (selection.kpis && overview) {
    const s = overview.summary;
    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? '1. Core Scientometric Indicators' : '1. Indicadores Cientométricos Globais',
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 200, after: 100 },
      }),
      createWordTable(
        [
          [isEn ? 'Metric' : 'Métrica', isEn ? 'Value' : 'Valor', isEn ? 'Metric' : 'Métrica', isEn ? 'Value' : 'Valor'],
          [
            isEn ? 'Total Documents' : 'Total de Documentos',
            s.totalDocs.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
            isEn ? 'Total Authors' : 'Total de Autores',
            s.authorsCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
          ],
          [
            isEn ? 'Total Citations' : 'Total de Citações',
            totalCitations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
            isEn ? 'Mean Citations/Doc' : 'Média de Citações/Doc',
            meanCitations.toFixed(2),
          ],
          [
            isEn ? 'Annual Growth Rate' : 'Crescimento Anual',
            `${(s.bibliometrix.growthRate * 100).toFixed(2)}%`,
            isEn ? 'Co-authors / Doc' : 'Coautores / Artigo',
            s.bibliometrix.coauthIndex.toFixed(2),
          ],
          [
            isEn ? 'Unique Countries' : 'Países Únicos',
            s.countriesCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
            isEn ? 'Unique Venues' : 'Periódicos (Venues)',
            s.venuesCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
          ],
        ],
        [2800, 1710, 2800, 1710],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 1: EVOLUÇÃO TEMPORAL DA PRODUÇÃO ---
  if (selection.chartProduction && overview && overview.docsPerYear.length > 0) {
    const chartPng = renderProductionTimelineCanvas(overview.docsPerYear, { width: 1000, height: 420, locale });
    if (chartPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(chartPng),
              transformation: { width: 560, height: 235 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 3. TOP AUTORES ---
  if (selection.authors && tables && tables.authors.length > 0) {
    const authorRows: string[][] = [
      ['#', isEn ? 'Author' : 'Autor', 'Docs', isEn ? 'Citations' : 'Citações', 'h', 'g', 'i10', 'm', isEn ? 'Mean Cit.' : 'Média Cit.'],
      ...tables.authors.slice(0, topN).map((a, idx) => [
        String(idx + 1),
        a.entity,
        a.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        a.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        String(a.h),
        String(a.g),
        String(a.i10),
        a.m.toFixed(2),
        a.meanCitations.toFixed(1),
      ]),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? `2. Top ${topN} Authors by Production & Impact` : `2. Principais Autores (Top ${topN})`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        authorRows,
        [500, 2920, 800, 1000, 600, 600, 600, 700, 1300],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 2: TOP AUTORES ---
  if (selection.chartAuthors && tables && tables.authors.length > 0) {
    const authorItems = tables.authors.slice(0, 10).map((a) => ({
      label: a.entity,
      value: a.docCount,
      sub: `${a.citations} ${isEn ? 'cit.' : 'cit.'}`,
    }));
    const chartTitle = isEn ? 'Top 10 Most Prolific Authors (Published Papers)' : 'Top 10 Autores Mais Produtivos';
    const chartPng = renderHorizontalBarChart(chartTitle, authorItems, {
      width: 1000,
      height: 420,
      locale,
    });
    if (chartPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(chartPng),
              transformation: { width: 560, height: 235 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 4. TOP PAÍSES ---
  if (selection.countries && tables && tables.countries.length > 0) {
    const countryRows: string[][] = [
      ['#', isEn ? 'Country' : 'País', 'Docs', isEn ? 'Citations' : 'Citações', 'h', isEn ? 'Mean Cit.' : 'Média Cit.'],
      ...tables.countries.slice(0, topN).map((c, idx) => [
        String(idx + 1),
        c.entity,
        c.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        c.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        String(c.h),
        c.meanCitations.toFixed(1),
      ]),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? `3. Geographic Distribution (Top ${topN} Countries)` : `3. Distribuição Geográfica (Top ${topN} Países)`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        countryRows,
        [600, 3420, 1200, 1400, 900, 1500],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 3: TOP PAÍSES ---
  if (selection.chartCountries && tables && tables.countries.length > 0) {
    const countryItems = tables.countries.slice(0, 10).map((c) => ({
      label: c.entity,
      value: c.docCount,
      sub: `${c.citations} ${isEn ? 'cit.' : 'cit.'}`,
    }));
    const chartTitle = isEn ? 'Top 10 Leading Countries by Scientific Output' : 'Top 10 Países com Maior Produção Científica';
    const chartPng = renderHorizontalBarChart(chartTitle, countryItems, {
      width: 1000,
      height: 420,
      locale,
    });
    if (chartPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(chartPng),
              transformation: { width: 560, height: 235 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- GRÁFICO 4: MAPA-MÚNDI DE COLABORAÇÃO INTERNACIONAL ---
  if (selection.chartWorldMap && collaboration && collaboration.nodes.length > 0) {
    const mapPng = renderWorldCollaborationMapCanvas(collaboration, {
      width: 1000,
      height: 500,
      locale,
    });
    if (mapPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(mapPng),
              transformation: { width: 560, height: 280 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 5. TOP VENUES ---
  if (selection.venues && tables && tables.venues.length > 0) {
    const venueRows: string[][] = [
      ['#', 'Venue / Journal', 'Docs', isEn ? 'Citations' : 'Citações', 'h', isEn ? 'Mean Cit.' : 'Média Cit.'],
      ...tables.venues.slice(0, topN).map((v, idx) => [
        String(idx + 1),
        v.entity,
        v.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        v.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        String(v.h),
        v.meanCitations.toFixed(1),
      ]),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? `4. Top Publishing Venues (Top ${topN})` : `4. Principais Veículos de Publicação (Top ${topN})`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        venueRows,
        [600, 4120, 1100, 1200, 800, 1200],
        tableHeaderBg,
      ),
    );
  }

  // --- 6. PALAVRAS-CHAVE ---
  if (selection.keywords && tables && tables.keywords.length > 0) {
    const kwRows: string[][] = [
      ['#', isEn ? 'Keyword' : 'Palavra-chave', 'Docs', isEn ? 'Citations' : 'Citações', 'h', isEn ? 'Mean Cit.' : 'Média Cit.'],
      ...tables.keywords.slice(0, topN).map((k, idx) => [
        String(idx + 1),
        k.entity,
        k.docCount.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        k.citations.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
        String(k.h),
        k.meanCitations.toFixed(1),
      ]),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? `5. Top Keywords & Lexicometrics (Top ${topN})` : `5. Palavras-Chave & Lexicometria (Top ${topN})`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        kwRows,
        [600, 3920, 1100, 1200, 800, 1400],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 5: NUVEM DE PALAVRAS-CHAVE ---
  if (selection.chartKeywords && tables && tables.keywords.length > 0) {
    const chartPng = renderWordCloudCanvas(tables.keywords, { width: 1000, height: 420, locale });
    if (chartPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(chartPng),
              transformation: { width: 560, height: 235 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 7. CLUSTERS TEMÁTICOS ---
  if (selection.themes && clustering && clustering.clusters.length > 0) {
    const themeRows: string[][] = [
      ['#', isEn ? 'Theme Name' : 'Nome do Tema', 'Docs', '% Share', isEn ? 'Key Terms' : 'Termos Característicos'],
      ...clustering.clusters.map((c) => {
        const share = dataset.length > 0 ? (c.size / dataset.length) * 100 : 0;
        return [
          String(c.clusterId + 1),
          `Tema ${c.clusterId + 1}`,
          c.size.toLocaleString(isEn ? 'en-US' : 'pt-BR'),
          `${share.toFixed(1)}%`,
          c.topTerms.slice(0, 6).join(', '),
        ];
      }),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn
              ? `6. AI Semantic Thematic Clusters (Silhouette Score: ${clustering.silhouette.toFixed(3)})`
              : `6. Agrupamento Temático por IA (Score Silhouette: ${clustering.silhouette.toFixed(3)})`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        themeRows,
        [600, 2420, 1000, 1000, 4000],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 6: DISTRIBUIÇÃO DE TEMAS ---
  if (selection.chartThemes && clustering && clustering.clusters.length > 0) {
    const themeItems = clustering.clusters.map((c) => ({
      clusterId: c.clusterId,
      name: `Tema ${c.clusterId + 1}`,
      docCount: c.size,
      share: dataset.length > 0 ? (c.size / dataset.length) * 100 : 0,
    }));
    const chartPng = renderThemesPieChart(themeItems, { width: 1000, height: 420, locale });
    if (chartPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(chartPng),
              transformation: { width: 560, height: 235 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 8. TOPOLOGIA DA REDE ---
  if (selection.networkTopology && sna) {
    const g = sna.global;
    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? '7. Deep Knowledge Ecology & Network Topology' : '7. Topologia da Rede & Ecologia Profunda',
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        [
          [isEn ? 'Topology Metric' : 'Métrica Topológica', isEn ? 'Value' : 'Valor', isEn ? 'Topology Metric' : 'Métrica Topológica', isEn ? 'Value' : 'Valor'],
          [isEn ? 'Density' : 'Densidade', formatVal(g.density, 4), isEn ? 'Avg Clustering' : 'Clustering Médio', formatVal(g.clustering, 4)],
          [isEn ? 'Shannon Entropy' : 'Entropia de Shannon', formatVal(g.entropy, 3), isEn ? 'Global Efficiency' : 'Eficiência Global', formatVal(g.efficiency, 4)],
          [isEn ? 'Mean Degree' : 'Grau Médio', formatVal(g.meanDegree, 2), isEn ? 'Degree Std Dev' : 'Desvio do Grau', formatVal(g.stdDegree, 2)],
          [isEn ? 'Mean PageRank' : 'PageRank Médio', formatVal(g.meanPageRank, 4), isEn ? 'Assortativity' : 'Assortatividade', formatVal(g.assortativity, 3)],
          [isEn ? 'Power Law Exponent' : 'Lei de Potência', formatVal(g.powerLawExponent, 2), isEn ? 'Degree×Betweenness' : 'Spearman Grau×Ponte', formatVal(g.spearmanDegreeBetweenness, 3)],
        ],
        [2800, 1710, 2800, 1710],
        tableHeaderBg,
      ),
    );
  }

  // --- GRÁFICO 7: REDE DE COOCORRÊNCIA (GRAFOS) ---
  if (selection.chartNetwork && network && network.nodes.length > 0) {
    const netPng = renderNetworkGraphCanvas(network.nodes, network.edges, {
      width: 1000,
      height: 520,
      locale,
    });
    if (netPng) {
      sectionsChildren.push(
        new Paragraph({
          children: [
            new ImageRun({
              data: dataUrlToUint8Array(netPng),
              transformation: { width: 560, height: 290 },
              type: 'png',
            }),
          ],
          spacing: { before: 200, after: 200 },
        }),
      );
    }
  }

  // --- 9. TOP DOCUMENTOS ---
  if (selection.topDocuments && dataset.length > 0) {
    const columns = collectColumns(dataset);
    const titleCol = pickColumn(columns, FIELD_CANDIDATES.title);
    const authCol = pickColumn(columns, FIELD_CANDIDATES.authors);
    const venueCol = pickColumn(columns, FIELD_CANDIDATES.venue);

    const sortedDocs = [...dataset]
      .sort((a, b) => (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0))
      .slice(0, topN);

    const docRows: string[][] = [
      ['#', isEn ? 'Title' : 'Título', isEn ? 'Authors' : 'Autores', isEn ? 'Year' : 'Ano', isEn ? 'Citations' : 'Citações', 'Venue'],
      ...sortedDocs.map((d, idx) => [
        String(idx + 1),
        titleCol ? String(d[titleCol] ?? '').slice(0, 80) : '—',
        authCol ? String(d[authCol] ?? '').slice(0, 30) : '—',
        String(toNumeric(d[FIELD.YEAR_CLEAN]) ?? '—'),
        String(toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0),
        venueCol ? String(d[venueCol] ?? '').slice(0, 40) : '—',
      ]),
    ];

    sectionsChildren.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [
          new TextRun({
            text: isEn ? `8. Highly Cited Seminal Documents (Top ${topN})` : `8. Documentos Mais Citados da Base (Top ${topN})`,
            bold: true,
            color: darkHex,
          }),
        ],
        spacing: { before: 240, after: 100 },
      }),
      createWordTable(
        docRows,
        [500, 3620, 1900, 700, 800, 1500],
        tableHeaderBg,
      ),
    );
  }

  // Monta o arquivo Docx com margens e dimensões A4 exatas
  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: 1440,
              bottom: 1440,
              left: 1440,
              right: 1440,
            },
          },
        },
        children: sectionsChildren,
      },
    ],
  });

  const blob = await Packer.toBlob(doc);
  const filename = `simetrics-relatorio-${new Date().toISOString().slice(0, 10)}.docx`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function createWordTable(data: string[][], colWidthsDxa: number[], headerBgHex: string): Table {
  const totalTableWidth = colWidthsDxa.reduce((acc, w) => acc + w, 0);

  return new Table({
    columnWidths: colWidthsDxa,
    width: { size: totalTableWidth, type: WidthType.DXA },
    rows: data.map((row, rowIndex) => {
      const isHeader = rowIndex === 0;
      return new TableRow({
        tableHeader: isHeader,
        cantSplit: true,
        children: row.map((cellText, colIndex) => {
          const colWidth = colWidthsDxa[colIndex] ?? 1000;
          const borders = {
            top: { style: BorderStyle.SINGLE, size: 1, color: 'CBD5E1' },
            bottom: { style: BorderStyle.SINGLE, size: 1, color: 'CBD5E1' },
            left: { style: BorderStyle.SINGLE, size: 1, color: 'CBD5E1' },
            right: { style: BorderStyle.SINGLE, size: 1, color: 'CBD5E1' },
          };

          const paragraph = new Paragraph({
            alignment: isHeader ? AlignmentType.LEFT : AlignmentType.LEFT,
            children: [
              new TextRun({
                text: cellText,
                bold: isHeader,
                size: 17,
                color: isHeader ? '0F172A' : '334155',
              }),
            ],
          });

          if (isHeader) {
            return new TableCell({
              width: { size: colWidth, type: WidthType.DXA },
              borders,
              shading: { type: ShadingType.CLEAR, fill: headerBgHex },
              children: [paragraph],
            });
          }

          return new TableCell({
            width: { size: colWidth, type: WidthType.DXA },
            borders,
            children: [paragraph],
          });
        }),
      });
    }),
  });
}
