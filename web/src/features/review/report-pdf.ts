import jsPDF from 'jspdf';
import autoTable, { type UserOptions } from 'jspdf-autotable';

import { finalSelection, hasQualityChecklist, includedStudies, maxQualityScore, scoreStudy } from '@/core/review/quality';
import { buildSearchString, SEARCH_TARGETS } from '@/core/review/search-string';
import { formatExtractionValue } from '@/core/review/synthesis';
import { FRAMEWORK_FIELDS, REVIEW_DEFAULTS } from '@/core/review/types';
import { BRAND } from '@/features/report/chart-renderer';
import { buildReportModel, fullTextParagraphs, quotesCell, type ReportAnswer } from './report-model';
import { fillText, REPORT_TEXT, reference, shortLabel, type ReviewReportInput } from './report-shared';

/**
 * As fontes padrão do PDF só têm o alfabeto Windows-1252. Trechos de artigos trazem
 * ligaduras (ﬁ), letras gregas, símbolos: ligaduras e acentos compostos se decompõem; o
 * que não tem equivalente vira "?" em vez de um glifo quebrado.
 */
const WIN_ANSI_EXTRA = new Set([...'€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ']);
export function pdfSafe(value: string): string {
  return [...value.normalize('NFKC')]
    .map((char) => {
      // A normalização troca o micro (µ, que a fonte tem) pelo mu grego (que não tem).
      if (char === 'μ') return 'µ';
      const code = char.codePointAt(0)!;
      if (char === '\n' || (code >= 0x20 && code <= 0x7e) || (code >= 0xa0 && code <= 0xff) || WIN_ANSI_EXTRA.has(char)) return char;
      const plain = char.normalize('NFKD').replace(/\p{M}/gu, '');
      return /^[\x20-\x7e\xa0-\xff]+$/.test(plain) ? plain : '?';
    })
    .join('');
}

/**
 * Relatório da revisão em PDF, com a identidade do relatório bibliométrico
 * (features/report/pdf-generator.ts): fundo papel, rótulos mono em caixa-alta, régua pinho
 * com o bloco lima, cartões chapados e tabelas em tinta. O fluxo PRISMA é desenhado como
 * diagrama — exclusões com a luz vermelha da revisão, incluídos com a verde.
 */

type Rgb = [number, number, number];

function rgb(hex: string): Rgb {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

const C = {
  paper: rgb(BRAND.paper),
  card: rgb(BRAND.card),
  border: rgb(BRAND.border),
  ink: rgb(BRAND.ink),
  inkMuted: rgb(BRAND.inkMuted),
  pine: rgb(BRAND.pine),
  lime: rgb(BRAND.lime),
  // Mesmos tons de --glow-include / --glow-exclude do tema claro.
  include: rgb('#16a34a'),
  includeTint: rgb('#e3f1e4'),
  warning: rgb('#b45309'),
  edited: rgb('#1d4ed8'),
  exclude: rgb('#dc2626'),
  excludeTint: rgb('#f6e3dc'),
};

/** Monta o documento; `downloadReviewPdf` o baixa. Separado para poder ser testado sem navegador. */
export function buildReviewPdf({ review, flow, records, copy, locale }: ReviewReportInput): jsPDF {
  const text = REPORT_TEXT[locale];
  const isEn = locale === 'en';
  const labels = { yes: copy.yes, no: copy.no };
  const num = (value: number) => value.toLocaleString(isEn ? 'en' : 'pt-BR', { maximumFractionDigits: 2 });

  const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 40;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  const painted = new Set<number>();
  const paintPage = (): void => {
    const page = doc.getCurrentPageInfo().pageNumber;
    if (painted.has(page)) return;
    painted.add(page);
    doc.setFillColor(...C.paper);
    doc.rect(0, 0, pageWidth, pageHeight, 'F');
  };
  paintPage();

  const ensure = (height: number): void => {
    if (y + height > pageHeight - margin - 30) {
      doc.addPage();
      paintPage();
      y = margin + 10;
    }
  };

  const eyebrow = (label: string, x: number, top: number, color: Rgb = C.pine, size = 7.5): void => {
    doc.setFont('courier', 'bold');
    doc.setFontSize(size);
    doc.setTextColor(...color);
    doc.text(label.toUpperCase(), x, top, { charSpace: 0.8 });
  };

  let section = 0;
  const heading = (title: string): void => {
    ensure(60);
    section += 1;
    eyebrow(`— ${String(section).padStart(2, '0')}`, margin, y);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(...C.ink);
    doc.text(title, margin, y + 16);
    y += 34;
  };

  const subheading = (title: string, color: Rgb = C.ink): void => {
    ensure(30);
    y += 4;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9.5);
    doc.setTextColor(...color);
    doc.text(title, margin, y);
    y += 13;
  };

  const paragraph = (body: string, options: { size?: number; color?: Rgb; font?: 'helvetica' | 'courier'; indent?: number } = {}): void => {
    const size = options.size ?? 9;
    const indent = options.indent ?? 0;
    doc.setFont(options.font ?? 'helvetica', 'normal');
    doc.setFontSize(size);
    doc.setTextColor(...(options.color ?? C.ink));
    const lines = doc.splitTextToSize(body, contentWidth - indent) as string[];
    const lineHeight = size * 1.35;
    for (const line of lines) {
      ensure(lineHeight);
      doc.text(line, margin + indent, y);
      y += lineHeight;
    }
    y += 3;
  };

  /** Item de lista com um marcador quadrado na cor pedida (verde inclusão, vermelho exclusão). */
  const bullet = (body: string, color: Rgb = C.pine): void => {
    ensure(14);
    doc.setFillColor(...color);
    doc.rect(margin + 1, y - 5.5, 4, 4, 'F');
    paragraph(body, { indent: 12 });
    y -= 2;
  };

  const tableBase: Partial<UserOptions> = {
    margin: { left: margin, right: margin, bottom: margin + 30 },
    theme: 'grid',
    willDrawPage: paintPage,
    headStyles: { fillColor: C.ink, textColor: C.paper, fontSize: 8, fontStyle: 'bold' },
    bodyStyles: { fillColor: C.card },
    alternateRowStyles: { fillColor: C.paper },
    styles: { font: 'helvetica', fontSize: 8, cellPadding: 4, textColor: C.ink, lineColor: C.border, lineWidth: 0.5, overflow: 'linebreak' },
  };
  const table = (options: Partial<UserOptions>): void => {
    ensure(40);
    autoTable(doc, { startY: y, ...tableBase, ...options });
    y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 14;
  };

  // --- Capa ---
  eyebrow(text.eyebrow, margin, y + 6);
  y += 28;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.setTextColor(...C.ink);
  const titleLines = doc.splitTextToSize(review.title.trim() || text.untitled, contentWidth) as string[];
  doc.text(titleLines, margin, y);
  y += titleLines.length * 22;

  // Tipo de revisão em serifa itálica pinho, como o destaque do título do relatório bibliométrico.
  doc.setFont('times', 'italic');
  doc.setFontSize(14);
  doc.setTextColor(...C.pine);
  doc.text(`${copy.types[review.type]} · ${REVIEW_DEFAULTS[review.type].guideline}`, margin, y);
  y += 15;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(...C.inkMuted);
  const date = new Date().toLocaleDateString(isEn ? 'en-US' : 'pt-BR', { day: '2-digit', month: 'long', year: 'numeric' });
  doc.text(`${text.generatedOn} ${date} · ${isEn ? 'Simetrics · A Scientata application' : 'Simetrics · Uma aplicação Scientata'}`, margin, y);
  y += 12;

  doc.setDrawColor(...C.pine);
  doc.setLineWidth(1);
  doc.line(margin, y, pageWidth - margin, y);
  doc.setFillColor(...C.lime);
  doc.rect(margin, y - 2, 36, 4, 'F');
  y += 20;

  // --- Indicadores ---
  const included = includedStudies(records, review);
  const selected = finalSelection(records, review);
  const assessedScores = included.map((study) => scoreStudy(review, study.key)).filter((score) => score.complete);
  const kpis: [string, string, Rgb?][] = [
    [text.kpiIdentified, num(flow.identified)],
    [text.kpiScreened, num(flow.screened)],
    [text.kpiIncluded, num(flow.included), C.include],
  ];
  if (hasQualityChecklist(review) && assessedScores.length > 0) {
    const mean = assessedScores.reduce((sum, score) => sum + score.score, 0) / assessedScores.length;
    kpis.push([copy.meanScore, `${num(mean)} / ${num(maxQualityScore(review))}`]);
  }
  const gap = 8;
  const cardW = (contentWidth - gap * (kpis.length - 1)) / kpis.length;
  kpis.forEach(([label, value, accent], index) => {
    const x = margin + index * (cardW + gap);
    doc.setFillColor(...C.card);
    doc.setDrawColor(...(accent ?? C.border));
    doc.setLineWidth(accent ? 1 : 0.5);
    doc.rect(x, y, cardW, 52, 'FD');
    eyebrow(label, x + 9, y + 15, C.inkMuted, 6.5);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(17);
    doc.setTextColor(...(accent ?? C.ink));
    doc.text(value, x + 9, y + 40);
  });
  y += 70;

  const model = buildReportModel(review, records, copy, locale);
  const longDate = (iso: string) =>
    iso ? new Date(iso).toLocaleDateString(isEn ? 'en-US' : 'pt-BR', { day: '2-digit', month: 'long', year: 'numeric' }) : '—';

  // --- Protocolo ---
  heading(text.protocol);
  paragraph(fillText(text.dates, { created: longDate(review.createdAt), updated: longDate(review.updatedAt) }), { color: C.inkMuted, size: 8.5 });
  if (review.objective.trim()) {
    subheading(copy.objectiveLabel);
    paragraph(review.objective.trim());
  }
  const frameworkRows = FRAMEWORK_FIELDS[review.framework]
    .filter((field) => review.frameworkValues[field]?.trim())
    .map((field) => [copy.frameworkFields[field] ?? field, review.frameworkValues[field]!.trim()]);
  if (frameworkRows.length > 0) {
    subheading(`${copy.frameworkLabel} (${copy.frameworks[review.framework]})`);
    table({ body: frameworkRows, columnStyles: { 0: { cellWidth: 130, fontStyle: 'bold' } } });
  }
  const questions = review.questions.filter((question) => question.text.trim());
  if (questions.length > 0) {
    subheading(copy.questionsLabel);
    questions.forEach((question, index) => paragraph(`Q${index + 1}. ${question.text.trim()}`));
  }
  for (const kind of ['inclusion', 'exclusion'] as const) {
    const criteria = review.criteria.filter((criterion) => criterion.kind === kind && criterion.text.trim());
    if (criteria.length === 0) continue;
    const color = kind === 'inclusion' ? C.include : C.exclude;
    subheading(`${copy.criteriaLabel} — ${kind === 'inclusion' ? copy.inclusion : copy.exclusion}`, color);
    criteria.forEach((criterion) => bullet(criterion.text.trim(), color));
    y += 4;
  }
  if (model.extractionForm.length > 0) {
    subheading(copy.extractionForm);
    table({
      head: [[text.field, text.type, text.options]],
      body: model.extractionForm.map((row) => row.map(pdfSafe)),
      columnStyles: { 0: { cellWidth: 150, fontStyle: 'bold' }, 1: { cellWidth: 90 } },
    });
  }

  // --- Busca ---
  heading(text.search);
  const concepts = review.concepts.filter((concept) => concept.terms.length > 0);
  if (concepts.length > 0) {
    table({
      head: [[text.concept, text.terms]],
      body: concepts.map((concept) => [concept.label || '—', concept.terms.join(' OR ')]),
      columnStyles: { 0: { cellWidth: 130, fontStyle: 'bold' } },
    });
    for (const target of SEARCH_TARGETS.filter((item) => review.searchTargets.includes(item.id))) {
      const query = buildSearchString(review.concepts, target.id);
      doc.setFont('courier', 'normal');
      doc.setFontSize(8);
      const lines = doc.splitTextToSize(query, contentWidth - 20) as string[];
      const boxHeight = lines.length * 10.5 + 26;
      ensure(boxHeight + 6);
      doc.setFillColor(...C.card);
      doc.setDrawColor(...C.border);
      doc.setLineWidth(0.5);
      doc.rect(margin, y, contentWidth, boxHeight, 'FD');
      doc.setFillColor(...C.pine);
      doc.rect(margin, y, 2.5, boxHeight, 'F');
      eyebrow(target.id === 'generic' ? copy.genericTarget : target.name, margin + 10, y + 13, C.pine, 6.5);
      doc.setFont('courier', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(...C.ink);
      doc.text(lines, margin + 10, y + 26);
      y += boxHeight + 8;
    }
  }
  if (flow.identifiedBySource.length > 0) {
    subheading(copy.identifiedFrom);
    table({
      head: [[text.database, text.records]],
      body: flow.identifiedBySource.map((source) => [source.name, num(source.count)]),
      columnStyles: { 1: { halign: 'right', cellWidth: 90 } },
    });
  }

  // --- Seleção (diagrama PRISMA) ---
  heading(text.selection);
  const assessedReports = flow.fullText.eligible - flow.fullText.notRetrieved;
  const rows: { main: [string, number]; side?: [string, number, string[]?] }[] = [
    { main: [copy.identifiedFrom, flow.identified], side: [copy.duplicatesRemoved, flow.duplicatesRemoved] },
    { main: [copy.recordsScreened, flow.screened], side: [copy.recordsExcluded, flow.titleAbstract.exclude] },
    { main: [copy.reportsSought, flow.fullText.eligible], side: [copy.reportsNotRetrieved, flow.fullText.notRetrieved] },
    {
      main: [copy.reportsAssessed, assessedReports],
      side: [
        copy.reportsExcluded,
        flow.fullText.exclude,
        flow.fullTextExclusions.map((reason) => `${reason.label}: ${num(reason.count)}`),
      ],
    },
    { main: [copy.studiesIncluded, flow.included] },
  ];
  const mainW = contentWidth * 0.52;
  const sideX = margin + mainW + 28;
  const sideW = contentWidth - mainW - 28;
  rows.forEach((row, index) => {
    const reasons = row.side?.[2] ?? [];
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    const reasonLines = reasons.flatMap((reason) => doc.splitTextToSize(`· ${reason}`, sideW - 16) as string[]);
    const boxH = Math.max(46, 40 + reasonLines.length * 9.5);
    ensure(boxH + 22);
    const last = index === rows.length - 1;

    // Caixa principal (a última, dos incluídos, acesa em verde).
    doc.setFillColor(...(last ? C.includeTint : C.card));
    doc.setDrawColor(...(last ? C.include : C.border));
    doc.setLineWidth(last ? 1.2 : 0.6);
    doc.rect(margin, y, mainW, 46, 'FD');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(...C.inkMuted);
    doc.text(doc.splitTextToSize(row.main[0], mainW - 16) as string[], margin + 8, y + 14);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.setTextColor(...(last ? C.include : C.ink));
    doc.text(`n = ${num(row.main[1])}`, margin + 8, y + 37);

    if (row.side) {
      // Seta até a caixa de exclusão e a caixa em vermelho.
      doc.setDrawColor(...C.inkMuted);
      doc.setLineWidth(0.6);
      doc.line(margin + mainW, y + 23, sideX - 4, y + 23);
      doc.setFillColor(...C.inkMuted);
      doc.triangle(sideX - 4, y + 20, sideX - 4, y + 26, sideX, y + 23, 'F');
      doc.setFillColor(...C.excludeTint);
      doc.setDrawColor(...C.exclude);
      doc.setLineWidth(0.8);
      doc.rect(sideX, y, sideW, boxH, 'FD');
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(...C.inkMuted);
      doc.text(doc.splitTextToSize(row.side[0], sideW - 16) as string[], sideX + 8, y + 14);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(12);
      doc.setTextColor(...C.exclude);
      doc.text(`n = ${num(row.side[1])}`, sideX + 8, y + 33);
      if (reasonLines.length > 0) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.5);
        doc.setTextColor(...C.ink);
        doc.text(reasonLines, sideX + 8, y + 45);
      }
    }

    y += Math.max(46, row.side ? boxH : 46);
    if (!last) {
      // Seta para a próxima etapa.
      const cx = margin + mainW / 2;
      doc.setDrawColor(...C.inkMuted);
      doc.setLineWidth(0.6);
      doc.line(cx, y, cx, y + 14);
      doc.setFillColor(...C.inkMuted);
      doc.triangle(cx - 3, y + 12, cx + 3, y + 12, cx, y + 17, 'F');
      y += 18;
    }
  });
  y += 16;

  if (model.taReasons.length > 0) {
    subheading(text.taReasons, C.exclude);
    table({
      head: [[text.reason, text.count]],
      body: model.taReasons.map((reason) => [pdfSafe(reason.label), num(reason.count)]),
      columnStyles: { 1: { halign: 'right', cellWidth: 60 } },
    });
  }
  if (model.fullTextExcluded.length > 0) {
    subheading(text.ftExcluded, C.exclude);
    table({
      head: [[copy.study, text.reason, text.excerpt]],
      body: model.fullTextExcluded.map((item) => [
        pdfSafe(item.label),
        pdfSafe(item.reason),
        pdfSafe(quotesCell(item.quotes, text.page, '—')),
      ]),
      columnStyles: { 0: { cellWidth: 170 }, 1: { cellWidth: 110 } },
    });
  }

  // --- Qualidade ---
  if (hasQualityChecklist(review)) {
    heading(text.quality);
    review.qualityQuestions.forEach((question, index) => paragraph(`Q${index + 1}. ${question.text.trim() || '—'}`));
    paragraph(`${text.answers}: ${review.qualityAnswers.map((answer) => `${answer.label} (${num(answer.weight)})`).join(' · ')}`, {
      color: C.inkMuted,
    });
    paragraph(
      `${text.maxScore}: ${num(maxQualityScore(review))} · ${text.cutoff}: ${review.qualityCutoff === null ? text.none : num(review.qualityCutoff)}`,
      { color: C.inkMuted },
    );
    if (review.excludeBelowCutoff && review.qualityCutoff !== null) paragraph(text.excludedByQuality, { color: C.exclude });
    if (included.length > 0) {
      const answers = new Map(review.qualityAnswers.map((answer) => [answer.id, answer.label]));
      const scoreColumn = review.qualityQuestions.length + 1;
      table({
        head: [[copy.study, ...review.qualityQuestions.map((_, index) => `Q${index + 1}`), copy.score]],
        body: included.map((study) => {
          const responses = review.quality[study.key] ?? {};
          const { score, max, complete } = scoreStudy(review, study.key);
          return [
            shortLabel(study),
            ...review.qualityQuestions.map((question) => answers.get(responses[question.id] ?? '') ?? '—'),
            complete ? `${num(score)}/${num(max)}` : '—',
          ];
        }),
        columnStyles: { 0: { cellWidth: 120 }, [scoreColumn]: { halign: 'right', fontStyle: 'bold' } },
        // Nota que passa em verde, abaixo da nota de corte em vermelho.
        didParseCell: (data) => {
          if (data.section !== 'body' || data.column.index !== scoreColumn) return;
          const study = included[data.row.index];
          const passes = study ? scoreStudy(review, study.key).passes : null;
          if (passes === true) data.cell.styles.textColor = C.include;
          if (passes === false) data.cell.styles.textColor = C.exclude;
        },
      });
    }
  }

  // --- Características ---
  if (review.extractionFields.length > 0 && selected.length > 0) {
    heading(text.characteristics);
    if (review.extractionFields.length <= 4) {
      table({
        head: [[copy.study, ...review.extractionFields.map((field) => field.label || '—')]],
        body: selected.map((study) => [
          shortLabel(study),
          ...review.extractionFields.map((field) => formatExtractionValue(review.extraction[study.key]?.values[field.id], labels) || '—'),
        ]),
        columnStyles: { 0: { cellWidth: 120 } },
      });
    } else {
      // Muitos campos não cabem como colunas: uma linha por estudo e campo.
      table({
        head: [[copy.study, isEn ? 'Field' : 'Campo', isEn ? 'Value' : 'Valor']],
        body: selected.flatMap((study) =>
          review.extractionFields.map((field, index) => [
            index === 0 ? shortLabel(study) : '',
            field.label || '—',
            formatExtractionValue(review.extraction[study.key]?.values[field.id], labels) || '—',
          ]),
        ),
        columnStyles: { 0: { cellWidth: 120 }, 1: { cellWidth: 130 } },
      });
    }
  }

  // --- Síntese ---
  if (selected.length > 0 && (model.synthesis.length > 0 || model.byYear.length > 0)) {
    heading(text.synthesis);
    if (model.synthesis.length > 0) {
      subheading(copy.fieldsSummary);
      table({
        head: [[text.field, text.filled, text.summary]],
        body: model.synthesis.map((row) => [pdfSafe(row.field), row.filled, pdfSafe(row.summary)]),
        columnStyles: { 0: { cellWidth: 130, fontStyle: 'bold' }, 1: { cellWidth: 90 } },
      });
    }
    if (model.byYear.length > 0) {
      subheading(copy.byYear);
      table({
        head: [[text.year, text.studies]],
        body: model.byYear.map((row) => [row.year, num(row.count)]),
        columnStyles: { 1: { halign: 'right', cellWidth: 80 } },
      });
    }
  }

  // --- Textos completos e IA ---
  if (model.ai.studiesWithPdf > 0 || model.evidence.length > 0) {
    heading(text.fullText);
    fullTextParagraphs(model, text, locale).forEach((body) => paragraph(pdfSafe(body)));
  }

  // --- Evidências por estudo ---
  if (model.evidence.length > 0) {
    heading(text.evidence);
    paragraph(text.evidenceHint, { color: C.inkMuted, size: 8.5 });
    const statusColor: Partial<Record<NonNullable<ReportAnswer['statusKind']>, Rgb>> = {
      confirmed: C.include,
      edited: C.edited,
      rejected: C.exclude,
      suggested: C.warning,
    };
    for (const block of model.evidence) {
      ensure(90);
      subheading(pdfSafe(block.label));
      paragraph(pdfSafe(block.reference), { size: 8, color: C.inkMuted });
      if (block.pdf) paragraph(pdfSafe(`${text.pdfLabel}: ${block.pdf}`), { size: 8, color: C.inkMuted, font: 'courier' });
      if (block.note) paragraph(pdfSafe(`${copy.note}: ${block.note}`), { size: 8 });
      // Linhas de grupo (extração, qualidade, exclusão) separam as perguntas na mesma tabela.
      type Row = { group: string } | { answer: ReportAnswer };
      const rows: Row[] = [];
      let group = '';
      for (const answer of block.answers) {
        if (answer.group !== group) {
          group = answer.group;
          rows.push({ group: text.groups[answer.group] });
        }
        rows.push({ answer });
      }
      table({
        head: [[text.question, text.answer, text.verification, text.excerpt]],
        body: rows.map((row) =>
          'group' in row
            ? [{ content: row.group.toUpperCase(), colSpan: 4, styles: { fillColor: C.paper, textColor: C.pine, fontStyle: 'bold', fontSize: 7 } }]
            : [
                pdfSafe(row.answer.question),
                pdfSafe(row.answer.suggestion ? `${row.answer.answer}\n${row.answer.suggestion}` : row.answer.answer),
                row.answer.status || '—',
                pdfSafe(quotesCell(row.answer.quotes, text.page, text.noExcerpt)),
              ],
        ),
        columnStyles: { 0: { cellWidth: 110, fontStyle: 'bold' }, 1: { cellWidth: 115 }, 2: { cellWidth: 62 } },
        didParseCell: (data) => {
          if (data.section !== 'body' || data.column.index !== 2) return;
          const row = rows[data.row.index];
          const kind = row && 'answer' in row ? row.answer.statusKind : null;
          const color = kind ? statusColor[kind] : undefined;
          if (color) {
            data.cell.styles.textColor = color;
            data.cell.styles.fontStyle = 'bold';
          }
        },
      });
    }
  }

  // --- Referências ---
  if (selected.length > 0) {
    heading(text.references);
    selected.forEach((study, index) => paragraph(`${index + 1}. ${reference(study)}`, { size: 8.5 }));
  }

  // --- Rodapé em todas as páginas ---
  const totalPages = doc.getNumberOfPages();
  const footer = isEn ? 'Simetrics · A Scientata application · scientata.com' : 'Simetrics · Uma aplicação Scientata · scientata.com';
  for (let page = 1; page <= totalPages; page += 1) {
    doc.setPage(page);
    doc.setDrawColor(...C.border);
    doc.setLineWidth(0.5);
    doc.line(margin, pageHeight - 25, pageWidth - margin, pageHeight - 25);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...C.inkMuted);
    doc.text(footer, margin, pageHeight - 14);
    doc.setFont('courier', 'normal');
    const pageLabel = isEn ? `Page ${page} of ${totalPages}` : `Página ${page} de ${totalPages}`;
    doc.text(pageLabel, pageWidth - margin - doc.getTextWidth(pageLabel), pageHeight - 14);
  }

  return doc;
}

export function downloadReviewPdf(input: ReviewReportInput): void {
  const doc = buildReviewPdf(input);
  const base = (input.review.title.trim() || 'revisao')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-zA-Z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  doc.save(`simetrics-${base}-relatorio-${new Date().toISOString().slice(0, 10)}.pdf`);
}
