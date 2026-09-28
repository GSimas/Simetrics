import {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  Packer,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TabStopPosition,
  TabStopType,
  TextRun,
  WidthType,
} from 'docx';

import { downloadBlob, timestampedFilename } from '@/core/export';
import { finalSelection, hasQualityChecklist, includedStudies, maxQualityScore, scoreStudy } from '@/core/review/quality';
import { buildSearchString, SEARCH_TARGETS } from '@/core/review/search-string';
import { formatExtractionValue } from '@/core/review/synthesis';
import { FRAMEWORK_FIELDS, REVIEW_DEFAULTS } from '@/core/review/types';
import { BRAND } from '@/features/report/chart-renderer';
import { REPORT_TEXT, reference, shortLabel, type ReviewReportInput } from './report-shared';

/**
 * Relatório da revisão em Word: protocolo, estratégia de busca, fluxo PRISMA, qualidade e
 * características dos estudos. Mesma identidade visual do relatório bibliométrico
 * (features/report/docx-generator.ts), num documento próprio da revisão.
 */

const hex = (color: string) => color.slice(1).toUpperCase();
const INK = hex(BRAND.ink);
const INK_MUTED = hex(BRAND.inkMuted);
const PINE = hex(BRAND.pine);
const CARD = hex(BRAND.card);
const PAPER = hex(BRAND.paper);
const BORDER = hex(BRAND.border);
const SANS = 'Manrope';
const MONO = 'DM Mono';
const TEXT_WIDTH = 9020;

const hairline = { style: BorderStyle.SINGLE, size: 4, color: BORDER };
const hairlines = { top: hairline, bottom: hairline, left: hairline, right: hairline };

function heading(num: number, title: string): Paragraph[] {
  return [
    new Paragraph({
      children: [new TextRun({ text: `— ${String(num).padStart(2, '0')}`, font: MONO, color: PINE, size: 15, characterSpacing: 20 })],
      spacing: { before: 320, after: 40 },
      keepNext: true,
    }),
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: title, bold: true, font: SANS, size: 28, color: INK })],
      spacing: { after: 120 },
      keepNext: true,
    }),
  ];
}

function subheading(text: string): Paragraph {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, font: SANS, size: 21, color: INK })],
    spacing: { before: 200, after: 80 },
    keepNext: true,
  });
}

function body(text: string, options: { mono?: boolean; muted?: boolean } = {}): Paragraph {
  return new Paragraph({
    children: [
      new TextRun({ text, font: options.mono ? MONO : SANS, size: options.mono ? 17 : 20, color: options.muted ? INK_MUTED : INK }),
    ],
    spacing: { after: 100 },
  });
}

function labeled(label: string, value: string): Paragraph {
  return new Paragraph({
    children: [
      new TextRun({ text: `${label}: `, bold: true, font: SANS, size: 20, color: INK }),
      new TextRun({ text: value, font: SANS, size: 20, color: INK }),
    ],
    spacing: { after: 80 },
  });
}

function bullet(text: string): Paragraph {
  return new Paragraph({
    children: [new TextRun({ text, font: SANS, size: 20, color: INK })],
    bullet: { level: 0 },
    spacing: { after: 60 },
  });
}

function table(rows: string[][], widths?: number[]): Table {
  const columns = rows[0]?.length ?? 1;
  const colWidths = widths ?? Array<number>(columns).fill(Math.floor(TEXT_WIDTH / columns));
  return new Table({
    columnWidths: colWidths,
    width: { size: colWidths.reduce((sum, width) => sum + width, 0), type: WidthType.DXA },
    rows: rows.map(
      (row, rowIndex) =>
        new TableRow({
          tableHeader: rowIndex === 0,
          cantSplit: true,
          children: row.map(
            (cell, colIndex) =>
              new TableCell({
                width: { size: colWidths[colIndex] ?? 1000, type: WidthType.DXA },
                borders: hairlines,
                shading: {
                  type: ShadingType.CLEAR,
                  fill: rowIndex === 0 ? INK : rowIndex % 2 === 1 ? CARD : PAPER,
                },
                margins: { top: 60, bottom: 60, left: 100, right: 100 },
                children: [
                  new Paragraph({
                    alignment: AlignmentType.LEFT,
                    children: [
                      new TextRun({ text: cell, bold: rowIndex === 0, font: SANS, size: 17, color: rowIndex === 0 ? PAPER : INK }),
                    ],
                  }),
                ],
              }),
          ),
        }),
    ),
  });
}


export async function downloadReviewReport({ review, flow, records, copy, locale }: ReviewReportInput): Promise<void> {
  const text = REPORT_TEXT[locale];
  const labels = { yes: copy.yes, no: copy.no };
  const num = (value: number) => value.toLocaleString(locale === 'en' ? 'en' : 'pt-BR', { maximumFractionDigits: 2 });
  const children: (Paragraph | Table)[] = [];
  let section = 0;

  // Capa
  children.push(
    new Paragraph({
      children: [new TextRun({ text: text.eyebrow.toUpperCase(), font: MONO, color: PINE, size: 16, characterSpacing: 20 })],
      spacing: { after: 120 },
    }),
    new Paragraph({
      heading: HeadingLevel.TITLE,
      children: [new TextRun({ text: review.title.trim() || text.untitled, bold: true, font: SANS, size: 44, color: INK })],
      spacing: { after: 160 },
    }),
    body(`${copy.types[review.type]} · ${REVIEW_DEFAULTS[review.type].guideline}`, { muted: true }),
    body(`${text.generatedOn} ${new Date().toLocaleDateString(locale === 'en' ? 'en' : 'pt-BR')}`, { muted: true }),
  );

  // Protocolo
  children.push(...heading((section += 1), text.protocol));
  if (review.objective.trim()) children.push(labeled(copy.objectiveLabel, review.objective.trim()));
  const frameworkRows = FRAMEWORK_FIELDS[review.framework]
    .filter((field) => review.frameworkValues[field]?.trim())
    .map((field) => [copy.frameworkFields[field] ?? field, review.frameworkValues[field]!.trim()]);
  if (frameworkRows.length > 0) {
    children.push(subheading(`${copy.frameworkLabel} (${copy.frameworks[review.framework]})`), table(frameworkRows, [2600, TEXT_WIDTH - 2600]));
  }
  const questions = review.questions.filter((question) => question.text.trim());
  if (questions.length > 0) {
    children.push(subheading(copy.questionsLabel), ...questions.map((question, index) => body(`Q${index + 1}. ${question.text.trim()}`)));
  }
  for (const kind of ['inclusion', 'exclusion'] as const) {
    const criteria = review.criteria.filter((criterion) => criterion.kind === kind && criterion.text.trim());
    if (criteria.length === 0) continue;
    children.push(
      subheading(`${copy.criteriaLabel} — ${kind === 'inclusion' ? copy.inclusion : copy.exclusion}`),
      ...criteria.map((criterion) => bullet(criterion.text.trim())),
    );
  }

  // Busca
  children.push(...heading((section += 1), text.search));
  const concepts = review.concepts.filter((concept) => concept.terms.length > 0);
  if (concepts.length > 0) {
    children.push(
      table(
        [[text.concept, text.terms], ...concepts.map((concept) => [concept.label || '—', concept.terms.join(' OR ')])],
        [2600, TEXT_WIDTH - 2600],
      ),
    );
    for (const target of SEARCH_TARGETS.filter((item) => review.searchTargets.includes(item.id))) {
      children.push(subheading(target.id === 'generic' ? copy.genericTarget : target.name), body(buildSearchString(review.concepts, target.id), { mono: true }));
    }
  }
  if (flow.identifiedBySource.length > 0) {
    children.push(
      subheading(copy.identifiedFrom),
      table([[text.database, text.records], ...flow.identifiedBySource.map((source) => [source.name, String(source.count)])], [6000, TEXT_WIDTH - 6000]),
    );
  }

  // Seleção
  children.push(...heading((section += 1), text.selection));
  const assessed = flow.fullText.eligible - flow.fullText.notRetrieved;
  children.push(
    table(
      [
        [text.stage, text.count],
        [copy.identifiedFrom, String(flow.identified)],
        [copy.duplicatesRemoved, String(flow.duplicatesRemoved)],
        [copy.recordsScreened, String(flow.screened)],
        [copy.recordsExcluded, String(flow.titleAbstract.exclude)],
        [copy.reportsSought, String(flow.fullText.eligible)],
        [copy.reportsNotRetrieved, String(flow.fullText.notRetrieved)],
        [copy.reportsAssessed, String(assessed)],
        [copy.reportsExcluded, String(flow.fullText.exclude)],
        [copy.studiesIncluded, String(flow.included)],
      ],
      [7000, TEXT_WIDTH - 7000],
    ),
  );
  if (flow.fullTextExclusions.length > 0) {
    children.push(subheading(copy.reportsExcluded), ...flow.fullTextExclusions.map((reason) => bullet(`${reason.label}: ${reason.count}`)));
  }

  // Qualidade
  const included = includedStudies(records, review);
  const selected = finalSelection(records, review);
  if (hasQualityChecklist(review)) {
    children.push(...heading((section += 1), text.quality));
    children.push(
      ...review.qualityQuestions.map((question, index) => body(`Q${index + 1}. ${question.text.trim() || '—'}`)),
      labeled(text.answers, review.qualityAnswers.map((answer) => `${answer.label} (${num(answer.weight)})`).join(' · ')),
      labeled(text.maxScore, num(maxQualityScore(review))),
      labeled(text.cutoff, review.qualityCutoff === null ? text.none : num(review.qualityCutoff)),
    );
    if (review.excludeBelowCutoff && review.qualityCutoff !== null) children.push(body(text.excludedByQuality, { muted: true }));
    if (included.length > 0) {
      const answers = new Map(review.qualityAnswers.map((answer) => [answer.id, answer.label]));
      const questionCount = review.qualityQuestions.length;
      const studyWidth = 3000;
      const scoreWidth = 1100;
      const questionWidth = Math.max(500, Math.floor((TEXT_WIDTH - studyWidth - scoreWidth) / Math.max(1, questionCount)));
      children.push(
        table(
          [
            [copy.study, ...review.qualityQuestions.map((_, index) => `Q${index + 1}`), copy.score],
            ...included.map((study) => {
              const responses = review.quality[study.key] ?? {};
              const { score, max, complete } = scoreStudy(review, study.key);
              return [
                shortLabel(study),
                ...review.qualityQuestions.map((question) => answers.get(responses[question.id] ?? '') ?? '—'),
                complete ? `${num(score)}/${num(max)}` : '—',
              ];
            }),
          ],
          [studyWidth, ...Array<number>(questionCount).fill(questionWidth), scoreWidth],
        ),
      );
    }
  }

  // Características
  if (review.extractionFields.length > 0 && selected.length > 0) {
    children.push(...heading((section += 1), text.characteristics));
    if (review.extractionFields.length <= 4) {
      const fieldWidth = Math.floor((TEXT_WIDTH - 3000) / review.extractionFields.length);
      children.push(
        table(
          [
            [copy.study, ...review.extractionFields.map((field) => field.label || '—')],
            ...selected.map((study) => [
              shortLabel(study),
              ...review.extractionFields.map(
                (field) => formatExtractionValue(review.extraction[study.key]?.values[field.id], labels) || '—',
              ),
            ]),
          ],
          [3000, ...Array<number>(review.extractionFields.length).fill(fieldWidth)],
        ),
      );
    } else {
      // Muitos campos não cabem como colunas: um bloco por estudo.
      for (const study of selected) {
        children.push(subheading(shortLabel(study)));
        for (const field of review.extractionFields) {
          const value = formatExtractionValue(review.extraction[study.key]?.values[field.id], labels);
          if (value) children.push(labeled(field.label || '—', value));
        }
      }
    }
  }

  // Referências
  if (selected.length > 0) {
    children.push(...heading(section + 1, text.references), ...selected.map((study) => body(reference(study))));
  }

  const doc = new Document({
    creator: 'Simetrics',
    title: review.title.trim() || text.untitled,
    sections: [
      {
        properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
                children: [
                  new TextRun({ text: 'SIMETRICS · SCIENTATA', font: MONO, size: 15, color: INK_MUTED }),
                  new TextRun({
                    children: ['\t', PageNumber.CURRENT, ' / ', PageNumber.TOTAL_PAGES],
                    font: MONO,
                    size: 15,
                    color: INK_MUTED,
                  }),
                ],
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  const blob = await Packer.toBlob(doc);
  downloadBlob(timestampedFilename(`${review.title.trim() || 'revisao'}-relatorio`, 'docx'), blob);
}
