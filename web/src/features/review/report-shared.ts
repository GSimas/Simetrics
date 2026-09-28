import type { ReviewFlow } from '@/core/review/flow';
import type { ScreeningRecord } from '@/core/review/records';
import type { ReviewState } from '@/core/review/types';
import type { ReviewCopy } from './copy';

/** Textos e ajudantes comuns aos relatórios da revisão em Word e em PDF. */

export interface ReviewReportInput {
  review: ReviewState;
  flow: ReviewFlow;
  records: readonly ScreeningRecord[];
  copy: ReviewCopy;
  locale: 'pt' | 'en';
}

export const REPORT_TEXT = {
  pt: {
    eyebrow: 'Relatório da revisão · Simetrics',
    untitled: 'Revisão sem título',
    generatedOn: 'Gerado em',
    protocol: 'Protocolo',
    search: 'Estratégia de busca',
    selection: 'Seleção dos estudos (PRISMA 2020)',
    quality: 'Avaliação da qualidade',
    characteristics: 'Características dos estudos',
    references: 'Estudos incluídos',
    stage: 'Etapa',
    count: 'n',
    concept: 'Conceito',
    terms: 'Termos',
    database: 'Base',
    records: 'Registros',
    answers: 'Respostas (peso)',
    cutoff: 'Nota de corte',
    none: 'não definida',
    excludedByQuality: 'Estudos abaixo da nota de corte excluídos da seleção final.',
    maxScore: 'Nota máxima',
    kpiIdentified: 'Identificados',
    kpiScreened: 'Triados',
    kpiIncluded: 'Incluídos',
  },
  en: {
    eyebrow: 'Review report · Simetrics',
    untitled: 'Untitled review',
    generatedOn: 'Generated on',
    protocol: 'Protocol',
    search: 'Search strategy',
    selection: 'Study selection (PRISMA 2020)',
    quality: 'Quality assessment',
    characteristics: 'Characteristics of the studies',
    references: 'Included studies',
    stage: 'Stage',
    count: 'n',
    concept: 'Concept',
    terms: 'Terms',
    database: 'Database',
    records: 'Records',
    answers: 'Answers (weight)',
    cutoff: 'Cutoff score',
    none: 'not set',
    excludedByQuality: 'Studies below the cutoff were excluded from the final selection.',
    maxScore: 'Maximum score',
    kpiIdentified: 'Identified',
    kpiScreened: 'Screened',
    kpiIncluded: 'Included',
  },
};

export function reference(study: ScreeningRecord): string {
  const authors = study.authors.split(';').map((author) => author.trim()).filter(Boolean);
  const byline = authors.length > 3 ? `${authors.slice(0, 3).join('; ')} et al.` : authors.join('; ');
  const doi = study.doi ? ` https://doi.org/${study.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '')}` : '';
  return `${byline}${byline ? ' ' : ''}(${study.year ?? 's.d.'}). ${study.title}. ${study.venue}${study.venue ? '.' : ''}${doi}`.trim();
}


export function shortLabel(study: ScreeningRecord): string {
  const firstAuthor = study.authors.split(';')[0]?.trim();
  const label = [firstAuthor, study.year].filter(Boolean).join(', ');
  return label || study.title.slice(0, 80);
}
