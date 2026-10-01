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
    dates: 'Revisão criada em {created} · atualizada em {updated}',
    field: 'Campo',
    type: 'Tipo',
    options: 'Opções',
    taReasons: 'Motivos de exclusão na triagem por título e resumo',
    ftExcluded: 'Estudos excluídos na leitura do texto completo',
    reason: 'Motivo',
    excerpt: 'Trecho do artigo (página)',
    synthesis: 'Síntese dos resultados',
    filled: 'Preenchimento',
    summary: 'Resumo',
    year: 'Ano',
    studies: 'Estudos',
    fullText: 'Textos completos, evidências e uso de IA',
    fullTextMethod:
      'Os textos completos de {pdfs} estudo(s) foram anexados em PDF ({scanned} escaneado(s), sem camada de texto). {answers} resposta(s) foram ligadas a trechos do artigo que as sustentam, com a página.',
    aiUsed:
      'Automação: um modelo de linguagem ({models}) leu o texto completo e propôs {suggested} resposta(s), cada uma com o trecho do artigo citado. Cada trecho foi procurado no texto do PDF — {located} de {quotes} achado(s), {notFound} não localizado(s) — e toda proposta passou por conferência humana: {confirmed} confirmada(s), {edited} corrigida(s), {rejected} rejeitada(s){pending}.',
    aiPending: '; {n} ainda aguardando conferência',
    aiNotUsed: 'Nenhuma resposta foi proposta por IA: extração e avaliação foram feitas pelo revisor.',
    evidence: 'Evidências por estudo',
    evidenceHint:
      'Para cada estudo com texto completo ou evidência: a resposta registrada, a conferência dela e o trecho do artigo que a sustenta, com a página. Respostas propostas por IA trazem a sugestão original e o modelo.',
    question: 'Pergunta',
    answer: 'Resposta',
    verification: 'Conferência',
    groups: { extraction: 'Extração de dados', quality: 'Avaliação da qualidade', exclusion: 'Exclusão no texto completo' },
    noExcerpt: 'sem trecho ligado',
    page: 'p.',
    pdfLabel: 'Texto completo',
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
    dates: 'Review created on {created} · updated on {updated}',
    field: 'Field',
    type: 'Type',
    options: 'Options',
    taReasons: 'Exclusion reasons at title and abstract screening',
    ftExcluded: 'Studies excluded at full-text reading',
    reason: 'Reason',
    excerpt: 'Article passage (page)',
    synthesis: 'Synthesis of results',
    filled: 'Filled in',
    summary: 'Summary',
    year: 'Year',
    studies: 'Studies',
    fullText: 'Full texts, evidence and use of AI',
    fullTextMethod:
      'Full texts of {pdfs} study(ies) were attached as PDF ({scanned} scanned, with no text layer). {answers} answer(s) were linked to the article passages that support them, with the page.',
    aiUsed:
      'Automation: a language model ({models}) read the full text and proposed {suggested} answer(s), each with a quoted article passage. Every quote was searched for in the PDF text — {located} of {quotes} found, {notFound} not found — and every proposal was checked by a human reviewer: {confirmed} confirmed, {edited} corrected, {rejected} rejected{pending}.',
    aiPending: '; {n} still awaiting check',
    aiNotUsed: 'No answer was proposed by AI: extraction and assessment were done by the reviewer.',
    evidence: 'Evidence by study',
    evidenceHint:
      'For each study with a full text or evidence: the recorded answer, its check and the article passage that supports it, with the page. AI-proposed answers show the original suggestion and the model.',
    question: 'Question',
    answer: 'Answer',
    verification: 'Check',
    groups: { extraction: 'Data extraction', quality: 'Quality assessment', exclusion: 'Full-text exclusion' },
    noExcerpt: 'no passage linked',
    page: 'p.',
    pdfLabel: 'Full text',
  },
};

/** Preenche `{chave}` num modelo de frase. */
export function fillText(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? String(values[key]) : match));
}

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
