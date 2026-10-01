import type {
  EvidenceTargetKey,
  ExtractionField,
  ExtractionValue,
  QualityAnswer,
  QuoteLocation,
  ReviewState,
  StudyDocument,
  TargetEvidence,
  VerificationStatus,
} from './types';

/**
 * Evidências: o trecho do artigo que sustenta cada resposta e a conferência humana dela.
 *
 * Tudo aqui é puro — localizar uma citação no texto do PDF, decidir o status depois de uma
 * mudança, contar os números do piloto — para ser testado sem navegador.
 */

export const FT_EXCLUSION: EvidenceTargetKey = 'ft-exclusion';
export const extractionTarget = (fieldId: string): EvidenceTargetKey => `extraction:${fieldId}`;
export const qualityTarget = (questionId: string): EvidenceTargetKey => `quality:${questionId}`;

// ---------------------------------------------------------------------------------------
// Localizar uma citação no texto

/**
 * Texto só com letras e dígitos, minúsculo e sem acentos, e a posição de cada caractere no
 * original. Espaços, quebras de linha, hifenização no fim da linha, aspas tipográficas e
 * ligaduras (ﬁ) variam entre a extração do pdf.js e o que o modelo devolve; letras e
 * números, não.
 */
export function normalizeForMatch(text: string): { norm: string; starts: number[]; ends: number[] } {
  let norm = '';
  const starts: number[] = [];
  const ends: number[] = [];
  let index = 0;
  for (const char of text) {
    const end = index + char.length;
    const folded = char.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase();
    for (const piece of folded) {
      if (/[\p{L}\p{N}]/u.test(piece)) {
        norm += piece;
        starts.push(index);
        ends.push(end);
      }
    }
    index = end;
  }
  return { norm, starts, ends };
}

export interface QuoteMatch {
  /** Posições no texto original (fim exclusivo). */
  start: number;
  end: number;
  location: Exclude<QuoteLocation, 'not-found'>;
}

/** Citações mais curtas que isso só valem se baterem por inteiro — senão qualquer palavra "acha". */
const MIN_APPROXIMATE = 24;

/** Tamanho dos fragmentos comparados na busca aproximada. */
const GRAM = 8;
/** Fração dos fragmentos da citação que precisa aparecer, em ordem, no mesmo lugar do texto. */
const MIN_COVERAGE = 0.6;

/**
 * Acha `quote` em `haystack`. Primeiro igual (fora espaços e pontuação); depois aproximado,
 * por fragmentos: cada pedaço de 8 caracteres da citação que aparece no texto "vota" na
 * posição em que a citação começaria ali. Uma região que reúne a maior parte dos votos é a
 * citação com algumas palavras trocadas — o modelo às vezes conjuga, resume ou corrige.
 */
export function locateQuote(haystack: string, quote: string): QuoteMatch | null {
  const target = normalizeForMatch(quote).norm;
  if (!target) return null;
  const hay = normalizeForMatch(haystack);
  const toOriginal = (from: number, to: number, location: QuoteMatch['location']): QuoteMatch => ({
    start: hay.starts[from]!,
    end: hay.ends[to - 1]!,
    location,
  });

  const exact = hay.norm.indexOf(target);
  if (exact >= 0) return toOriginal(exact, exact + target.length, 'exact');
  if (target.length < MIN_APPROXIMATE) return null;

  const positions = new Map<string, number[]>();
  for (let at = 0; at + GRAM <= hay.norm.length; at += 1) {
    const gram = hay.norm.slice(at, at + GRAM);
    const list = positions.get(gram);
    if (list) list.push(at);
    else positions.set(gram, [at]);
  }
  // Cada acerto: a diagonal (onde a citação começaria) e a posição achada no texto.
  const hits: { diagonal: number; at: number }[] = [];
  const grams = target.length - GRAM + 1;
  for (let offset = 0; offset < grams; offset += 1) {
    for (const at of positions.get(target.slice(offset, offset + GRAM)) ?? []) hits.push({ diagonal: at - offset, at });
  }
  if (hits.length === 0) return null;
  hits.sort((a, b) => a.diagonal - b.diagonal);

  // A janela de diagonais com mais acertos tolera palavras inseridas ou removidas.
  const slack = Math.max(GRAM, Math.round(target.length * 0.15));
  let best = { count: 0, from: 0, to: 0 };
  for (let left = 0, right = 0; right < hits.length; right += 1) {
    while (hits[right]!.diagonal - hits[left]!.diagonal > slack) left += 1;
    if (right - left + 1 > best.count) best = { count: right - left + 1, from: left, to: right };
  }
  // Um fragmento repetido no texto pode votar mais de uma vez na mesma janela: conta-se a
  // cobertura pelos fragmentos distintos da citação, não pelos votos.
  const window = hits.slice(best.from, best.to + 1);
  const covered = new Set(window.map((hit) => hit.at - hit.diagonal)).size;
  if (covered / grams < MIN_COVERAGE) return null;
  const from = Math.min(...window.map((hit) => hit.at));
  const to = Math.max(...window.map((hit) => hit.at)) + GRAM;
  return toOriginal(from, to, 'approximate');
}

/** Acha a citação nas páginas, começando pela que o modelo indicou. Páginas contam de 1. */
export function locateInPages(
  pages: readonly string[],
  quote: string,
  pageHint?: number,
): (QuoteMatch & { page: number }) | null {
  const order = pages.map((_, index) => index + 1);
  if (pageHint && pageHint >= 1 && pageHint <= pages.length) {
    order.splice(pageHint - 1, 1);
    order.unshift(pageHint);
  }
  let approximate: (QuoteMatch & { page: number }) | null = null;
  for (const page of order) {
    const match = locateQuote(pages[page - 1] ?? '', quote);
    if (match?.location === 'exact') return { ...match, page };
    if (match && !approximate) approximate = { ...match, page };
  }
  return approximate;
}

const CONTEXT_CHARS = 40;

/** Texto antes e depois do trecho — o que desempata citações repetidas na mesma página. */
export function quoteContext(pageText: string, start: number, end: number): { prefix: string; suffix: string } {
  return {
    prefix: pageText.slice(Math.max(0, start - CONTEXT_CHARS), start),
    suffix: pageText.slice(end, end + CONTEXT_CHARS),
  };
}

/**
 * Acha a citação de uma evidência já salva, desempatando pelo contexto: tenta o trecho com
 * o texto de antes e de depois; sem sucesso, o trecho sozinho.
 */
export function relocate(pageText: string, quote: string, prefix: string, suffix: string): QuoteMatch | null {
  if (prefix || suffix) {
    const wide = locateQuote(pageText, `${prefix}${quote}${suffix}`);
    if (wide?.location === 'exact') {
      const inner = locateQuote(pageText.slice(wide.start, wide.end), quote);
      if (inner) return { start: wide.start + inner.start, end: wide.start + inner.end, location: inner.location };
    }
  }
  return locateQuote(pageText, quote);
}

// ---------------------------------------------------------------------------------------
// PDF

/** Página com menos letras e dígitos que isso é tratada como imagem (escaneada, figura). */
const MIN_PAGE_CHARS = 40;

export function detectTextLayer(pageTexts: readonly string[]): StudyDocument['textLayer'] {
  if (pageTexts.length === 0) return 'none';
  const empty = pageTexts.filter((text) => normalizeForMatch(text).norm.length < MIN_PAGE_CHARS).length;
  if (empty === pageTexts.length) return 'none';
  return empty > 0 ? 'partial' : 'ok';
}

// ---------------------------------------------------------------------------------------
// Conferência

export function sameValue(a: ExtractionValue | undefined, b: ExtractionValue | undefined): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && [...a].sort().every((item, index) => item === [...b].sort()[index]);
  }
  if (typeof a === 'string' && typeof b === 'string') return a.trim().toLowerCase() === b.trim().toLowerCase();
  return a === b;
}

/**
 * Status depois que o revisor muda a resposta. Com proposta da IA em aberto: igual a ela
 * é confirmar, diferente é corrigir. Uma resposta já conferida que muda de novo segue a
 * mesma regra contra a proposta — "corrigida" não volta a "confirmada" à toa. Sem IA, é manual.
 */
export function statusAfterAnswer(entry: TargetEvidence | undefined, value: ExtractionValue | null): VerificationStatus {
  const suggestion = entry?.suggestion;
  if (!suggestion) return 'manual';
  if (value === null) return entry.status === 'suggested' ? 'suggested' : 'rejected';
  return sameValue(value, suggestion.value) ? 'confirmed' : 'edited';
}

/** Uma proposta só pode ser aceita como veio se algum trecho dela foi achado no PDF. */
export function canAccept(entry: TargetEvidence | undefined): boolean {
  return !!entry?.suggestion && entry.evidence.some((item) => item.location !== 'not-found');
}

// ---------------------------------------------------------------------------------------
// Resposta da IA → valor do formulário

const YES = /^(sim|yes|true|verdadeiro|s|y)$/i;
const NO = /^(n[aã]o|no|false|falso|n)$/i;

function matchOption(raw: string, options: readonly string[]): string | undefined {
  const wanted = normalizeForMatch(raw).norm;
  return options.find((option) => normalizeForMatch(option).norm === wanted);
}

/**
 * Converte a resposta do modelo para o tipo do campo. Devolve `null` quando não cabe —
 * número que não é número, opção que não existe —, e a proposta é descartada em vez de
 * entrar no formulário com um valor que o campo não aceita.
 */
export function coerceToField(field: ExtractionField, raw: unknown): ExtractionValue | null {
  if (raw === null || raw === undefined || raw === '') return null;
  switch (field.type) {
    case 'number': {
      const number = typeof raw === 'number' ? raw : Number(String(raw).replace(/\s/g, '').replace(',', '.'));
      return Number.isFinite(number) ? number : null;
    }
    case 'boolean':
      if (typeof raw === 'boolean') return raw;
      if (YES.test(String(raw).trim())) return true;
      if (NO.test(String(raw).trim())) return false;
      return null;
    case 'date': {
      const text = String(raw).trim();
      if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
      if (/^\d{4}$/.test(text)) return `${text}-01-01`;
      if (/^\d{4}-\d{2}$/.test(text)) return `${text}-01`;
      return null;
    }
    case 'select':
      return matchOption(Array.isArray(raw) ? String(raw[0] ?? '') : String(raw), field.options) ?? null;
    case 'multiselect': {
      const items = (Array.isArray(raw) ? raw : String(raw).split(/[;,]/)).map(String);
      const picked = [...new Set(items.map((item) => matchOption(item, field.options)).filter((item): item is string => !!item))];
      return picked.length > 0 ? picked : null;
    }
    default: {
      const text = Array.isArray(raw) ? raw.join('; ') : String(raw);
      return text.trim() ? text.trim() : null;
    }
  }
}

/** Rótulo de resposta de qualidade → id dela. */
export function coerceToQualityAnswer(answers: readonly QualityAnswer[], raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const label = matchOption(raw, answers.map((answer) => answer.label));
  return answers.find((answer) => answer.label === label)?.id ?? null;
}

// ---------------------------------------------------------------------------------------
// Números do piloto

export interface PilotMetrics {
  studies: number;
  withPdf: number;
  scanned: number;
  /** Respostas que a IA propôs, e o que o revisor fez com elas. */
  suggested: number;
  pending: number;
  confirmed: number;
  edited: number;
  rejected: number;
  manual: number;
  /** Trechos citados pela IA e onde foram achados. */
  aiQuotes: number;
  exact: number;
  approximate: number;
  notFound: number;
  /** Respostas com ao menos um trecho que as sustenta. */
  answersWithEvidence: number;
}

export function pilotMetrics(review: ReviewState, studyKeys: readonly string[]): PilotMetrics {
  const metrics: PilotMetrics = {
    studies: studyKeys.length,
    withPdf: 0,
    scanned: 0,
    suggested: 0,
    pending: 0,
    confirmed: 0,
    edited: 0,
    rejected: 0,
    manual: 0,
    aiQuotes: 0,
    exact: 0,
    approximate: 0,
    notFound: 0,
    answersWithEvidence: 0,
  };
  for (const key of studyKeys) {
    const document = review.documents[key];
    if (document) {
      metrics.withPdf += 1;
      if (document.textLayer === 'none') metrics.scanned += 1;
    }
    for (const entry of Object.values(review.evidence[key] ?? {})) {
      if (!entry) continue;
      if (entry.suggestion) metrics.suggested += 1;
      if (entry.status === 'suggested') metrics.pending += 1;
      else metrics[entry.status] += 1;
      if (entry.evidence.length > 0) metrics.answersWithEvidence += 1;
      for (const item of entry.evidence) {
        if (item.origin !== 'ai') continue;
        metrics.aiQuotes += 1;
        if (item.location === 'exact') metrics.exact += 1;
        else if (item.location === 'approximate') metrics.approximate += 1;
        else metrics.notFound += 1;
      }
    }
  }
  return metrics;
}

// ---------------------------------------------------------------------------------------
// Envio em lote e exportação

/**
 * A qual estudo um PDF pertence: pelo DOI achado nas primeiras páginas ou, sem ele, pelo
 * título do registro aparecendo na primeira página. Títulos curtos demais não valem —
 * "Editorial" casaria com metade das revistas.
 */
export function matchPdfToStudy<T extends { key: string; doi: string; title: string }>(
  studies: readonly T[],
  pdf: { doi: string | null; firstPageText: string },
): T | null {
  if (pdf.doi) {
    const doi = normalizeDoi(pdf.doi);
    const byDoi = studies.find((study) => study.doi && normalizeDoi(study.doi) === doi);
    if (byDoi) return byDoi;
  }
  const page = normalizeForMatch(pdf.firstPageText).norm;
  if (!page) return null;
  return (
    studies.find((study) => {
      const title = normalizeForMatch(study.title).norm;
      return title.length >= 20 && page.includes(title);
    }) ?? null
  );
}

function normalizeDoi(value: string): string {
  return value.trim().toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, '').replace(/^doi:\s*/, '').replace(/[.,;)\]]+$/, '');
}

export interface EvidenceRowLabels {
  question: (target: EvidenceTargetKey) => string;
  answer: (key: string, target: EvidenceTargetKey) => string;
  suggestion: (target: EvidenceTargetKey, value: ExtractionValue) => string;
  status: (status: VerificationStatus) => string;
  origin: (origin: 'manual' | 'ai') => string;
  location: (location: QuoteLocation) => string;
}

/** Planilha das evidências: uma linha por trecho; resposta conferida sem trecho também aparece. */
export function evidenceRows(
  review: ReviewState,
  studies: readonly { key: string; title: string; doi: string }[],
  labels: EvidenceRowLabels,
): Record<string, string | number>[] {
  const rows: Record<string, string | number>[] = [];
  for (const study of studies) {
    for (const [target, entry] of Object.entries(review.evidence[study.key] ?? {}) as [EvidenceTargetKey, TargetEvidence | undefined][]) {
      if (!entry) continue;
      const base = {
        key: study.key,
        title: study.title,
        doi: study.doi,
        pdf: review.documents[study.key]?.name ?? '',
        question: labels.question(target),
        answer: labels.answer(study.key, target),
        ai_suggestion: entry.suggestion ? labels.suggestion(target, entry.suggestion.value) : '',
        ai_rationale: entry.suggestion?.rationale ?? '',
        ai_model: entry.suggestion?.model ?? '',
        verification: labels.status(entry.status),
        verified_at: entry.verifiedAt ?? '',
      };
      const items = entry.evidence.length > 0 ? entry.evidence : [null];
      for (const item of items) {
        rows.push({
          ...base,
          page: item?.page ?? '',
          quote: item ? item.quote || (item.rects.length ? '[área]' : '') : '',
          origin: item ? labels.origin(item.origin) : '',
          found_in_pdf: item ? labels.location(item.location) : '',
        });
      }
    }
  }
  return rows;
}
