import { useCallback, useEffect, useRef, useState } from 'react';

import { aiQuestions, buildEvidencePrompt, parseEvidenceResponse, prepareArticle, type PreparedArticle } from '@/core/review/ai-evidence';
import { coerceToField, coerceToQualityAnswer, locateInPages, quoteContext } from '@/core/review/evidence';
import type { ExtractionValue } from '@/core/review/types';
import { aiDestination, requestEvidenceJson, type AiDestination } from '@/lib/evidence-ai';
import { getPdf } from '@/lib/pdf-store';
import { useLocale } from '@/state/locale.store';
import { useReview, type AiProposal, type NewEvidence } from '@/state/review.store';
import { fill, type EvidenceCopy } from './copy';

export interface AiMessage {
  tone: 'info' | 'error';
  text: string;
}

/**
 * A IA lê o PDF de um estudo e propõe respostas com o trecho que as sustenta. Cada trecho
 * citado é procurado no texto real do PDF antes de ser gravado: o que não for achado fica
 * marcado como "não localizado" e a resposta não pode ser aceita como veio.
 */
export function useEvidenceAi(studyKey: string, scope: 'extraction' | 'quality', title: string, copy: EvidenceCopy) {
  const [running, setRunning] = useState(false);
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [prepared, setPrepared] = useState<{ hash: string; article: PreparedArticle; destination: AiDestination } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const hash = useReview((state) => state.review?.documents[studyKey]?.hash);
  const pageLabel = useLocale((state) => (state.locale === 'en' ? 'Page' : 'Página'));

  const preview = prepared && prepared.hash === hash ? prepared : null;

  // O que vai ser enviado, calculado antes — o aviso mostra o tamanho e o destino.
  useEffect(() => {
    let cancelled = false;
    if (!hash) return;
    void getPdf(hash).then((stored) => {
      if (!cancelled && stored) setPrepared({ hash, article: prepareArticle(stored.pageTexts, pageLabel), destination: aiDestination() });
    });
    return () => {
      cancelled = true;
    };
  }, [hash, pageLabel]);

  useEffect(() => () => controller.current?.abort(), []);

  const run = useCallback(async () => {
    const review = useReview.getState().review;
    const document = review?.documents[studyKey];
    if (!review || !document) return;
    const stored = await getPdf(document.hash);
    if (!stored) return setMessages([{ tone: 'error', text: copy.missingFile }]);
    if (document.textLayer === 'none') return setMessages([{ tone: 'error', text: copy.aiNoText }]);
    const questions = aiQuestions(review, scope);
    if (questions.length === 0) return setMessages([{ tone: 'error', text: copy.aiNoQuestions }]);

    const locale = useLocale.getState().locale === 'en' ? 'en' : 'pt';
    const article = prepareArticle(stored.pageTexts, pageLabel);
    controller.current?.abort();
    controller.current = new AbortController();
    setRunning(true);
    setMessages([]);
    try {
      const response = await requestEvidenceJson(buildEvidencePrompt(questions, article, { title, locale }), controller.current.signal);
      const answers = parseEvidenceResponse(response.text, questions);
      const now = new Date().toISOString();
      let discarded = 0;
      const proposals: AiProposal[] = [];
      for (const answer of answers) {
        const [kind, id] = answer.target.split(':') as ['extraction' | 'quality', string];
        let value: ExtractionValue | null = null;
        if (kind === 'extraction') {
          const field = review.extractionFields.find((item) => item.id === id);
          value = field ? coerceToField(field, answer.answer) : null;
        } else {
          value = coerceToQualityAnswer(review.qualityAnswers, answer.answer);
        }
        if (value === null) {
          discarded += 1;
          continue;
        }
        const evidence: NewEvidence[] = answer.quotes.map((quote) => {
          const match = locateInPages(stored.pageTexts, quote.text, quote.page);
          if (!match) {
            const page = Math.min(Math.max(quote.page || 1, 1), stored.pageTexts.length || 1);
            return { page, quote: quote.text, prefix: '', suffix: '', rects: [], origin: 'ai', location: 'not-found' };
          }
          const pageText = stored.pageTexts[match.page - 1] ?? '';
          // O trecho guardado é o do PDF, não o que o modelo escreveu: é ele que se destaca.
          return {
            page: match.page,
            quote: pageText.slice(match.start, match.end),
            ...quoteContext(pageText, match.start, match.end),
            rects: [],
            origin: 'ai',
            location: match.location,
          };
        });
        proposals.push({ target: answer.target, suggestion: { value, rationale: answer.rationale, model: response.model, createdAt: now }, evidence });
      }
      useReview.getState().applyAiProposals(studyKey, proposals);

      const next: AiMessage[] = [
        { tone: 'info', text: proposals.length > 0 ? fill(copy.aiDone, { n: proposals.length }) : copy.aiNothing },
      ];
      if (discarded > 0) next.push({ tone: 'info', text: fill(copy.aiDiscarded, { n: discarded }) });
      if (article.truncated) next.push({ tone: 'info', text: fill(copy.aiTruncated, { chars: article.chars.toLocaleString() }) });
      setMessages(next);
    } catch (cause) {
      if ((cause as Error)?.name === 'AbortError') return;
      setMessages([{ tone: 'error', text: fill(copy.aiFailed, { error: cause instanceof Error ? cause.message : String(cause) }) }]);
    } finally {
      setRunning(false);
    }
  }, [studyKey, scope, title, copy, pageLabel]);

  return { run, running, messages, preview, refreshDestination: () => preview && setPrepared({ ...preview, destination: aiDestination() }) };
}
