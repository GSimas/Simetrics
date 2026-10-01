import { useCallback, useMemo } from 'react';

import { Badge } from '@/components/ui/badge';
import { qualityTarget } from '@/core/review/evidence';
import { hasQualityChecklist, includedStudies, scoreStudy } from '@/core/review/quality';
import type { QualityAnswer, ReviewState } from '@/core/review/types';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useReview, useReviewReadOnly, useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';
import { chipClass, useFlash, type Tone } from './motion';
import { ReadOnlyScope } from './parts';
import { DocumentSlot, EvidenceInline } from './evidence/DocumentSlot';
import { EvidenceBlock } from './evidence/EvidenceBlock';
import { EmptyStep, StudyWorkspace } from './StudyWorkspace';

/** A resposta de maior peso acende em verde, a de peso zero em vermelho, as do meio em âmbar. */
function answerTone(answer: QualityAnswer, answers: readonly QualityAnswer[]): Tone {
  const weights = answers.map((item) => item.weight);
  const max = Math.max(...weights);
  const min = Math.min(...weights);
  if (max === min) return 'highlight';
  if (answer.weight === max) return 'include';
  if (answer.weight === min) return 'exclude';
  return 'warning';
}

export function ScoreBadge({ review, studyKey, copy }: { review: ReviewState; studyKey: string; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const { score, max, complete, passes } = scoreStudy(review, studyKey);
  if (!complete && score === 0) return null;
  const fmt = (value: number) => value.toLocaleString(locale === 'en' ? 'en' : 'pt-BR');
  return (
    <Badge
      key={`${score}-${passes}`}
      variant={passes === true ? 'success' : passes === false ? 'destructive' : complete ? 'secondary' : 'outline'}
      className={cn(
        'px-1.5 py-0 text-[9.5px] animate-in fade-in-0 zoom-in-90 duration-200',
        passes === true && 'shadow-[0_0_10px_-3px_var(--glow-include)]',
        passes === false && 'shadow-[0_0_10px_-3px_var(--glow-exclude)]',
      )}
      title={passes === true ? copy.passes : passes === false ? copy.fails : complete ? undefined : copy.incomplete}
    >
      {fmt(score)}/{fmt(max)}
    </Badge>
  );
}

export function QualityPanel({ review, copy, onEditProtocol }: { review: ReviewState; copy: ReviewCopy; onEditProtocol: () => void }) {
  const locale = useLocale((state) => state.locale);
  const fmt = (value: number) => value.toLocaleString(locale === 'en' ? 'en' : 'pt-BR');
  const records = useScreeningRecords();
  const answerQuality = useReview((state) => state.answerQuality);
  const readOnly = useReviewReadOnly();
  const [flash, triggerFlash] = useFlash();
  const studies = useMemo(() => includedStudies(records, review), [records, review]);
  const isDone = useCallback((key: string) => scoreStudy(review, key).complete, [review]);

  if (studies.length === 0) return <EmptyStep message={copy.noIncluded} />;
  if (!hasQualityChecklist(review)) {
    return <EmptyStep message={copy.noChecklist} action={{ label: copy.editInProtocol, onClick: onEditProtocol }} />;
  }

  /** Responde e, se a avaliação ficou completa, acende o resultado: verde passa, vermelho não. */
  const answer = (key: string, questionId: string, answerId: string | null): void => {
    answerQuality(key, questionId, answerId);
    if (!answerId) return;
    const responses = { ...(review.quality[key] ?? {}), [questionId]: answerId };
    const result = scoreStudy({ ...review, quality: { ...review.quality, [key]: responses } }, key);
    if (result.complete) triggerFlash(result.passes === false ? 'exclude' : result.passes === true ? 'include' : 'neutral');
  };

  return (
    <div className="space-y-4">
    <EvidenceBlock review={review} studies={studies} />
    <StudyWorkspace
      studies={studies}
      isDone={isDone}
      badge={(study) => <ScoreBadge review={review} studyKey={study.key} copy={copy} />}
      copy={copy}
      label={copy.steps.quality}
      flash={flash}
    >
      {(study) => {
        const responses = review.quality[study.key] ?? {};
        const result = scoreStudy(review, study.key);
        return (
          <div className="space-y-4">
          <DocumentSlot studyKey={study.key} scope="quality" />
          <ReadOnlyScope readOnly={readOnly} className="space-y-4">
            <ol className="space-y-3">
              {review.qualityQuestions.map((question, index) => (
                <li key={question.id} className="space-y-1.5">
                  <p className="text-sm">
                    <span className="mr-2 font-mono text-xs text-muted-foreground">Q{index + 1}</span>
                    {question.text || '—'}
                  </p>
                  <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={`Q${index + 1}`}>
                    {review.qualityAnswers.map((item) => {
                      const active = responses[question.id] === item.id;
                      return (
                        <button
                          key={item.id}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          onClick={() => answer(study.key, question.id, active ? null : item.id)}
                          className={chipClass(active, answerTone(item, review.qualityAnswers))}
                        >
                          {item.label || '—'} <span className="font-mono text-[10px] opacity-70">({fmt(item.weight)})</span>
                        </button>
                      );
                    })}
                  </div>
                  <EvidenceInline studyKey={study.key} target={qualityTarget(question.id)} scope="quality" />
                </li>
              ))}
            </ol>
            <div className="flex flex-wrap items-center gap-3 border-t border-border/80 pt-3 text-sm">
              <span key={result.score} className="font-semibold tabular-nums animate-in fade-in-0 duration-200">
                {copy.score}: {fmt(result.score)} / {fmt(result.max)}
              </span>
              <Badge
                key={`${result.complete}-${result.passes}`}
                variant={result.passes === true ? 'success' : result.passes === false ? 'destructive' : 'outline'}
                className={cn(
                  'animate-in fade-in-0 zoom-in-90 duration-200',
                  result.passes === true && 'shadow-[0_0_14px_-4px_var(--glow-include)]',
                  result.passes === false && 'shadow-[0_0_14px_-4px_var(--glow-exclude)]',
                )}
              >
                {!result.complete
                  ? `${copy.incomplete} (${result.answered}/${result.total})`
                  : result.passes === true
                    ? copy.passes
                    : result.passes === false
                      ? copy.fails
                      : copy.assessedComplete}
              </Badge>
            </div>
          </ReadOnlyScope>
          </div>
        );
      }}
    </StudyWorkspace>
    </div>
  );
}
