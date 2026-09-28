import { useCallback, useMemo } from 'react';

import { Badge } from '@/components/ui/badge';
import { hasQualityChecklist, includedStudies, scoreStudy } from '@/core/review/quality';
import type { ReviewState } from '@/core/review/types';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useReview, useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';
import { EmptyStep, StudyWorkspace } from './StudyWorkspace';

export function ScoreBadge({ review, studyKey, copy }: { review: ReviewState; studyKey: string; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const { score, max, complete, passes } = scoreStudy(review, studyKey);
  if (!complete && score === 0) return null;
  const fmt = (value: number) => value.toLocaleString(locale === 'en' ? 'en' : 'pt-BR');
  return (
    <Badge
      variant={passes === true ? 'success' : passes === false ? 'destructive' : complete ? 'secondary' : 'outline'}
      className="px-1.5 py-0 text-[9.5px]"
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
  const studies = useMemo(() => includedStudies(records, review), [records, review]);
  const isDone = useCallback((key: string) => scoreStudy(review, key).complete, [review]);

  if (studies.length === 0) return <EmptyStep message={copy.noIncluded} />;
  if (!hasQualityChecklist(review)) {
    return <EmptyStep message={copy.noChecklist} action={{ label: copy.editInProtocol, onClick: onEditProtocol }} />;
  }

  return (
    <StudyWorkspace
      studies={studies}
      isDone={isDone}
      badge={(study) => <ScoreBadge review={review} studyKey={study.key} copy={copy} />}
      copy={copy}
      label={copy.steps.quality}
    >
      {(study) => {
        const responses = review.quality[study.key] ?? {};
        const result = scoreStudy(review, study.key);
        return (
          <div className="space-y-4">
            <ol className="space-y-3">
              {review.qualityQuestions.map((question, index) => (
                <li key={question.id} className="space-y-1.5">
                  <p className="text-sm">
                    <span className="mr-2 font-mono text-xs text-muted-foreground">Q{index + 1}</span>
                    {question.text || '—'}
                  </p>
                  <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={`Q${index + 1}`}>
                    {review.qualityAnswers.map((answer) => {
                      const active = responses[question.id] === answer.id;
                      return (
                        <button
                          key={answer.id}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          onClick={() => answerQuality(study.key, question.id, active ? null : answer.id)}
                          className={cn(
                            'rounded-md border px-3 py-1 text-xs transition-colors',
                            active ? 'border-highlight text-highlight' : 'border-border text-muted-foreground hover:text-foreground',
                          )}
                        >
                          {answer.label || '—'} <span className="font-mono text-[10px] opacity-70">({fmt(answer.weight)})</span>
                        </button>
                      );
                    })}
                  </div>
                </li>
              ))}
            </ol>
            <div className="flex flex-wrap items-center gap-3 border-t border-border/80 pt-3 text-sm">
              <span className="font-semibold tabular-nums">
                {copy.score}: {fmt(result.score)} / {fmt(result.max)}
              </span>
              <Badge variant={result.passes === true ? 'success' : result.passes === false ? 'destructive' : 'outline'}>
                {!result.complete
                  ? `${copy.incomplete} (${result.answered}/${result.total})`
                  : result.passes === true
                    ? copy.passes
                    : result.passes === false
                      ? copy.fails
                      : copy.assessedComplete}
              </Badge>
            </div>
          </div>
        );
      }}
    </StudyWorkspace>
  );
}
