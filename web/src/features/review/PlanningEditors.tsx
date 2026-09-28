import { useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { defaultQualityAnswers, maxQualityScore } from '@/core/review/quality';
import { EXTRACTION_FIELD_TYPES, type ExtractionFieldType, type ReviewState } from '@/core/review/types';
import { useLocale } from '@/state/locale.store';
import { useReview } from '@/state/review.store';
import { ChipInput } from './ChipInput';
import type { ReviewCopy } from './copy';
import { Block } from './parts';

function RemoveButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button type="button" variant="ghost" size="icon" onClick={onClick} aria-label={label} title={label}>
      <Trash2 aria-hidden />
    </Button>
  );
}

export function QualityChecklistEditor({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const update = useReview((state) => state.update);
  const addQualityQuestion = useReview((state) => state.addQualityQuestion);
  const addQualityAnswer = useReview((state) => state.addQualityAnswer);
  const removeItem = useReview((state) => state.removeItem);
  const [focusId, setFocusId] = useState<string | null>(null);
  const max = maxQualityScore(review);

  return (
    <Block
      title={copy.qualityChecklist}
      hint={`${copy.qualityChecklistHint}${review.type === 'scoping' ? ` ${copy.qualityOptional}` : ''}`}
    >
      {review.qualityQuestions.map((question, index) => (
        <div key={question.id} className="flex items-center gap-2">
          <span className="w-8 shrink-0 font-mono text-xs text-muted-foreground">Q{index + 1}</span>
          <Input
            value={question.text}
            onChange={(event) =>
              update({
                qualityQuestions: review.qualityQuestions.map((item) =>
                  item.id === question.id ? { ...item, text: event.target.value } : item,
                ),
              })
            }
            placeholder={copy.qualityQuestionPlaceholder}
            aria-label={`Q${index + 1}`}
            autoFocus={question.id === focusId}
            className="h-8 text-sm"
          />
          <RemoveButton label={copy.remove} onClick={() => removeItem('qualityQuestions', question.id)} />
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => setFocusId(addQualityQuestion(() => defaultQualityAnswers(locale)))}
      >
        <Plus aria-hidden />
        {copy.addQualityQuestion}
      </Button>

      {review.qualityQuestions.length > 0 && (
        <div className="space-y-3 border-t border-border/80 pt-3">
          <p className="eyebrow">{copy.answersLabel}</p>
          {review.qualityAnswers.map((answer) => (
            <div key={answer.id} className="flex items-center gap-2">
              <Input
                value={answer.label}
                onChange={(event) =>
                  update({
                    qualityAnswers: review.qualityAnswers.map((item) =>
                      item.id === answer.id ? { ...item, label: event.target.value } : item,
                    ),
                  })
                }
                placeholder={copy.answerPlaceholder}
                aria-label={copy.answerPlaceholder}
                autoFocus={answer.id === focusId}
                className="h-8 text-sm"
              />
              <Input
                type="number"
                step="0.5"
                value={answer.weight}
                onChange={(event) => {
                  const weight = Number(event.target.value);
                  if (!Number.isFinite(weight)) return;
                  update({
                    qualityAnswers: review.qualityAnswers.map((item) => (item.id === answer.id ? { ...item, weight } : item)),
                  });
                }}
                aria-label={`${copy.weight}: ${answer.label}`}
                title={copy.weight}
                className="h-8 w-20 shrink-0 text-sm tabular-nums"
              />
              <RemoveButton label={copy.remove} onClick={() => removeItem('qualityAnswers', answer.id)} />
            </div>
          ))}
          <Button type="button" variant="outline" size="sm" onClick={() => setFocusId(addQualityAnswer())}>
            <Plus aria-hidden />
            {copy.addAnswer}
          </Button>

          <div className="grid gap-2 sm:grid-cols-[10rem_1fr] sm:items-center">
            <Label htmlFor="quality-cutoff" className="text-xs">
              {copy.cutoffLabel}
            </Label>
            <div className="space-y-1">
              <Input
                id="quality-cutoff"
                type="number"
                step="0.5"
                min={0}
                value={review.qualityCutoff ?? ''}
                onChange={(event) => {
                  const raw = event.target.value;
                  const value = Number(raw);
                  update({ qualityCutoff: raw === '' || !Number.isFinite(value) ? null : value });
                }}
                className="h-8 w-28 text-sm tabular-nums"
              />
              <p className="text-[11px] text-muted-foreground">
                {copy.cutoffHint.replace('{max}', max.toLocaleString(locale === 'en' ? 'en' : 'pt-BR'))}
              </p>
            </div>
          </div>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={review.excludeBelowCutoff}
              disabled={review.qualityCutoff === null}
              onChange={(event) => update({ excludeBelowCutoff: event.target.checked })}
              className="size-4 accent-[var(--highlight)]"
            />
            {copy.excludeBelowCutoff}
          </label>
        </div>
      )}
    </Block>
  );
}

const WITH_OPTIONS: ExtractionFieldType[] = ['select', 'multiselect'];

export function ExtractionFormEditor({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const update = useReview((state) => state.update);
  const addExtractionField = useReview((state) => state.addExtractionField);
  const removeItem = useReview((state) => state.removeItem);
  const [focusId, setFocusId] = useState<string | null>(null);

  const patchField = (id: string, patch: Partial<ReviewState['extractionFields'][number]>): void =>
    update({ extractionFields: review.extractionFields.map((field) => (field.id === id ? { ...field, ...patch } : field)) });

  return (
    <Block title={copy.extractionForm} hint={copy.extractionFormHint}>
      {review.extractionFields.map((field) => (
        <div key={field.id} className="space-y-2 rounded-lg border border-border/60 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={field.label}
              onChange={(event) => patchField(field.id, { label: event.target.value })}
              placeholder={copy.fieldPlaceholder}
              aria-label={copy.fieldPlaceholder}
              autoFocus={field.id === focusId}
              className="h-8 min-w-[12rem] flex-1 text-sm"
            />
            <Select value={field.type} onValueChange={(value) => patchField(field.id, { type: value as ExtractionFieldType })}>
              <SelectTrigger className="h-8 w-40 text-xs" aria-label={field.label}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EXTRACTION_FIELD_TYPES.map((type) => (
                  <SelectItem key={type} value={type}>
                    {copy.fieldTypes[type]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <RemoveButton label={copy.remove} onClick={() => removeItem('extractionFields', field.id)} />
          </div>
          {WITH_OPTIONS.includes(field.type) && (
            <ChipInput
              items={field.options}
              onChange={(options) => patchField(field.id, { options })}
              placeholder={copy.optionsPlaceholder}
              label={`${copy.optionsPlaceholder} — ${field.label}`}
              removeLabel={copy.remove}
            />
          )}
        </div>
      ))}
      <Button type="button" variant="outline" size="sm" onClick={() => setFocusId(addExtractionField())}>
        <Plus aria-hidden />
        {copy.addField}
      </Button>
    </Block>
  );
}
