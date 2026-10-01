import { useState } from 'react';
import { Plus } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { NumberInput } from '@/components/ui/number-input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { defaultQualityAnswers, maxQualityScore } from '@/core/review/quality';
import { EXTRACTION_FIELD_TYPES, type ExtractionFieldType, type ReviewState } from '@/core/review/types';
import { useLocale } from '@/state/locale.store';
import { useReview } from '@/state/review.store';
import { ChipInput } from './ChipInput';
import { addOnEnter } from './keys';
import type { ReviewCopy } from './copy';
import { ADD_BUTTON, ENTER, useExitRemove } from './motion';
import { AnimatedItem } from './motion-parts';
import { Block, RemoveButton } from './parts';
import { cn } from '@/lib/utils';

export function QualityChecklistEditor({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const update = useReview((state) => state.update);
  const addQualityQuestion = useReview((state) => state.addQualityQuestion);
  const addQualityAnswer = useReview((state) => state.addQualityAnswer);
  const removeItem = useReview((state) => state.removeItem);
  const [focusId, setFocusId] = useState<string | null>(null);
  const questions = useExitRemove();
  const answers = useExitRemove();
  const max = maxQualityScore(review);

  return (
    <Block
      target="quality-checklist"
      title={copy.qualityChecklist}
      hint={`${copy.qualityChecklistHint}${review.type === 'scoping' ? ` ${copy.qualityOptional}` : ''}`}
    >
      {review.qualityQuestions.map((question, index) => (
        <AnimatedItem key={question.id} leaving={questions.isLeaving(question.id)} className="flex items-center gap-2">
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
            onKeyDown={addOnEnter(() => setFocusId(addQualityQuestion(() => defaultQualityAnswers(locale))))}
            placeholder={copy.qualityQuestionPlaceholder}
            aria-label={`Q${index + 1}`}
            autoFocus={question.id === focusId}
            className="h-8 text-sm"
          />
          <RemoveButton
            label={copy.remove}
            onClick={() => questions.remove(question.id, () => removeItem('qualityQuestions', question.id))}
          />
        </AnimatedItem>
      ))}
      <Button
        type="button"
        variant="outline"
        size="sm"
        className={ADD_BUTTON}
        onClick={() => setFocusId(addQualityQuestion(() => defaultQualityAnswers(locale)))}
      >
        <Plus aria-hidden />
        {copy.addQualityQuestion}
      </Button>

      {review.qualityQuestions.length > 0 && (
        <div className={cn('space-y-3 border-t border-border/80 pt-3', ENTER)}>
          <p className="eyebrow">{copy.answersLabel}</p>
          {review.qualityAnswers.map((answer) => (
            <AnimatedItem key={answer.id} leaving={answers.isLeaving(answer.id)} className="flex items-center gap-2">
              <Input
                value={answer.label}
                onChange={(event) =>
                  update({
                    qualityAnswers: review.qualityAnswers.map((item) =>
                      item.id === answer.id ? { ...item, label: event.target.value } : item,
                    ),
                  })
                }
                onKeyDown={addOnEnter(() => setFocusId(addQualityAnswer()))}
                placeholder={copy.answerPlaceholder}
                aria-label={copy.answerPlaceholder}
                autoFocus={answer.id === focusId}
                className="h-8 text-sm"
              />
              <NumberInput
                step={0.5}
                value={answer.weight}
                onValueChange={(weight) => {
                  if (weight === null) return;
                  update({
                    qualityAnswers: review.qualityAnswers.map((item) => (item.id === answer.id ? { ...item, weight } : item)),
                  });
                }}
                aria-label={`${copy.weight}: ${answer.label}`}
                title={copy.weight}
                className="w-24 shrink-0"
                inputClassName="h-8 text-sm"
              />
              <RemoveButton
                label={copy.remove}
                onClick={() => answers.remove(answer.id, () => removeItem('qualityAnswers', answer.id))}
              />
            </AnimatedItem>
          ))}
          <Button type="button" variant="outline" size="sm" className={ADD_BUTTON} onClick={() => setFocusId(addQualityAnswer())}>
            <Plus aria-hidden />
            {copy.addAnswer}
          </Button>

          <div className="grid gap-2 sm:grid-cols-[10rem_1fr] sm:items-center">
            <Label htmlFor="quality-cutoff" className="text-xs">
              {copy.cutoffLabel}
            </Label>
            <div className="space-y-1">
              <NumberInput
                id="quality-cutoff"
                step={0.5}
                min={0}
                max={max || undefined}
                value={review.qualityCutoff}
                onValueChange={(qualityCutoff) => update({ qualityCutoff })}
                className="w-28"
                inputClassName="h-8 text-sm"
              />
              <p className="text-[11px] text-muted-foreground">
                {copy.cutoffHint.replace('{max}', max.toLocaleString(locale === 'en' ? 'en' : 'pt-BR'))}
              </p>
            </div>
          </div>
          <label
            className={cn(
              'flex items-center gap-2 rounded-md px-1 py-0.5 text-xs transition-colors duration-200',
              review.excludeBelowCutoff && 'text-exclude',
            )}
          >
            <Checkbox
              checked={review.excludeBelowCutoff}
              disabled={review.qualityCutoff === null}
              onCheckedChange={(excludeBelowCutoff) => update({ excludeBelowCutoff })}
              tone="exclude"
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
  const fields = useExitRemove();

  const patchField = (id: string, patch: Partial<ReviewState['extractionFields'][number]>): void =>
    update({ extractionFields: review.extractionFields.map((field) => (field.id === id ? { ...field, ...patch } : field)) });

  return (
    <Block title={copy.extractionForm} hint={copy.extractionFormHint} target="extraction-form">
      {review.extractionFields.map((field) => (
        <AnimatedItem key={field.id} leaving={fields.isLeaving(field.id)}>
        <div className="space-y-2 rounded-lg border border-border/60 p-3 transition-[border-color,box-shadow] duration-300 focus-within:border-highlight/50 focus-within:shadow-[0_0_24px_-14px_var(--highlight)]">
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={field.label}
              onChange={(event) => patchField(field.id, { label: event.target.value })}
              onKeyDown={addOnEnter(() => setFocusId(addExtractionField()))}
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
            <RemoveButton
              label={copy.remove}
              onClick={() => fields.remove(field.id, () => removeItem('extractionFields', field.id))}
            />
          </div>
          {WITH_OPTIONS.includes(field.type) && (
            <div className={ENTER}>
            <ChipInput
              items={field.options}
              onChange={(options) => patchField(field.id, { options })}
              placeholder={copy.optionsPlaceholder}
              label={`${copy.optionsPlaceholder} — ${field.label}`}
              removeLabel={copy.remove}
            />
            </div>
          )}
        </div>
        </AnimatedItem>
      ))}
      <Button type="button" variant="outline" size="sm" className={ADD_BUTTON} onClick={() => setFocusId(addExtractionField())}>
        <Plus aria-hidden />
        {copy.addField}
      </Button>
    </Block>
  );
}
