import { useState } from 'react';
import { Check, Copy, Plus } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { buildSearchString, SEARCH_TARGETS } from '@/core/review/search-string';
import {
  FRAMEWORK_FIELDS,
  FRAMEWORKS,
  REVIEW_DEFAULTS,
  REVIEW_TYPES,
  type CriterionKind,
  type Framework,
  type ReviewState,
  type ReviewType,
  type SearchConcept,
} from '@/core/review/types';
import { cn } from '@/lib/utils';
import { useReview, useReviewReadOnly } from '@/state/review.store';
import { ChipInput } from './ChipInput';
import { addOnEnter } from './keys';
import type { ReviewCopy } from './copy';
import { ADD_BUTTON, chipClass, ENTER, useExitRemove } from './motion';
import { AnimatedItem } from './motion-parts';
import { Block, ReadOnlyScope, RemoveButton } from './parts';
import { ExtractionFormEditor, QualityChecklistEditor } from './PlanningEditors';

export function CopyButton({ text, copy }: { text: string; copy: ReviewCopy }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      disabled={!text}
      className={cn(
        ADD_BUTTON,
        copied && 'border-include text-include shadow-[0_0_18px_-6px_var(--glow-include)] hover:border-include hover:text-include',
      )}
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        });
      }}
    >
      {copied ? (
        <Check key="done" className="animate-in zoom-in-50 duration-200" aria-hidden />
      ) : (
        <Copy key="copy" className="animate-in fade-in-0 duration-200" aria-hidden />
      )}
      {copied ? copy.copied : copy.copy}
    </Button>
  );
}

function ConceptEditor({
  concept,
  copy,
  autoFocus,
  onChange,
  onRemove,
}: {
  concept: SearchConcept;
  copy: ReviewCopy;
  autoFocus: boolean;
  onChange: (concept: SearchConcept) => void;
  onRemove: () => void;
}) {
  return (
    <div className="space-y-2 rounded-lg border border-border/60 p-3 transition-[border-color,box-shadow] duration-300 focus-within:border-highlight/50 focus-within:shadow-[0_0_24px_-14px_var(--highlight)]">
      <div className="flex items-center gap-2">
        <Input
          value={concept.label}
          onChange={(event) => onChange({ ...concept, label: event.target.value })}
          placeholder={copy.conceptLabelPlaceholder}
          aria-label={copy.conceptLabelPlaceholder}
          autoFocus={autoFocus}
          className="h-8 text-sm font-medium"
        />
        <RemoveButton label={copy.remove} onClick={onRemove} />
      </div>
      <ChipInput
        items={concept.terms}
        onChange={(terms) => onChange({ ...concept, terms })}
        placeholder={copy.termsPlaceholder}
        label={`${copy.termsPlaceholder} — ${concept.label}`}
        removeLabel={copy.removeTerm}
        separator="OR"
      />
    </div>
  );
}

function CriteriaList({
  kind,
  review,
  copy,
  focusId,
  onAdded,
}: {
  kind: CriterionKind;
  review: ReviewState;
  copy: ReviewCopy;
  focusId: string | null;
  onAdded: (id: string) => void;
}) {
  const update = useReview((state) => state.update);
  const addCriterion = useReview((state) => state.addCriterion);
  const removeItem = useReview((state) => state.removeItem);
  const { isLeaving, remove } = useExitRemove();
  const items = review.criteria.filter((criterion) => criterion.kind === kind);
  const prefix = kind === 'inclusion' ? 'I' : 'E';

  return (
    <div className="space-y-2" data-review-target={kind}>
      <p className={cn('eyebrow', kind === 'inclusion' ? 'text-include' : 'text-exclude')}>
        {kind === 'inclusion' ? copy.inclusion : copy.exclusion}
      </p>
      {items.map((criterion, index) => (
        <AnimatedItem key={criterion.id} leaving={isLeaving(criterion.id)} className="flex items-center gap-2">
          <span className="w-8 shrink-0 font-mono text-xs text-muted-foreground">
            {prefix}
            {index + 1}
          </span>
          <Input
            value={criterion.text}
            onChange={(event) =>
              update({
                criteria: review.criteria.map((item) =>
                  item.id === criterion.id ? { ...item, text: event.target.value } : item,
                ),
              })
            }
            onKeyDown={addOnEnter(() => onAdded(addCriterion(kind)))}
            placeholder={copy.criterionPlaceholder}
            aria-label={`${prefix}${index + 1}`}
            autoFocus={criterion.id === focusId}
            className="h-8 text-sm"
          />
          <RemoveButton label={copy.remove} onClick={() => remove(criterion.id, () => removeItem('criteria', criterion.id))} />
        </AnimatedItem>
      ))}
      <Button type="button" variant="outline" size="sm" className={ADD_BUTTON} onClick={() => onAdded(addCriterion(kind))}>
        <Plus aria-hidden />
        {kind === 'inclusion' ? copy.addInclusion : copy.addExclusion}
      </Button>
    </div>
  );
}

export function ProtocolPanel({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const update = useReview((state) => state.update);
  const setType = useReview((state) => state.setType);
  const addQuestion = useReview((state) => state.addQuestion);
  const addConcept = useReview((state) => state.addConcept);
  const removeItem = useReview((state) => state.removeItem);
  const readOnly = useReviewReadOnly();
  const questions = useExitRemove();
  const concepts = useExitRemove();
  // Item recém-adicionado: o campo dele nasce com o foco.
  const [focusId, setFocusId] = useState<string | null>(null);

  const targetName = (id: string, name: string) => (id === 'generic' ? copy.genericTarget : name);
  const toggleTarget = (id: string): void => {
    const selected = review.searchTargets.includes(id)
      ? review.searchTargets.filter((target) => target !== id)
      : [...review.searchTargets, id];
    update({ searchTargets: SEARCH_TARGETS.map((target) => target.id).filter((target) => selected.includes(target)) });
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ReadOnlyScope readOnly={readOnly} className="space-y-4">
        <Block title={copy.typeLabel} hint={copy.typeHints[review.type]} tour="review-protocol">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label htmlFor="review-type" className="text-xs">
                {copy.typeLabel}
              </Label>
              <Select value={review.type} onValueChange={(value) => setType(value as ReviewType)}>
                <SelectTrigger id="review-type" className="h-9 transition-[border-color,box-shadow] duration-200 hover:border-highlight/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {REVIEW_TYPES.map((type) => (
                    <SelectItem key={type} value={type}>
                      {copy.types[type]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium">{copy.guideline}</p>
              <p key={review.type} className={cn('flex h-9 items-center font-mono text-sm text-highlight', ENTER)}>
                {REVIEW_DEFAULTS[review.type].guideline}
              </p>
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="review-title" className="text-xs">
              {copy.titleLabel}
            </Label>
            <Input
              id="review-title"
              data-review-target="title"
              value={review.title}
              onChange={(event) => update({ title: event.target.value })}
              placeholder={copy.titlePlaceholder}
              className="h-9 text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="review-objective" className="text-xs">
              {copy.objectiveLabel}
            </Label>
            <Textarea
              id="review-objective"
              data-review-target="objective"
              value={review.objective}
              onChange={(event) => update({ objective: event.target.value })}
              placeholder={copy.objectivePlaceholder}
              rows={3}
              className="text-sm"
            />
          </div>
        </Block>

        <Block title={copy.frameworkLabel} target="framework">
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={copy.frameworkLabel}>
            {FRAMEWORKS.map((framework: Framework) => (
              <button
                key={framework}
                type="button"
                role="radio"
                aria-checked={review.framework === framework}
                onClick={() => update({ framework })}
                className={cn(chipClass(review.framework === framework), 'font-mono')}
              >
                {copy.frameworks[framework]}
              </button>
            ))}
          </div>
          {/* Trocar a estrutura troca os campos: eles entram de novo, sem saltar. */}
          <div key={review.framework} className={cn('space-y-2', ENTER)}>
            {FRAMEWORK_FIELDS[review.framework].map((field) => (
              <div key={field} className="grid gap-1 sm:grid-cols-[10rem_1fr] sm:items-center sm:gap-3">
                <Label htmlFor={`fw-${field}`} className="text-xs">
                  {copy.frameworkFields[field] ?? field}
                </Label>
                <Input
                  id={`fw-${field}`}
                  value={review.frameworkValues[field] ?? ''}
                  onChange={(event) =>
                    update({ frameworkValues: { ...review.frameworkValues, [field]: event.target.value } })
                  }
                  className="h-8 text-sm"
                />
              </div>
            ))}
          </div>
        </Block>

        <Block title={copy.questionsLabel} target="question">
          {review.questions.map((question, index) => (
            <AnimatedItem key={question.id} leaving={questions.isLeaving(question.id)} className="flex items-center gap-2">
              <span className="w-8 shrink-0 font-mono text-xs text-muted-foreground">Q{index + 1}</span>
              <Input
                value={question.text}
                onChange={(event) =>
                  update({
                    questions: review.questions.map((item) =>
                      item.id === question.id ? { ...item, text: event.target.value } : item,
                    ),
                  })
                }
                onKeyDown={addOnEnter(() => setFocusId(addQuestion()))}
                placeholder={copy.questionPlaceholder}
                aria-label={`Q${index + 1}`}
                autoFocus={question.id === focusId}
                className="h-8 text-sm"
              />
              <RemoveButton
                label={copy.remove}
                onClick={() => questions.remove(question.id, () => removeItem('questions', question.id))}
              />
            </AnimatedItem>
          ))}
          <Button type="button" variant="outline" size="sm" className={ADD_BUTTON} onClick={() => setFocusId(addQuestion())}>
            <Plus aria-hidden />
            {copy.addQuestion}
          </Button>
        </Block>

        <QualityChecklistEditor review={review} copy={copy} />
      </ReadOnlyScope>

      <div className="space-y-4">
        <ReadOnlyScope readOnly={readOnly}>
          <Block title={copy.conceptsLabel} hint={copy.conceptsHint} target="concept">
            {review.concepts.map((concept, index) => (
              <AnimatedItem key={concept.id} leaving={concepts.isLeaving(concept.id)} className="space-y-2">
                {index > 0 && <p className="text-center font-mono text-[10px] text-muted-foreground">AND</p>}
                <ConceptEditor
                  concept={concept}
                  copy={copy}
                  autoFocus={concept.id === focusId}
                  onChange={(next) =>
                    update({ concepts: review.concepts.map((item) => (item.id === next.id ? next : item)) })
                  }
                  onRemove={() => concepts.remove(concept.id, () => removeItem('concepts', concept.id))}
                />
              </AnimatedItem>
            ))}
            <Button type="button" variant="outline" size="sm" className={ADD_BUTTON} onClick={() => setFocusId(addConcept())}>
              <Plus aria-hidden />
              {copy.addConcept}
            </Button>
          </Block>
        </ReadOnlyScope>

        {/* Fora do escopo travado: no exemplo as strings continuam copiáveis. */}
        <Block title={copy.searchStringLabel} tour="review-search">
          <div className="flex flex-wrap gap-1.5">
            {SEARCH_TARGETS.map((target) => {
              const active = review.searchTargets.includes(target.id);
              return (
                <button
                  key={target.id}
                  type="button"
                  aria-pressed={active}
                  disabled={readOnly}
                  onClick={() => toggleTarget(target.id)}
                  className={chipClass(active)}
                >
                  {targetName(target.id, target.name)}
                </button>
              );
            })}
          </div>
          {SEARCH_TARGETS.filter((target) => review.searchTargets.includes(target.id)).map((target) => {
            const query = buildSearchString(review.concepts, target.id);
            return (
              <div key={target.id} className={cn('space-y-1.5', ENTER)}>
                <div className="flex items-center justify-between gap-2">
                  <p className="eyebrow">{targetName(target.id, target.name)}</p>
                  <CopyButton text={query} copy={copy} />
                </div>
                <pre
                  key={query}
                  className="whitespace-pre-wrap break-words rounded-lg border border-border/60 bg-muted/30 p-3 font-mono text-xs leading-relaxed animate-in fade-in-0 duration-300"
                >
                  {query || <span className="text-muted-foreground">{copy.searchStringEmpty}</span>}
                </pre>
              </div>
            );
          })}
        </Block>

        <ReadOnlyScope readOnly={readOnly} className="space-y-4">
          <Block title={copy.criteriaLabel} hint={copy.exclusionHint}>
            <CriteriaList kind="inclusion" review={review} copy={copy} focusId={focusId} onAdded={setFocusId} />
            <CriteriaList kind="exclusion" review={review} copy={copy} focusId={focusId} onAdded={setFocusId} />
          </Block>

          <ExtractionFormEditor review={review} copy={copy} />
        </ReadOnlyScope>
      </div>
    </div>
  );
}
