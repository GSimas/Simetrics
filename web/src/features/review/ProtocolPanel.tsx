import { useState } from 'react';
import { Check, Copy, Plus, Trash2 } from 'lucide-react';

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
import { useReview } from '@/state/review.store';
import { ChipInput } from './ChipInput';
import { Block } from './parts';
import { ExtractionFormEditor, QualityChecklistEditor } from './PlanningEditors';
import type { ReviewCopy } from './copy';

export function CopyButton({ text, copy }: { text: string; copy: ReviewCopy }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      disabled={!text}
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        });
      }}
    >
      {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
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
    <div className="space-y-2 rounded-lg border border-border/60 p-3">
      <div className="flex items-center gap-2">
        <Input
          value={concept.label}
          onChange={(event) => onChange({ ...concept, label: event.target.value })}
          placeholder={copy.conceptLabelPlaceholder}
          aria-label={copy.conceptLabelPlaceholder}
          autoFocus={autoFocus}
          className="h-8 text-sm font-medium"
        />
        <Button type="button" variant="ghost" size="icon" onClick={onRemove} aria-label={copy.remove} title={copy.remove}>
          <Trash2 aria-hidden />
        </Button>
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
  const items = review.criteria.filter((criterion) => criterion.kind === kind);
  const prefix = kind === 'inclusion' ? 'I' : 'E';

  return (
    <div className="space-y-2">
      <p className="eyebrow">{kind === 'inclusion' ? copy.inclusion : copy.exclusion}</p>
      {items.map((criterion, index) => (
        <div key={criterion.id} className="flex items-center gap-2">
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
            placeholder={copy.criterionPlaceholder}
            aria-label={`${prefix}${index + 1}`}
            autoFocus={criterion.id === focusId}
            className="h-8 text-sm"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => removeItem('criteria', criterion.id)}
            aria-label={copy.remove}
            title={copy.remove}
          >
            <Trash2 aria-hidden />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" size="sm" onClick={() => onAdded(addCriterion(kind))}>
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
      <div className="space-y-4">
        <Block title={copy.typeLabel} hint={copy.typeHints[review.type]}>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label htmlFor="review-type" className="text-xs">
                {copy.typeLabel}
              </Label>
              <Select value={review.type} onValueChange={(value) => setType(value as ReviewType)}>
                <SelectTrigger id="review-type" className="h-9">
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
              <p className="flex h-9 items-center font-mono text-sm text-highlight">
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
              value={review.objective}
              onChange={(event) => update({ objective: event.target.value })}
              placeholder={copy.objectivePlaceholder}
              rows={3}
              className="text-sm"
            />
          </div>
        </Block>

        <Block title={copy.frameworkLabel}>
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={copy.frameworkLabel}>
            {FRAMEWORKS.map((framework: Framework) => (
              <button
                key={framework}
                type="button"
                role="radio"
                aria-checked={review.framework === framework}
                onClick={() => update({ framework })}
                className={cn(
                  'rounded-md border px-3 py-1 font-mono text-xs transition-colors',
                  review.framework === framework
                    ? 'border-highlight text-highlight'
                    : 'border-border text-muted-foreground hover:text-foreground',
                )}
              >
                {copy.frameworks[framework]}
              </button>
            ))}
          </div>
          <div className="space-y-2">
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

        <Block title={copy.questionsLabel}>
          {review.questions.map((question, index) => (
            <div key={question.id} className="flex items-center gap-2">
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
                placeholder={copy.questionPlaceholder}
                aria-label={`Q${index + 1}`}
                autoFocus={question.id === focusId}
                className="h-8 text-sm"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => removeItem('questions', question.id)}
                aria-label={copy.remove}
                title={copy.remove}
              >
                <Trash2 aria-hidden />
              </Button>
            </div>
          ))}
          <Button type="button" variant="outline" size="sm" onClick={() => setFocusId(addQuestion())}>
            <Plus aria-hidden />
            {copy.addQuestion}
          </Button>
        </Block>

        <QualityChecklistEditor review={review} copy={copy} />
      </div>

      <div className="space-y-4">
        <Block title={copy.conceptsLabel} hint={copy.conceptsHint}>
          {review.concepts.map((concept, index) => (
            <div key={concept.id} className="space-y-2">
              {index > 0 && <p className="text-center font-mono text-[10px] text-muted-foreground">AND</p>}
              <ConceptEditor
                concept={concept}
                copy={copy}
                autoFocus={concept.id === focusId}
                onChange={(next) =>
                  update({ concepts: review.concepts.map((item) => (item.id === next.id ? next : item)) })
                }
                onRemove={() => removeItem('concepts', concept.id)}
              />
            </div>
          ))}
          <Button type="button" variant="outline" size="sm" onClick={() => setFocusId(addConcept())}>
            <Plus aria-hidden />
            {copy.addConcept}
          </Button>
        </Block>

        <Block title={copy.searchStringLabel}>
          <div className="flex flex-wrap gap-1.5">
            {SEARCH_TARGETS.map((target) => {
              const active = review.searchTargets.includes(target.id);
              return (
                <button
                  key={target.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => toggleTarget(target.id)}
                  className={cn(
                    'rounded-md border px-2.5 py-1 text-xs transition-colors',
                    active ? 'border-highlight text-highlight' : 'border-border text-muted-foreground hover:text-foreground',
                  )}
                >
                  {targetName(target.id, target.name)}
                </button>
              );
            })}
          </div>
          {SEARCH_TARGETS.filter((target) => review.searchTargets.includes(target.id)).map((target) => {
            const query = buildSearchString(review.concepts, target.id);
            return (
              <div key={target.id} className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="eyebrow">{targetName(target.id, target.name)}</p>
                  <CopyButton text={query} copy={copy} />
                </div>
                <pre className="whitespace-pre-wrap break-words rounded-lg border border-border/60 bg-muted/30 p-3 font-mono text-xs leading-relaxed">
                  {query || <span className="text-muted-foreground">{copy.searchStringEmpty}</span>}
                </pre>
              </div>
            );
          })}
        </Block>

        <Block title={copy.criteriaLabel} hint={copy.exclusionHint}>
          <CriteriaList kind="inclusion" review={review} copy={copy} focusId={focusId} onAdded={setFocusId} />
          <CriteriaList kind="exclusion" review={review} copy={copy} focusId={focusId} onAdded={setFocusId} />
        </Block>

        <ExtractionFormEditor review={review} copy={copy} />
      </div>
    </div>
  );
}
