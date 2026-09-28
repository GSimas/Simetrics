import { useCallback, useMemo } from 'react';
import { Check, RotateCcw } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { finalSelection } from '@/core/review/quality';
import type { ExtractionField, ExtractionValue, ReviewState } from '@/core/review/types';
import { cn } from '@/lib/utils';
import { useReview, useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';
import { EmptyStep, StudyWorkspace } from './StudyWorkspace';

function Choice({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        'rounded-md border px-3 py-1 text-xs transition-colors',
        active ? 'border-highlight text-highlight' : 'border-border text-muted-foreground hover:text-foreground',
      )}
    >
      {children}
    </button>
  );
}

function FieldInput({
  field,
  value,
  onChange,
  copy,
}: {
  field: ExtractionField;
  value: ExtractionValue | undefined;
  onChange: (value: ExtractionValue | null) => void;
  copy: ReviewCopy;
}) {
  const id = `extraction-${field.id}`;
  switch (field.type) {
    case 'number':
      return (
        <Input
          id={id}
          type="number"
          value={typeof value === 'number' ? value : ''}
          onChange={(event) => {
            const parsed = Number(event.target.value);
            onChange(event.target.value === '' || !Number.isFinite(parsed) ? null : parsed);
          }}
          className="h-8 w-40 text-sm tabular-nums"
        />
      );
    case 'date':
      return (
        <Input
          id={id}
          type="date"
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange(event.target.value || null)}
          className="h-8 w-48 text-sm"
        />
      );
    case 'boolean':
      return (
        <div className="flex gap-1.5" role="group" aria-labelledby={`${id}-label`}>
          <Choice active={value === true} onClick={() => onChange(value === true ? null : true)}>
            {copy.yes}
          </Choice>
          <Choice active={value === false} onClick={() => onChange(value === false ? null : false)}>
            {copy.no}
          </Choice>
        </div>
      );
    case 'select':
    case 'multiselect': {
      const selected = Array.isArray(value) ? value : typeof value === 'string' ? [value] : [];
      const toggle = (option: string): void => {
        if (field.type === 'select') onChange(selected.includes(option) ? null : option);
        else onChange(selected.includes(option) ? selected.filter((item) => item !== option) : [...selected, option]);
      };
      if (field.options.length === 0) return <p className="text-xs text-muted-foreground">{copy.noForm}</p>;
      return (
        <div className="flex flex-wrap gap-1.5" role="group" aria-labelledby={`${id}-label`}>
          {field.options.map((option) => (
            <Choice key={option} active={selected.includes(option)} onClick={() => toggle(option)}>
              {option}
            </Choice>
          ))}
        </div>
      );
    }
    default:
      return (
        <Textarea
          id={id}
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange(event.target.value)}
          rows={2}
          className="text-sm"
        />
      );
  }
}

export function ExtractionPanel({ review, copy, onEditProtocol }: { review: ReviewState; copy: ReviewCopy; onEditProtocol: () => void }) {
  const records = useScreeningRecords();
  const setExtractionValue = useReview((state) => state.setExtractionValue);
  const setExtractionDone = useReview((state) => state.setExtractionDone);
  const studies = useMemo(() => finalSelection(records, review), [records, review]);
  const isDone = useCallback((key: string) => review.extraction[key]?.done === true, [review]);

  if (studies.length === 0) return <EmptyStep message={copy.noIncluded} />;
  if (review.extractionFields.length === 0) {
    return <EmptyStep message={copy.noForm} action={{ label: copy.editInProtocol, onClick: onEditProtocol }} />;
  }

  return (
    <StudyWorkspace
      studies={studies}
      isDone={isDone}
      badge={(study) =>
        isDone(study.key) ? (
          <Badge variant="success" className="px-1.5 py-0 text-[9.5px]">
            {copy.markedDone}
          </Badge>
        ) : null
      }
      copy={copy}
      label={copy.steps.extraction}
    >
      {(study) => {
        const extraction = review.extraction[study.key];
        const done = extraction?.done === true;
        return (
          <div className="space-y-4">
            {review.extractionFields.map((field) => (
              <div key={field.id} className="space-y-1.5">
                <label id={`extraction-${field.id}-label`} htmlFor={`extraction-${field.id}`} className="text-xs font-medium">
                  {field.label || '—'}
                </label>
                <FieldInput
                  field={field}
                  value={extraction?.values[field.id]}
                  onChange={(value) => setExtractionValue(study.key, field.id, value)}
                  copy={copy}
                />
              </div>
            ))}
            <div className="border-t border-border/80 pt-3">
              {done ? (
                <Button type="button" variant="outline" onClick={() => setExtractionDone(study.key, false)}>
                  <RotateCcw aria-hidden />
                  {copy.reopen}
                </Button>
              ) : (
                <Button type="button" onClick={() => setExtractionDone(study.key, true)}>
                  <Check aria-hidden />
                  {copy.markDone}
                </Button>
              )}
            </div>
          </div>
        );
      }}
    </StudyWorkspace>
  );
}
