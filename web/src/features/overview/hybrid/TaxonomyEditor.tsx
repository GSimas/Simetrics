import { Plus, Trash2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import type { HybridCategory } from '@/core/hybrid/types';
import type { HybridCopy } from './copy';

interface TaxonomyEditorProps {
  categories: HybridCategory[];
  /** Quantos documentos da amostra o modelo gerativo pôs em cada categoria (por id). */
  support: Record<string, number>;
  onChange: (categories: HybridCategory[]) => void;
  copy: HybridCopy;
}

/** Edição das categorias antes da classificação: nome, o que é, o que não é. */
export function TaxonomyEditor({ categories, support, onChange, copy }: TaxonomyEditorProps) {
  const patch = (position: number, change: Partial<HybridCategory>) =>
    onChange(categories.map((category, i) => (i === position ? { ...category, ...change } : category)));

  const remove = (position: number) => onChange(categories.filter((_, i) => i !== position));

  const add = () => onChange([...categories, { id: '', name: '', what: '', notFor: '', examples: [] }]);

  return (
    <div className="space-y-3">
      <ol className="space-y-3">
        {categories.map((category, position) => {
          const fieldId = `hybrid-cat-${position}`;
          const count = category.id ? support[category.id] : undefined;
          return (
            <li key={position} className="space-y-2 rounded-xl border border-border/80 p-3">
              <div className="flex items-end gap-2">
                <div className="min-w-0 flex-1 space-y-1">
                  <Label htmlFor={`${fieldId}-name`} className="text-[11px] text-muted-foreground">
                    {position + 1}. {copy.name}
                  </Label>
                  <Input
                    id={`${fieldId}-name`}
                    value={category.name}
                    maxLength={60}
                    onChange={(event) => patch(position, { name: event.target.value })}
                    className="h-8 text-sm font-medium"
                  />
                </div>
                {count !== undefined && (
                  <Badge variant="purple" className="mb-1.5 shrink-0 tabular-nums">
                    {count} {copy.inSample}
                  </Badge>
                )}
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => remove(position)}
                  aria-label={`${copy.removeCategory} ${category.name}`}
                  className="shrink-0 text-muted-foreground hover:text-red-700"
                >
                  <Trash2 aria-hidden />
                </Button>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="space-y-1">
                  <Label htmlFor={`${fieldId}-what`} className="text-[11px] text-muted-foreground">{copy.what}</Label>
                  <Textarea
                    id={`${fieldId}-what`}
                    value={category.what}
                    rows={2}
                    onChange={(event) => patch(position, { what: event.target.value })}
                    className="min-h-14 text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor={`${fieldId}-not`} className="text-[11px] text-muted-foreground">{copy.notFor}</Label>
                  <Textarea
                    id={`${fieldId}-not`}
                    value={category.notFor}
                    rows={2}
                    onChange={(event) => patch(position, { notFor: event.target.value })}
                    className="min-h-14 text-xs"
                  />
                </div>
              </div>
              {category.examples.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">{copy.examples}</p>
                  <ul className="space-y-0.5 text-xs text-muted-foreground">
                    {category.examples.map((example) => (
                      <li key={example} className="truncate" title={example}>
                        · {example}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </li>
          );
        })}
      </ol>
      <Button type="button" variant="outline" size="sm" onClick={add}>
        <Plus aria-hidden />
        {copy.addCategory}
      </Button>
    </div>
  );
}
