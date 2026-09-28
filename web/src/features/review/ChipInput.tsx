import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { X } from 'lucide-react';

import { cn } from '@/lib/utils';
import { ENTER, EXIT, PRESS, useExitRemove } from './motion';

function splitItems(value: string): string[] {
  return value
    .split(/[,;\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/**
 * Lista editável de termos curtos: Enter ou vírgula confirma, Backspace no campo vazio
 * apaga o último, e colar uma lista separada por vírgulas cria vários de uma vez.
 */
export function ChipInput({
  items,
  onChange,
  placeholder,
  label,
  removeLabel,
  separator,
}: {
  items: string[];
  onChange: (items: string[]) => void;
  placeholder: string;
  label: string;
  removeLabel: string;
  /** Rótulo entre os itens (ex.: "OR" nos sinônimos da busca). */
  separator?: string;
}) {
  const [draft, setDraft] = useState('');
  const { isLeaving, remove } = useExitRemove();
  // A remoção só acontece ao fim da animação de saída: ela filtra a lista de então.
  const latest = useRef({ items, onChange });
  useEffect(() => {
    latest.current = { items, onChange };
  }, [items, onChange]);

  const commit = (value: string): void => {
    const incoming = splitItems(value).filter((item, index, all) => !items.includes(item) && all.indexOf(item) === index);
    if (incoming.length > 0) onChange([...items, ...incoming]);
    setDraft('');
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === 'Enter' || event.key === ',') {
      event.preventDefault();
      commit(draft);
    } else if (event.key === 'Backspace' && !draft && items.length > 0) {
      onChange(items.slice(0, -1));
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {items.map((item, index) => (
        <span
          key={item}
          className={cn(
            'inline-flex items-center gap-1 rounded-md border border-border bg-secondary px-2 py-0.5 font-mono text-xs transition-[border-color,box-shadow] duration-200 hover:border-highlight/50 hover:shadow-[0_0_14px_-8px_var(--highlight)]',
            isLeaving(item) ? EXIT : cn(ENTER, 'zoom-in-95'),
          )}
        >
          {separator && index > 0 && <span className="text-[10px] text-muted-foreground">{separator}</span>}
          {item}
          <button
            type="button"
            onClick={() => remove(item, () => latest.current.onChange(latest.current.items.filter((other) => other !== item)))}
            aria-label={`${removeLabel}: ${item}`}
            className={cn('rounded-sm text-muted-foreground hover:text-exclude', PRESS)}
          >
            <X className="size-3" aria-hidden />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        onBlur={() => commit(draft)}
        onPaste={(event) => {
          const text = event.clipboardData.getData('text');
          if (/[,;\n]/.test(text)) {
            event.preventDefault();
            commit(text);
          }
        }}
        placeholder={placeholder}
        aria-label={label}
        autoComplete="off"
        className="min-w-[10rem] flex-1 bg-transparent px-1 py-0.5 text-sm outline-none placeholder:text-muted-foreground"
      />
    </div>
  );
}
