import { useState, type KeyboardEvent } from 'react';
import { X } from 'lucide-react';

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
          className="inline-flex items-center gap-1 rounded-md border border-border bg-secondary px-2 py-0.5 font-mono text-xs"
        >
          {separator && index > 0 && <span className="text-[10px] text-muted-foreground">{separator}</span>}
          {item}
          <button
            type="button"
            onClick={() => onChange(items.filter((other) => other !== item))}
            aria-label={`${removeLabel}: ${item}`}
            className="text-muted-foreground hover:text-foreground"
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
        className="min-w-[10rem] flex-1 bg-transparent px-1 py-0.5 text-sm outline-none placeholder:text-muted-foreground"
      />
    </div>
  );
}
