import { Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

/**
 * Busca dentro de um gráfico (rede, mapa, cordas): destaca os elementos cujo rótulo
 * contém o texto digitado. A lógica de destaque fica em cada gráfico; aqui só a caixa e
 * a correspondência.
 */

interface ChartSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Quantos elementos a busca encontrou (`null` com a busca vazia). */
  found: number | null;
  className?: string;
}

export function ChartSearch({ value, onChange, found, className }: ChartSearchProps) {
  const en = useLocale((state) => state.locale === 'en');
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="relative w-full max-w-64">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input
          type="search"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') onChange('');
          }}
          placeholder={en ? 'Search in the chart…' : 'Buscar no gráfico…'}
          aria-label={en ? 'Search in the chart' : 'Buscar no gráfico'}
          className="h-8 bg-background/90 pl-8 pr-7 text-xs [&::-webkit-search-cancel-button]:hidden"
        />
        {value && (
          <button
            type="button"
            onClick={() => onChange('')}
            aria-label={en ? 'Clear search' : 'Limpar busca'}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 cursor-pointer p-0.5 text-muted-foreground hover:text-foreground"
          >
            <X className="size-3.5" aria-hidden />
          </button>
        )}
      </div>
      {found !== null && (
        <span className="eyebrow whitespace-nowrap text-muted-foreground" aria-live="polite">
          {found === 0
            ? en ? 'No match' : 'Nada encontrado'
            : en ? `${found} found` : `${found} encontrado${found > 1 ? 's' : ''}`}
        </span>
      )}
    </div>
  );
}
