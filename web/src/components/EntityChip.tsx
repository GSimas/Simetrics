import { ArrowUpRight } from 'lucide-react';

import { splitTokens } from '@/core/text';
import type { SearchEntityType } from '@/lib/types';
import { useLocale } from '@/state/locale.store';
import { openInSearch, resolveEntity } from '@/state/navigation.store';

export interface EntityChipProps {
  /** Texto exibido. */
  label: string;
  /** Termo procurado no acervo, quando difere do exibido ("Título (12 citações)" → "Título"). */
  term?: string;
  types: readonly SearchEntityType[];
}

/**
 * Chip que abre o perfil da entidade no Motor de Busca. Quando o termo não existe no
 * acervo carregado, cai para texto simples — um chip que não leva a lugar nenhum seria
 * uma promessa falsa.
 */
export function EntityChip({ label, term = label, types }: EntityChipProps) {
  const t = useLocale((state) => state.t);

  if (!label.trim()) return <span className="text-muted-foreground">—</span>;
  if (!resolveEntity(term, types)) {
    return (
      <span className="block max-w-80 truncate" title={label}>
        {label}
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => openInSearch(term, types)}
      title={`${t('open_profile')}: ${label}`}
      className="group inline-flex max-w-72 cursor-pointer items-center gap-1 border border-border px-2 py-0.5 text-left text-xs text-foreground transition-colors hover:border-highlight hover:text-highlight focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
    >
      <span className="truncate">{label}</span>
      <ArrowUpRight
        className="size-3 shrink-0 text-muted-foreground transition-colors group-hover:text-highlight"
        aria-hidden
      />
    </button>
  );
}

export interface EntityChipsProps {
  /** Lista de nomes, ou texto com vários separados por ";" (o formato das bases). */
  values: readonly string[] | unknown;
  types: readonly SearchEntityType[];
  /** Quantos chips mostrar antes de resumir em "+N". */
  max?: number;
}

/** Vários chips numa célula — autores, países, palavras-chave de um documento. */
export function EntityChips({ values, types, max = 3 }: EntityChipsProps) {
  const list = [...new Set(Array.isArray(values) ? (values as string[]) : splitTokens(values))];
  if (list.length === 0) return <span className="text-muted-foreground">—</span>;

  const shown = list.slice(0, max);
  const rest = list.length - shown.length;

  return (
    <span className="flex max-w-[28rem] flex-wrap items-center gap-1">
      {shown.map((value) => (
        <EntityChip key={value} label={value} types={types} />
      ))}
      {rest > 0 && (
        <span className="eyebrow px-1" title={list.slice(max).join('; ')}>
          +{rest}
        </span>
      )}
    </span>
  );
}
