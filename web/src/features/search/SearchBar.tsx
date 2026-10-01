import { useId, useMemo, useState, type KeyboardEvent } from 'react';
import { Search, Sparkles, X } from 'lucide-react';

import { matchEntities, type EntityMatch } from '@/core/search';
import { entityTypeLabel, numberLocale } from '@/lib/i18n/labels';
import { cn } from '@/lib/utils';
import { useChat } from '@/state/chat.store';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';

/**
 * Caixa única do Motor de Busca, no espírito de um buscador: enquanto se digita, sugere
 * entidades (autor, país, venue, palavra-chave, tema, título) que abrem o dossiê; Enter
 * busca o texto nos documentos. Sem consulta nem dossiê aberto, ocupa a tela como início.
 */
export function SearchBar({ hero }: { hero: boolean }) {
  const { t, locale } = useLocale();
  const searchOptions = useDataset((state) => state.searchOptions);
  const documentCount = useDataset((state) => state.active?.length ?? 0);
  const query = useNavigation((state) => state.searchQuery);
  const selectEntity = useNavigation((state) => state.selectEntity);
  const listId = useId();

  const [draft, setDraft] = useState(query);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(-1);

  // A consulta mudou por fora (limpa ao trocar de base): a caixa acompanha. Ajuste
  // durante o render, como no resto do app, em vez de um efeito que renderiza duas vezes.
  const [seenQuery, setSeenQuery] = useState(query);
  if (seenQuery !== query) {
    setSeenQuery(query);
    setDraft(query);
  }

  const suggestions = useMemo(
    () => (searchOptions ? matchEntities(searchOptions, draft) : []),
    [searchOptions, draft],
  );
  const showList = open && suggestions.length > 0;

  const submit = (): void => {
    setOpen(false);
    const text = draft.trim();
    if (text) useNavigation.setState({ searchQuery: text, searchTerm: null });
  };

  const choose = (match: EntityMatch): void => {
    setOpen(false);
    selectEntity(match.type, match.term);
  };

  // IA como passo a mais: a mesma frase vira pergunta à Simi, sobre a base inteira.
  const askAi = (): void => {
    setOpen(false);
    useNavigation.setState({ chatOpen: true });
    if (draft.trim()) void useChat.getState().ask(draft);
  };

  const clear = (): void => {
    setDraft('');
    setHighlight(-1);
    useNavigation.setState({ searchQuery: '', searchTerm: null });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      if (suggestions.length === 0) return;
      event.preventDefault();
      setOpen(true);
      const step = event.key === 'ArrowDown' ? 1 : -1;
      // -1 volta ao texto digitado, como num buscador.
      setHighlight((current) => ((current + 1 + step + suggestions.length + 1) % (suggestions.length + 1)) - 1);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const picked = showList ? suggestions[highlight] : undefined;
      if (picked) choose(picked);
      else submit();
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  const nf = numberLocale(locale);
  const stats = searchOptions
    ? t('search_stats')
        .replace('{docs}', documentCount.toLocaleString(nf))
        .replace('{authors}', searchOptions.authors.length.toLocaleString(nf))
        .replace('{countries}', searchOptions.countries.length.toLocaleString(nf))
        .replace('{venues}', searchOptions.venues.length.toLocaleString(nf))
    : null;

  const box = (
    <div className="relative w-full" data-tour="search-picker">
      <form
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className={cn(
          'flex items-center gap-2 border border-border bg-card px-4 transition-[border-color,box-shadow] duration-200',
          'focus-within:border-highlight focus-within:shadow-[0_0_32px_-12px_var(--highlight)]',
          hero ? 'h-14 rounded-2xl' : 'h-12 rounded-xl',
          showList && 'rounded-b-none',
        )}
      >
        <Search className="size-5 shrink-0 text-muted-foreground" aria-hidden />
        <input
          type="search"
          role="combobox"
          aria-label={t('search_input_placeholder')}
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={showList && highlight >= 0 ? `${listId}-${highlight}` : undefined}
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            setHighlight(-1);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKeyDown}
          placeholder={t('search_input_placeholder')}
          autoFocus={hero}
          className={cn(
            'min-w-0 flex-1 bg-transparent outline-hidden placeholder:text-muted-foreground [&::-webkit-search-cancel-button]:hidden',
            hero ? 'text-lg' : 'text-base',
          )}
        />
        {draft && (
          <button
            type="button"
            onClick={clear}
            title={t('search_clear')}
            className="shrink-0 rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
          >
            <X className="size-4" aria-hidden />
          </button>
        )}
        <button
          type="submit"
          // No celular o teclado já traz a tecla de busca; o botão só apertaria o campo.
          className="eyebrow hidden shrink-0 rounded-md border sm:inline-flex border-border px-3 py-1.5 text-foreground transition-colors hover:border-highlight hover:text-highlight focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
        >
          {t('search_submit')}
        </button>
        <button
          type="button"
          onClick={askAi}
          data-tour="search-ask"
          title={t('search_ask_ai_hint')}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-highlight/10 px-2.5 py-1.5 text-xs font-semibold text-highlight transition-colors hover:bg-highlight/20 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
        >
          <Sparkles className="size-3.5" aria-hidden />
          <span className="hidden sm:inline">{t('search_ask_ai')}</span>
          <span className="sr-only sm:hidden">{t('search_ask_ai')}</span>
        </button>
      </form>

      <ul
        id={listId}
        role="listbox"
        aria-label={t('search_suggestions')}
        hidden={!showList}
        className="absolute inset-x-0 top-full z-30 max-h-80 overflow-y-auto rounded-b-xl border border-t-0 border-highlight bg-card py-1 shadow-2xl"
      >
        {suggestions.map((match, index) => (
          <li
            key={`${match.type}:${match.term}`}
            id={`${listId}-${index}`}
            role="option"
            aria-selected={index === highlight}
            // mousedown, e não click: o blur do campo fecharia a lista antes do clique.
            onMouseDown={(event) => {
              event.preventDefault();
              choose(match);
            }}
            onMouseEnter={() => setHighlight(index)}
            className={cn(
              'flex cursor-pointer items-center gap-3 px-4 py-2 text-sm',
              index === highlight ? 'bg-highlight/10 text-foreground' : 'text-muted-foreground',
            )}
          >
            <Search className="size-3.5 shrink-0" aria-hidden />
            <span className="min-w-0 flex-1 truncate text-foreground">{match.term}</span>
            <span className="eyebrow shrink-0">
              {match.type === 'Local de Publicação (Venue)' ? 'Venue' : entityTypeLabel(match.type, locale)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );

  if (!hero) return <div className="max-w-3xl">{box}</div>;

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 px-2 text-center">
      <div className="space-y-3">
        <p className="eyebrow">{t('tab_search')}</p>
        <h3 className="text-3xl font-medium leading-none tracking-[-0.05em] sm:text-5xl">
          {t('search_hero_a')} <em className="accent-serif text-highlight">{t('search_hero_em')}</em>
        </h3>
      </div>
      <div className="w-full max-w-2xl">{box}</div>
      <div className="space-y-1.5">
        {stats && <p className="eyebrow tabular-nums">{stats}</p>}
        <p className="text-sm text-muted-foreground">{t('search_hero_hint')}</p>
      </div>
    </div>
  );
}
