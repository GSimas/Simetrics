import { Fragment, useCallback, useMemo, type ReactNode } from 'react';

import { buildEntityIndex, findEntityMentions, type EntityIndex } from '@/core/entity-links';
import type { SearchOptions } from '@/core/search';
import { entityTypeLabel } from '@/lib/i18n/labels';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { openDossier } from '@/state/navigation.store';

// Um índice por conjunto de opções: montá-lo percorre todas as entidades da base.
const indexCache = new WeakMap<SearchOptions, EntityIndex>();

function indexFor(options: SearchOptions): EntityIndex {
  let index = indexCache.get(options);
  if (!index) indexCache.set(options, (index = buildEntityIndex(options)));
  return index;
}

/**
 * Função para o `renderText` do Markdown: cada nome da base citado na resposta vira um
 * link para o dossiê no Motor de Busca. Estável enquanto a base não muda — as mensagens
 * memoizadas não re-renderizam à toa.
 */
export function useEntityLinks(): ((text: string) => ReactNode) | undefined {
  const options = useDataset((state) => state.searchOptions);
  const locale = useLocale((state) => state.locale);
  const index = useMemo(() => (options ? indexFor(options) : null), [options]);

  const render = useCallback(
    (text: string): ReactNode => {
      if (!index) return text;
      const mentions = findEntityMentions(text, index);
      if (mentions.length === 0) return text;
      const parts: ReactNode[] = [];
      let cursor = 0;
      for (const { start, end, type, term } of mentions) {
        if (start > cursor) parts.push(<Fragment key={`t${cursor}`}>{text.slice(cursor, start)}</Fragment>);
        const label = locale === 'en' ? `Open the dossier of ${term}` : `Abrir o dossiê de ${term}`;
        parts.push(
          <button
            key={`e${start}`}
            type="button"
            onClick={() => openDossier(type, term)}
            title={`${label} · ${entityTypeLabel(type, locale)}`}
            className="inline cursor-pointer text-left underline decoration-highlight/50 decoration-dotted underline-offset-2 transition-colors hover:text-highlight hover:decoration-solid"
          >
            {text.slice(start, end)}
          </button>,
        );
        cursor = end;
      }
      if (cursor < text.length) parts.push(<Fragment key={`t${cursor}`}>{text.slice(cursor)}</Fragment>);
      return parts;
    },
    [index, locale],
  );

  return index ? render : undefined;
}
