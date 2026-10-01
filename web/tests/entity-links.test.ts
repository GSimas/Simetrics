import { describe, expect, it } from 'vitest';

import { buildEntityIndex, findEntityMentions } from '@/core/entity-links';
import type { SearchOptions } from '@/core/search';

const options: SearchOptions = {
  documents: ['Memes, memetics and marketing'],
  authors: ['Price, I.', 'Voelpel, SC'],
  countries: ['Brazil', 'United Kingdom'],
  venues: ['Journal of Change Management'],
  keywords: ['memetics', 'marketing', 'AI', '2020'],
  themes: [],
};
const index = buildEntityIndex(options);

const linked = (text: string) =>
  findEntityMentions(text, index).map((m) => `${text.slice(m.start, m.end)}=${m.type}`);

describe('findEntityMentions', () => {
  it('links authors, countries and venues ignoring case and punctuation', () => {
    expect(linked('Segundo PRICE I e Voelpel, SC, no Journal of Change Management, o United Kingdom lidera.')).toEqual([
      'PRICE I=Autor',
      'Voelpel, SC=Autor',
      'Journal of Change Management=Local de Publicação (Venue)',
      'United Kingdom=País',
    ]);
  });

  it('prefers the longest name, so a title is not split into keywords', () => {
    expect(linked('Leia "Memes, memetics and marketing" sobre memetics.')).toEqual([
      'Memes, memetics and marketing=Documento',
      'memetics=Palavra-chave',
    ]);
  });

  it('matches whole words only and skips short or numeric names', () => {
    expect(linked('Brazilian studies in 2020 about AI')).toEqual([]);
  });
});
