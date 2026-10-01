import { describe, expect, it } from 'vitest';

import { matchEntities, type SearchOptions } from '@/core/search';

const options: SearchOptions = {
  documents: ['Knowledge management in organizations', 'A study of management'],
  authors: ['Silva, A.', 'Managers, B.'],
  countries: ['Brasil'],
  venues: [],
  keywords: ['gestão do conhecimento', 'Management', 'MANAGEMENT'],
  themes: [],
};

describe('matchEntities', () => {
  it('ranks name start, then word start, then substring, shorter first', () => {
    expect(matchEntities(options, 'manag').map((m) => m.term)).toEqual([
      'Management',
      'Managers, B.',
      'A study of management',
      'Knowledge management in organizations',
    ]);
  });

  it('collapses case variants of the same name', () => {
    expect(matchEntities(options, 'management').filter((m) => m.type === 'Palavra-chave')).toHaveLength(1);
  });

  it('matches inside a word last', () => {
    expect(matchEntities(options, 'ilva')).toEqual([{ type: 'Autor', term: 'Silva, A.' }]);
  });

  it('ignores case and accents, and needs two characters', () => {
    expect(matchEntities(options, 'GESTAO')).toEqual([{ type: 'Palavra-chave', term: 'gestão do conhecimento' }]);
    expect(matchEntities(options, 'b')).toEqual([]);
  });
});
