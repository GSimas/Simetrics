import { describe, expect, it } from 'vitest';

import { continentOf } from '@/core/continents';
import { COUNTRIES } from '@/core/parsers/countries';

describe('continentOf', () => {
  it('cobre todo país do dicionário geográfico', () => {
    expect(COUNTRIES.filter((country) => continentOf(country) === null)).toEqual([]);
  });

  it('ignora maiúsculas e classifica os casos de fronteira', () => {
    expect(continentOf('Usa')).toBe('North America');
    expect(continentOf('Peoples R China')).toBe('Asia');
    expect(continentOf('Russia')).toBe('Europe');
    expect(continentOf('Egypt')).toBe('Africa');
    expect(continentOf('Atlantis')).toBeNull();
  });
});
