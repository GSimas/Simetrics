import { describe, expect, it } from 'vitest';

import { matchKeys } from '@/components/charts/chart-search';

describe('chart search', () => {
  const items = [
    { key: 'br', label: 'Brasil' },
    { key: 'sp', label: 'São Paulo' },
    { key: 'us', label: 'United States' },
  ];
  const find = (query: string) => matchKeys(items, query, (item) => item.key, (item) => item.label);

  it('is null for an empty query and ignores accents and case', () => {
    expect(find('  ')).toBeNull();
    expect([...find('sao')!]).toEqual(['sp']);
    expect([...find('BRA')!]).toEqual(['br']);
    expect(find('xyz')!.size).toBe(0);
  });
});
