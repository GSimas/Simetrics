import { describe, expect, it } from 'vitest';

import { chordLayout } from '@/core/graph/chord';

describe('chordLayout', () => {
  it('põe todos os nós no círculo unitário, o primeiro no topo, sem repetir posição', () => {
    const layout = chordLayout(Array.from({ length: 12 }, (_, index) => ({ key: index, weight: index })));

    expect(layout).toHaveLength(12);
    for (const node of layout) expect(Math.hypot(node.x, node.y)).toBeCloseTo(1, 10);
    expect(layout[0]!.x).toBeCloseTo(0, 10);
    expect(layout[0]!.y).toBeCloseTo(-1, 10);
    expect(new Set(layout.map((node) => `${node.x.toFixed(6)},${node.y.toFixed(6)}`)).size).toBe(12);
  });

  it('agrupa por comunidade e ordena por peso dentro de cada uma', () => {
    const layout = chordLayout([
      { key: 'a', weight: 1, group: 1 },
      { key: 'b', weight: 9, group: 0 },
      { key: 'c', weight: 5, group: 1 },
      { key: 'd', weight: 3, group: 0 },
    ]);
    expect(layout.map((node) => node.key)).toEqual(['b', 'd', 'c', 'a']);
  });

  it('aceita lista vazia', () => {
    expect(chordLayout([])).toEqual([]);
  });
});
