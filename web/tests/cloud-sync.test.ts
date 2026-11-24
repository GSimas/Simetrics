import { describe, expect, it } from 'vitest';

import { planSync } from '@/lib/cloud-sync';

const at = (minute: number) => `2026-11-24T10:${String(minute).padStart(2, '0')}:00.000Z`;

describe('planSync', () => {
  it('o lado mais novo vence, só-local sobe, só-nuvem desce e excluído em outro aparelho sai daqui', () => {
    const local = [
      { id: 'iguais', updatedAt: at(5) },
      { id: 'local-mais-novo', updatedAt: at(9) },
      { id: 'nuvem-mais-nova', updatedAt: at(1) },
      { id: 'nunca-enviado', updatedAt: at(3) },
      { id: 'excluido-remoto', updatedAt: at(3) },
    ];
    const remote = new Map([
      ['iguais', at(5)],
      ['local-mais-novo', at(2)],
      ['nuvem-mais-nova', at(8)],
      ['so-na-nuvem', at(4)],
    ]);
    const synced = new Set(['iguais', 'local-mais-novo', 'nuvem-mais-nova', 'excluido-remoto']);

    expect(planSync(local, remote, synced)).toEqual([
      { id: 'local-mais-novo', action: 'upload' },
      { id: 'nuvem-mais-nova', action: 'download' },
      { id: 'nunca-enviado', action: 'upload' },
      { id: 'excluido-remoto', action: 'delete-local' },
      { id: 'so-na-nuvem', action: 'download' },
    ]);
  });

  it('compara instantes, não texto: o Postgres devolve outro formato de data', () => {
    const plan = planSync([{ id: 'a', updatedAt: '2026-11-24T10:05:00.000Z' }], new Map([['a', '2026-11-24 10:05:00+00']]), new Set());
    expect(plan).toEqual([]);
  });
});
