import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import { fitTransform, findSimilarPairs, type TfidfOptions } from '@/core/ml/tfidf';
import { computeIndices } from '@/core/scientometrics';
import { pyRound } from '@/core/stats';
import { titleCase } from '@/core/text';

/**
 * Paridade das primitivas — ver `scripts/export_primitives.py`.
 *
 * Estas funções são a base de tudo: um erro de arredondamento ou de title case não
 * quebra nada visivelmente, só desloca silenciosamente todas as tabelas analíticas.
 */

interface Primitives {
  _meta: { baseYear: number };
  round: { value: number; digits: number; expected: number }[];
  titleCase: { input: string; expected: string }[];
  indices: {
    citations: number[];
    years: number[] | null;
    expected: { h: number; g: number; i10: number; m: number };
  }[];
  tfidf: {
    corpus: string[];
    configs: Record<
      string,
      {
        vocabulary: string[];
        idf: number[];
        rows: [number, number][][];
        pairs: [number, number, number][];
      }
    >;
  };
}

const fixture = JSON.parse(
  readFileSync(new URL('./primitives.json', import.meta.url), 'utf8'),
) as Primitives;

describe('pyRound', () => {
  it(`replica round() do Python em ${fixture.round.length} casos`, () => {
    const divergentes = fixture.round.filter(
      ({ value, digits, expected }) => pyRound(value, digits) !== expected,
    );

    expect(
      divergentes.slice(0, 5).map((c) => `${c.value} @${c.digits}: py=${c.expected}`),
    ).toEqual([]);
    expect(divergentes).toHaveLength(0);
  });

  it('desempata para o dígito par, e não para longe do zero', () => {
    // O caso que motiva a implementação em BigInt: 5/8 é exatamente 0.625, um empate real.
    // O Python devolve 0.62; `Number((0.625).toFixed(2))` devolve 0.63.
    expect(pyRound(0.625, 2)).toBe(0.62);
    expect(pyRound(0.375, 2)).toBe(0.38);
    expect(Number((0.625).toFixed(2))).toBe(0.63);
  });

  it('arredonda pelo valor binário exato, não pela representação curta', () => {
    // 2.675 é, em binário, 2.67499999999999982…, então arredonda para baixo.
    expect(pyRound(2.675, 2)).toBe(2.67);
  });
});

describe('titleCase', () => {
  it.each(fixture.titleCase)('replica str.title() para %j', ({ input, expected }) => {
    expect(titleCase(input)).toBe(expected);
  });

  it('trata dígitos como fronteira de palavra, igual ao CPython', () => {
    expect(titleCase('abc123def')).toBe('Abc123Def');
  });
});

describe('índices cientométricos', () => {
  it(`reproduz h, g, i10 e m em ${fixture.indices.length} distribuições`, () => {
    for (const [index, caso] of fixture.indices.entries()) {
      const actual = computeIndices(caso.citations, caso.years, fixture._meta.baseYear);
      expect(actual, `caso ${index}: ${JSON.stringify(caso.citations.slice(0, 8))}`).toEqual(
        caso.expected,
      );
    }
  });

  it('mantém m em zero quando o primeiro ano é posterior ao ano-base', () => {
    expect(computeIndices([50, 40, 30], [2030, 2030, 2030], 2026).m).toBe(0);
  });
});

describe('TF-IDF', () => {
  const configs: Record<string, TfidfOptions> = {
    unigram: { stopWords: true },
    bigram: { stopWords: true, ngramRange: [1, 2], minDf: 1 },
  };

  for (const [label, options] of Object.entries(configs)) {
    describe(label, () => {
      const expected = fixture.tfidf.configs[label];
      const model = fitTransform(fixture.tfidf.corpus, options);

      it('constrói o mesmo vocabulário, na mesma ordem', () => {
        const inverse = new Map([...model.vocabulary].map(([term, index]) => [index, term]));
        const vocabulary = Array.from({ length: inverse.size }, (_, i) => inverse.get(i));
        expect(vocabulary).toEqual(expected?.vocabulary);
      });

      it('calcula os mesmos pesos idf', () => {
        expect(Array.from(model.idf)).toHaveLength(expected?.idf.length ?? 0);
        (expected?.idf ?? []).forEach((value, index) => {
          expect(model.idf[index]).toBeCloseTo(value, 12);
        });
      });

      it('produz as mesmas linhas esparsas, normalizadas em L2', () => {
        (expected?.rows ?? []).forEach((expectedRow, docIndex) => {
          const vector = model.vectors[docIndex];
          expect(vector, `documento ${docIndex}`).toBeDefined();
          expect(Array.from(vector?.indices ?? []), `documento ${docIndex}: índices`).toEqual(
            expectedRow.map(([index]) => index),
          );
          expectedRow.forEach(([, weight], position) => {
            expect(vector?.values[position], `documento ${docIndex}[${position}]`).toBeCloseTo(
              weight,
              12,
            );
          });
        });
      });

      it('encontra os mesmos pares de cosseno que o scikit-learn', () => {
        const actual = findSimilarPairs(model.vectors, 0.15).sort(
          (left, right) => left.a - right.a || left.b - right.b,
        );

        expect(actual.map((pair) => [pair.a, pair.b])).toEqual(
          (expected?.pairs ?? []).map(([a, b]) => [a, b]),
        );
        (expected?.pairs ?? []).forEach(([, , score], index) => {
          expect(actual[index]?.score).toBeCloseTo(score, 12);
        });
      });
    });
  }
});
