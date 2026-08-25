import { readFileSync } from 'node:fs';
import { fileURLToPath, URL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { dedupBoth, dedupByDoi, dedupBySimilarity } from '@/core/dedup';
import { processRisFiles, type RisSource } from '@/core/parsers/pipeline-ris';
import { bibliometrixMetrics, metadataCompleteness, summarize } from '@/core/summary';
import { authorsTable, countriesTable, keywordsTable, venuesTable } from '@/core/tables';
import type { EntityRow } from '@/core/tables';
import type { Dataset } from '@/lib/types';

/**
 * Paridade numérica com o pipeline Streamlit original.
 *
 * O golden.json é gerado por `scripts/export_golden.py`, que executa as funções reais de
 * utils.py sobre os mesmos três arquivos .ris. Regenere-o sempre que o Python mudar:
 *
 *     .venv/bin/python scripts/export_golden.py
 */

interface GoldenEntityRow {
  entity: string;
  citations: number;
  h: number;
  g: number;
  i10: number;
  m: number;
  meanCitations: number;
  medianCitations: number;
  stdCitations: number;
}

interface Golden {
  _meta: { baseYear: number; arquivos: Record<string, string>; totalDocumentos: number };
  dedupDoi: { kept: number; removed: number };
  dedupSimilaridade: { kept: number; removed: number };
  bibliometrix: Record<string, number>;
  resumo: Record<string, number | string | null>;
  completude: { field: string; missing: number; missingPct: number; status: string }[];
  tabelas: Record<string, GoldenEntityRow[]>;
}

const repoRoot = fileURLToPath(new URL('../../..', import.meta.url));
const golden = JSON.parse(
  readFileSync(new URL('./golden.json', import.meta.url), 'utf8'),
) as Golden;

const BASE_YEAR = golden._meta.baseYear;

const sources: RisSource[] = Object.entries(golden._meta.arquivos).map(([name, database]) => ({
  name,
  database,
  text: readFileSync(`${repoRoot}/${name}`, 'utf8'),
}));

const dataset: Dataset = processRisFiles(sources);

describe('ingestão RIS', () => {
  it('integra o mesmo número de documentos que o pandas', () => {
    expect(dataset.length).toBe(golden._meta.totalDocumentos);
  });
});

describe('deduplicação', () => {
  it('remove exatamente as mesmas duplicatas por DOI', () => {
    const { kept, removed } = dedupByDoi(dataset);
    expect(kept.length).toBe(golden.dedupDoi.kept);
    expect(removed.length).toBe(golden.dedupDoi.removed);
  });

  it('remove ao menos as duplicatas de título exato encontradas pelo Python', () => {
    // Divergência esperada, e é correção de bug — não regressão.
    //
    // No Python, `deduplicar_por_similaridade` monta o TfidfVectorizer com
    // `token_pattern=None` e sem `tokenizer`, o que levanta TypeError no scikit-learn.
    // Como a chamada está dentro de um `except Exception: pass` (utils.py:2954), a falha
    // é silenciosa: a etapa de similaridade NUNCA roda e o app só dedupe título exato.
    //
    // A versão TypeScript executa a etapa de fato, então remove um superconjunto.
    const { kept, removed } = dedupBySimilarity(dataset, { threshold: 0.9 });

    expect(removed.length).toBeGreaterThanOrEqual(golden.dedupSimilaridade.removed);
    expect(kept.length + removed.length).toBe(dataset.length);
  });

  it('executa deduplicação dupla combinando DOI e similaridade', () => {
    const { kept, removed } = dedupBoth(dataset, { threshold: 0.9 });
    const doiResult = dedupByDoi(dataset);

    // Deve remover no mínimo tantas duplicatas quanto a remoção por DOI
    expect(removed.length).toBeGreaterThanOrEqual(doiResult.removed.length);
    expect(kept.length + removed.length).toBe(dataset.length);
  });
});

describe('métricas bibliométricas', () => {
  const metrics = bibliometrixMetrics(dataset, BASE_YEAR);

  it.each([
    'growthRate',
    'mcp',
    'scp',
    'coauthIndex',
    'singleAuthorDocs',
    'avgCitPerYear',
  ])('%s bate com o Bibliometrix do Python', (key) => {
    expect(metrics[key as keyof typeof metrics]).toBe(golden.bibliometrix[key]);
  });
});

describe('resumo estrutural', () => {
  const summary = summarize(dataset, BASE_YEAR);

  it.each([
    'totalDocs',
    'timespan',
    'avgAge',
    'authorsCount',
    'countriesCount',
    'keywordsCount',
  ])('%s bate com o resumo do Python', (key) => {
    expect(summary[key as keyof typeof summary]).toBe(golden.resumo[key]);
  });

  /**
   * A contagem de venues diverge pelo mesmo motivo da tabela de venues: o `nunique()` do
   * Python opera sobre o valor bruto e conta a mesma revista mais de uma vez quando a
   * capitalização difere entre bases. Aqui a normalização vem antes da contagem, então o
   * total é sempre menor ou igual — e tem de casar exatamente com o número de linhas da
   * tabela, senão o indicador contradiz a tabela logo abaixo dele.
   */
  it('venuesCount desduplica capitalização e casa com a tabela de venues', () => {
    const fromTable = venuesTable(dataset, BASE_YEAR).length;

    expect(summary.venuesCount).toBe(fromTable);
    expect(summary.venuesCount).toBeLessThanOrEqual(golden.resumo['venuesCount'] as number);
  });
});

describe('completude de metadados', () => {
  const report = metadataCompleteness(dataset);

  it('reporta os mesmos campos, na mesma ordem', () => {
    expect(report.map((row) => row.field)).toEqual(golden.completude.map((row) => row.field));
  });

  it('conta os mesmos faltantes e atribui o mesmo status', () => {
    for (const [index, expected] of golden.completude.entries()) {
      const actual = report[index];
      expect(actual, `campo ${expected.field}`).toBeDefined();
      expect(actual?.missing, `faltantes de ${expected.field}`).toBe(expected.missing);
      expect(actual?.status, `status de ${expected.field}`).toBe(expected.status);
      expect(actual?.missingPct, `percentual de ${expected.field}`).toBeCloseTo(
        expected.missingPct,
        9,
      );
    }
  });
});

/** Compara linha a linha, casando por nome de entidade em vez de posição. */
function expectTableParity(name: string, actualRows: EntityRow[], expectedRows: GoldenEntityRow[]) {
  const actualByEntity = new Map(actualRows.map((row) => [row.entity, row]));

  expect(actualRows.length, `${name}: número de linhas`).toBe(expectedRows.length);

  for (const expected of expectedRows) {
    const actual = actualByEntity.get(expected.entity);
    const where = `${name} / ${expected.entity}`;

    expect(actual, `${where} ausente na tabela TypeScript`).toBeDefined();
    if (!actual) continue;

    expect(actual.citations, `${where} citações`).toBe(expected.citations);
    expect(actual.h, `${where} índice h`).toBe(expected.h);
    expect(actual.g, `${where} índice g`).toBe(expected.g);
    expect(actual.i10, `${where} índice i10`).toBe(expected.i10);
    expect(actual.m, `${where} índice m`).toBe(expected.m);
    expect(actual.meanCitations, `${where} média de citações`).toBe(expected.meanCitations);
    expect(actual.medianCitations, `${where} mediana de citações`).toBe(expected.medianCitations);
    expect(actual.stdCitations, `${where} desvio padrão`).toBe(expected.stdCitations);
  }
}

describe('tabelas analíticas', () => {
  it('autores: paridade total', () => {
    expectTableParity('autores', authorsTable(dataset, BASE_YEAR), golden.tabelas['autores'] ?? []);
  });

  it('países: paridade total', () => {
    expectTableParity('paises', countriesTable(dataset, BASE_YEAR), golden.tabelas['paises'] ?? []);
  });

  it('keywords: paridade total', () => {
    expectTableParity(
      'keywords',
      keywordsTable(dataset, BASE_YEAR),
      golden.tabelas['keywords'] ?? [],
    );
  });

  /**
   * Venues são o caso onde divergimos de propósito.
   *
   * `gerar_tabela_venues` agrupa pelo valor BRUTO da coluna e só converte para maiúsculas
   * no rótulo final (utils.py:1010, 1036). Quando a mesma revista aparece com
   * capitalização diferente entre bases — o que é a regra, não a exceção, ao misturar
   * WoS e Scopus — o Python emite várias linhas com o mesmo nome exibido, cada uma com um
   * pedaço dos totais. São 130 linhas duplicadas nos dados de exemplo.
   *
   * Aqui a normalização acontece ANTES do agrupamento, então cada revista aparece uma vez
   * com o total correto. O teste prova que isso é reagrupamento e não perda de dados: as
   * citações de cada linha nossa têm de bater com a SOMA das linhas fragmentadas do Python.
   *
   * Os índices h/g/i10/m não são aditivos e por isso não podem ser verificados assim —
   * a corretude deles já está coberta pelas outras três tabelas, que usam o mesmo motor.
   */
  it('venues: reagrupa os rótulos duplicados do Python sem perder dados', () => {
    const actualRows = venuesTable(dataset, BASE_YEAR);
    const expectedRows = golden.tabelas['venues'] ?? [];

    const expectedTotals = new Map<string, number>();
    for (const row of expectedRows) {
      expectedTotals.set(row.entity, (expectedTotals.get(row.entity) ?? 0) + row.citations);
    }

    // O Python de fato fragmenta: menos rótulos distintos do que linhas emitidas.
    expect(expectedTotals.size).toBeLessThan(expectedRows.length);

    expect(actualRows.length, 'uma linha por revista distinta').toBe(expectedTotals.size);
    expect(new Set(actualRows.map((row) => row.entity))).toEqual(new Set(expectedTotals.keys()));

    for (const row of actualRows) {
      expect(row.citations, `venues / ${row.entity}: soma das citações`).toBe(
        expectedTotals.get(row.entity),
      );
    }
  });
});
