import { describe, expect, it } from 'vitest';

import { flattenRisRecord, parseRis } from '@/core/parsers/ris';

/**
 * Casos de borda do parser RIS.
 *
 * O teste de pipeline já prova a paridade agregada sobre os 971 registros reais; aqui
 * ficam registradas as regras específicas do rispy que são fáceis de quebrar sem que
 * nenhum total mude visivelmente.
 */

const ris = (...lines: string[]): string => lines.join('\n');

describe('parseRis', () => {
  it('lê um registro simples entre TY e ER', () => {
    const [record] = parseRis(
      ris('TY  - JOUR', 'TI  - Um título qualquer', 'PY  - 2020', 'ER  - '),
    );

    expect(record).toEqual({
      type_of_reference: 'JOUR',
      title: 'Um título qualquer',
      year: '2020',
    });
  });

  it('ignora tudo que estiver fora de um registro', () => {
    const records = parseRis(
      ris('lixo antes', 'FN Clarivate Analytics', 'TY  - JOUR', 'TI  - Válido', 'ER  - ', 'sobra'),
    );

    expect(records).toHaveLength(1);
    expect(records[0]?.['title']).toBe('Válido');
  });

  it('acumula tags de lista e mantém só o primeiro valor das demais', () => {
    const [record] = parseRis(
      ris(
        'TY  - JOUR',
        'AU  - Silva, A',
        'AU  - Santos, B',
        'KW  - memética',
        'KW  - evolução',
        // TI não é tag de lista: o rispy usa setdefault, então o primeiro valor vence.
        'TI  - Primeiro',
        'TI  - Segundo',
        'ER  - ',
      ),
    );

    expect(record?.['authors']).toEqual(['Silva, A', 'Santos, B']);
    expect(record?.['keywords']).toEqual(['memética', 'evolução']);
    expect(record?.['title']).toBe('Primeiro');
  });

  it('junta linhas de continuação de tags simples com espaço', () => {
    const [record] = parseRis(
      ris('TY  - JOUR', 'AB  - Primeira parte', 'segunda parte', 'terceira parte', 'ER  - '),
    );

    expect(record?.['abstract']).toBe('Primeira parte segunda parte terceira parte');
  });

  it('separa a tag UR pelo delimitador declarado', () => {
    const [record] = parseRis(
      ris('TY  - JOUR', 'UR  - http://a.example; http://b.example', 'ER  - '),
    );

    expect(record?.['urls']).toEqual(['http://a.example', 'http://b.example']);
  });

  it('guarda tags desconhecidas em unknown_tag sob o rótulo original', () => {
    const [record] = parseRis(ris('TY  - JOUR', 'TC  - 42', 'Z9  - 45', 'ER  - '));

    expect(record?.['unknown_tag']).toEqual({ TC: ['42'], Z9: ['45'] });
  });

  it('NÃO divide em \\r isolado, preservando o blob de afiliações', () => {
    // O app alimenta o rispy via `io.StringIO`, que não traduz CR, e o rispy divide só em
    // \n. Blocos separados por CR permanecem numa única linha — é isso que mantém todas
    // as afiliações no mesmo campo em vez de descartar todas menos a primeira.
    const [record] = parseRis(
      ris(
        'TY  - JOUR',
        'AD  - Reed Coll, Portland, USA\rAD  - European Sch, Milan, Italy\rC3  - Reed College',
        'ER  - ',
      ),
    );

    expect(record?.['author_address']).toContain('Italy');
    expect(record?.['author_address']).toContain('USA');
    // C3 fica embutido no texto, e não vira uma coluna própria.
    expect(record?.['custom3']).toBeUndefined();
  });

  it('tolera CRLF, deixando o \\r ser removido pelo trim', () => {
    const [record] = parseRis('TY  - JOUR\r\nTI  - Com CRLF\r\nER  - \r\n');
    expect(record?.['title']).toBe('Com CRLF');
  });

  it('lê múltiplos registros em sequência', () => {
    const records = parseRis(
      ris('TY  - JOUR', 'TI  - Um', 'ER  - ', 'TY  - CONF', 'TI  - Dois', 'ER  - '),
    );

    expect(records.map((record) => record['title'])).toEqual(['Um', 'Dois']);
  });
});

describe('flattenRisRecord', () => {
  it('converte chaves para MAIÚSCULAS com espaço no lugar do underscore', () => {
    const flat = flattenRisRecord({ author_address: 'Rua X', type_of_reference: 'JOUR' });
    expect(flat).toEqual({ 'AUTHOR ADDRESS': 'Rua X', 'TYPE OF REFERENCE': 'JOUR' });
  });

  it('junta listas com "; "', () => {
    const flat = flattenRisRecord({ authors: ['Silva, A', 'Santos, B'] });
    expect(flat['AUTHORS']).toBe('Silva, A; Santos, B');
  });

  it('dissolve unknown_tag de volta no nível superior', () => {
    // É daqui que sai a coluna TC, de onde o pipeline extrai as citações do WoS.
    const flat = flattenRisRecord({ unknown_tag: { TC: ['42'], FU: ['a', 'b'] } });
    expect(flat['TC']).toBe('42');
    expect(flat['FU']).toBe('a; b');
  });
});
