import { describe, expect, it } from 'vitest';

import { encodeProjectEnvelope, parseProjectEnvelope, type ProjectRecord } from '@/lib/project';

function makeRecord(overrides: Partial<ProjectRecord> = {}): ProjectRecord {
  return {
    id: 'test-id',
    schemaVersion: 1,
    name: 'Minha base',
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-02T00:00:00.000Z',
    sourceFiles: [{ name: 'scopus.ris', database: 'Scopus' }],
    dedupStrategy: 'doi',
    dedupThreshold: null,
    original: [{ TITLE: 'A' }, { TITLE: 'B' }] as ProjectRecord['original'],
    active: [{ TITLE: 'A' }] as ProjectRecord['active'],
    duplicates: [],
    clustering: null,
    ...overrides,
  };
}

describe('encodeProjectEnvelope', () => {
  it('wraps the project in a versioned, identifiable envelope', () => {
    const record = makeRecord();
    const envelope = encodeProjectEnvelope(record);

    expect(envelope.kind).toBe('simetrics-project');
    expect(envelope.schemaVersion).toBe(1);
    expect(envelope.project).toEqual(record);
    expect(() => new Date(envelope.exportedAt).toISOString()).not.toThrow();
  });
});

describe('parseProjectEnvelope', () => {
  it('round-trips a record through encode/JSON/parse unchanged', () => {
    const record = makeRecord();
    const roundTripped = JSON.parse(JSON.stringify(encodeProjectEnvelope(record))) as unknown;

    expect(parseProjectEnvelope(roundTripped)).toEqual(record);
  });

  it('fills sensible defaults when optional fields are missing', () => {
    const envelope = encodeProjectEnvelope(makeRecord());
    const { sourceFiles: _sourceFiles, duplicates: _duplicates, clustering: _clustering, ...rest } =
      envelope.project;
    const parsed = parseProjectEnvelope({ ...envelope, project: rest });

    expect(parsed.sourceFiles).toEqual([]);
    expect(parsed.duplicates).toEqual([]);
    expect(parsed.clustering).toBeNull();
  });

  it.each([
    ['not an object', 'just a string'],
    ['null', null],
    ['missing kind', { schemaVersion: 1, project: makeRecord() }],
    ['wrong kind', { kind: 'something-else', schemaVersion: 1, project: makeRecord() }],
    [
      'non-numeric schemaVersion',
      { kind: 'simetrics-project', schemaVersion: '1', project: makeRecord() },
    ],
    [
      'missing project.original',
      {
        kind: 'simetrics-project',
        schemaVersion: 1,
        project: { ...makeRecord(), original: undefined },
      },
    ],
    [
      'project.active not an array',
      {
        kind: 'simetrics-project',
        schemaVersion: 1,
        project: { ...makeRecord(), active: 'oops' },
      },
    ],
    [
      'unsupported schemaVersion',
      { kind: 'simetrics-project', schemaVersion: 999, project: makeRecord() },
    ],
  ])('rejects %s', (_label, input) => {
    expect(() => parseProjectEnvelope(input)).toThrow();
  });
});
