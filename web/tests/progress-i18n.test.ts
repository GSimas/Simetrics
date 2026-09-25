import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { localizeProgress, PHASES_EN } from '@/lib/i18n/progress';

const SRC = join(__dirname, '..', 'src');

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(name) && !path.endsWith(join('i18n', 'progress.ts')) ? [path] : [];
  });
}

/** Toda fase escrita como literal: `phase: '…'` e `onProgress?.(ratio, '…')`. */
function literalPhases(): Set<string> {
  const phases = new Set<string>();
  for (const file of sourceFiles(SRC)) {
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      for (const match of line.matchAll(/phase:\s*(['"])(.*?)\1/g)) phases.add(match[2]!);
      // Chamada posicional (ratio, fase): pega todos os literais da linha, inclusive ternários.
      if (/onProgress\??\.?\(\s*[^\s{)]/.test(line)) {
        for (const match of line.matchAll(/(['"])(.*?)\1/g)) phases.add(match[2]!);
      }
    }
  }
  return phases;
}

describe('localizeProgress', () => {
  const phases = literalPhases();

  it('finds the phases written across the code', () => {
    expect(phases.size).toBeGreaterThan(20);
  });

  it.each([...phases])('translates phase "%s"', (phase) => {
    expect(PHASES_EN).toHaveProperty([phase]);
  });

  it('translates template phases and numeric details', () => {
    expect(localizeProgress({ phase: 'Testando 4 agrupamentos', ratio: 0 }, 'en').phase).toBe('Testing 4 clusters');
    const en = (detail: string) => localizeProgress({ phase: 'Lendo arquivos', ratio: 0, detail }, 'en').detail;
    expect(en('12 documentos')).toBe('12 documents');
    expect(en('3 de 10 documentos')).toBe('3 of 10 documents');
    expect(en('3 de 10')).toBe('3 of 10');
    expect(en('scopus.ris')).toBe('scopus.ris');
  });

  it('leaves Portuguese untouched', () => {
    const progress = { phase: 'Padronizando estrutura', ratio: 0.5, detail: '3 de 10 documentos' };
    expect(localizeProgress(progress, 'pt')).toBe(progress);
    expect(localizeProgress(progress, 'en')).toEqual({ phase: 'Standardizing structure', ratio: 0.5, detail: '3 of 10 documents' });
  });
});
