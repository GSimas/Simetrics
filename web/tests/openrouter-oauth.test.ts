import { describe, expect, it } from 'vitest';

import { codeChallengeS256 } from '@/lib/openrouter-oauth';

describe('codeChallengeS256', () => {
  it('reproduz o vetor de teste da RFC 7636 (apêndice B)', async () => {
    expect(await codeChallengeS256('dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk')).toBe(
      'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM',
    );
  });
});
