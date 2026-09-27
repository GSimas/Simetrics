import { DEFAULT_MODELS, useAiConfig } from '@/state/ai-config.store';

/**
 * Login no OpenRouter por OAuth PKCE: o usuário autoriza no site do OpenRouter e volta com
 * um `code`, trocado aqui por uma chave de API — sem precisar copiar e colar a chave.
 * https://openrouter.ai/docs/guides/overview/auth/oauth
 */

const VERIFIER_KEY = 'simetrics_openrouter_pkce_verifier';
const HASH_KEY = 'simetrics_openrouter_pkce_hash';

function base64Url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

export async function codeChallengeS256(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  return base64Url(new Uint8Array(digest));
}

export async function startOpenRouterLogin(): Promise<void> {
  const verifier = base64Url(crypto.getRandomValues(new Uint8Array(32)));
  sessionStorage.setItem(VERIFIER_KEY, verifier);
  // O OpenRouter anexa `?code=` ao callback; sem o hash na URL de volta, guardamos a rota.
  sessionStorage.setItem(HASH_KEY, window.location.hash);
  const params = new URLSearchParams({
    callback_url: window.location.origin + window.location.pathname,
    code_challenge: await codeChallengeS256(verifier),
    code_challenge_method: 'S256',
    key_label: 'Simetrics',
  });
  window.location.assign(`https://openrouter.ai/auth?${params}`);
}

/** Na volta do OpenRouter, troca o `code` pela chave e configura o provedor. */
export async function completeOpenRouterLogin(): Promise<void> {
  const url = new URL(window.location.href);
  const code = url.searchParams.get('code');
  const verifier = sessionStorage.getItem(VERIFIER_KEY);
  if (!code || !verifier) return;

  const hash = sessionStorage.getItem(HASH_KEY) ?? '';
  sessionStorage.removeItem(VERIFIER_KEY);
  sessionStorage.removeItem(HASH_KEY);
  url.searchParams.delete('code');
  // Síncrono, antes do primeiro render: a rota de hash já nasce restaurada.
  window.history.replaceState(null, '', `${url.pathname}${url.search}${hash}`);

  try {
    const res = await fetch('https://openrouter.ai/api/v1/auth/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: 'S256' }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { key } = (await res.json()) as { key?: string };
    if (!key) throw new Error('resposta sem chave');
    useAiConfig.getState().setConfig({ provider: 'openrouter', apiKey: key, model: DEFAULT_MODELS.openrouter });
  } catch (err) {
    console.error('Falha no login com OpenRouter:', err);
  }
}
