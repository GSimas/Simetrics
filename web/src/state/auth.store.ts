import type { User } from '@supabase/supabase-js';
import { create } from 'zustand';

import { releaseAccount, setCloudUser, syncAll } from '@/lib/cloud-sync';
import { supabase } from '@/lib/supabase';
import { useLocale } from './locale.store';
import { useProjectStore } from './project.store';

/**
 * Conta Scientata, opcional: sem login o Simetrics funciona como sempre (tudo no
 * navegador); com login, os projetos e PDFs passam a ser espelhados na nuvem.
 */

export type OAuthProvider = 'google' | 'azure';

interface AuthState {
  user: User | null;
  ready: boolean;
  /** Voltou do link de "esqueci a senha": o app pede a senha nova. */
  recovering: boolean;
  /** Erro devolvido pelo provedor na volta do login (ex.: acesso negado). */
  redirectError: string | null;
}

export const useAuth = create<AuthState>(() => ({
  user: null,
  ready: false,
  recovering: false,
  redirectError: null,
}));

const RETURN_FLAG = 'auth';
const HASH_KEY = 'simetrics-auth-hash';

/** Volta do Supabase (OAuth, confirmação de e-mail, troca de senha) com a rota preservada. */
function returnUrl(kind?: 'recovery'): string {
  try {
    sessionStorage.setItem(HASH_KEY, window.location.hash);
  } catch {
    // Sem sessionStorage a volta cai na tela inicial, o que não quebra nada.
  }
  const params = new URLSearchParams({ [RETURN_FLAG]: kind ?? 'supabase' });
  return `${window.location.origin}${window.location.pathname}?${params}`;
}

export async function signInWithPassword(email: string, password: string): Promise<void> {
  const { error } = await supabase.auth.signInWithPassword({ email, password });
  if (error) throw error;
}

/** Devolve `true` quando a conta precisa confirmar o e-mail antes de entrar. */
export async function signUp(email: string, password: string, fullName: string): Promise<boolean> {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: { emailRedirectTo: returnUrl(), ...(fullName ? { data: { full_name: fullName } } : {}) },
  });
  if (error) throw error;
  // E-mail já cadastrado (inclusive via Google/Microsoft): o Supabase responde "ok" sem
  // enviar nada e devolve um usuário sem identidades. Avisa em vez de prometer um e-mail.
  if (data.user && data.user.identities?.length === 0) {
    throw new Error(useLocale.getState().t('auth_email_exists'));
  }
  return !data.session;
}

export async function signInWithOAuth(provider: OAuthProvider): Promise<void> {
  const { error } = await supabase.auth.signInWithOAuth({
    provider,
    // Microsoft (Azure) só devolve o e-mail com o escopo explícito.
    options: { redirectTo: returnUrl(), ...(provider === 'azure' ? { scopes: 'email' } : {}) },
  });
  if (error) throw error;
}

export async function sendPasswordReset(email: string): Promise<void> {
  const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo: returnUrl('recovery') });
  if (error) throw error;
}

export async function updatePassword(password: string): Promise<void> {
  const { error } = await supabase.auth.updateUser({ password });
  if (error) throw error;
  useAuth.setState({ recovering: false });
}

export async function signOut(): Promise<void> {
  const removed = await releaseAccount().catch((cause) => {
    console.error('Simetrics: falha ao guardar os projetos antes de sair', cause);
    return [] as string[];
  });
  const projects = useProjectStore.getState();
  if (projects.activeProjectId && removed.includes(projects.activeProjectId)) projects.startBlank();
  await supabase.auth.signOut();
  await projects.refreshList();
}

/** Troca o `?code=` da volta pelo login. Só age na volta marcada (`?auth=`). */
async function completeRedirect(): Promise<void> {
  const url = new URL(window.location.href);
  const kind = url.searchParams.get(RETURN_FLAG);
  if (!kind) return;

  const code = url.searchParams.get('code');
  const providerError = url.searchParams.get('error_description') ?? url.searchParams.get('error');
  let hash = '';
  try {
    hash = sessionStorage.getItem(HASH_KEY) ?? '';
    sessionStorage.removeItem(HASH_KEY);
  } catch {
    // ignora
  }
  for (const key of [RETURN_FLAG, 'code', 'error', 'error_code', 'error_description']) url.searchParams.delete(key);
  window.history.replaceState(null, '', `${url.pathname}${url.search}${hash || url.hash}`);

  if (providerError) {
    useAuth.setState({ redirectError: providerError });
    return;
  }
  if (!code) return;
  const { error } = await supabase.auth.exchangeCodeForSession(code);
  if (error) useAuth.setState({ redirectError: error.message });
  else if (kind === 'recovery') useAuth.setState({ recovering: true });
}

let syncedFor: string | null = null;

async function syncFor(user: User | null): Promise<void> {
  setCloudUser(user?.id ?? null);
  if (!user) {
    syncedFor = null;
    return;
  }
  // `SIGNED_IN` volta a disparar ao reabrir a aba: só reconcilia uma vez por conta.
  if (syncedFor === user.id) return;
  syncedFor = user.id;
  if (await syncAll()) await useProjectStore.getState().refreshList();
}

/** Chamado uma vez no início do app (main.tsx). */
export async function initAuth(): Promise<void> {
  await completeRedirect().catch((cause) => console.error('Simetrics: falha na volta do login', cause));

  supabase.auth.onAuthStateChange((event, session) => {
    const user = session?.user ?? null;
    useAuth.setState({ user, ready: true, ...(event === 'PASSWORD_RECOVERY' ? { recovering: true } : {}) });
    // Fora do callback: chamar o Supabase dentro dele pode travar a sessão.
    setTimeout(() => void syncFor(user), 0);
  });
}
