import { createClient } from '@supabase/supabase-js';

/**
 * Conta Scientata (Supabase): login opcional e projetos na nuvem. A chave publicável é
 * pública por desenho (o acesso aos dados é barrado pelas políticas RLS do banco), por isso
 * vai no código como padrão; as variáveis de ambiente apontam o app para outro projeto
 * (ex.: o de produção) sem mexer aqui.
 */
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || 'https://xfjbwfjicqqxooxgvbjj.supabase.co';
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || 'sb_publishable_wppNZ1-GOfW8fzTKNL5vbA_oULtbYzh';

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: {
    flowType: 'pkce',
    // O login do OpenRouter também volta com `?code=`: a troca do código do Supabase só
    // acontece na volta marcada com `?auth=supabase` (auth.store.ts).
    detectSessionInUrl: false,
    persistSession: true,
    autoRefreshToken: true,
    storageKey: 'simetrics-auth',
  },
});

/** Bucket privado: {user_id}/projects/{id}.json.gz e {user_id}/pdfs/{hash}.pdf|.json */
export const CLOUD_BUCKET = 'simetrics';
