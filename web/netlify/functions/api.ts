import type { Config, Context } from '@netlify/functions';

import { BlobsQuotaStore } from '../../server/blobs-store.ts';
import { readServerEnv } from '../../server/env.ts';
import { API_PATHS, handleApi } from '../../server/handlers.ts';

/**
 * Todos os endpoints do Simetrics numa função só — a lógica mora em server/handlers.ts,
 * compartilhada com o plugin de desenvolvimento do Vite.
 *
 * As chaves (DEEPSEEK_API_KEY, TYPESAFE_API_KEY) vêm das variáveis de ambiente do site na
 * Netlify e nunca chegam ao navegador.
 */
export default async (request: Request, context: Context): Promise<Response> => {
  const response = await handleApi(request, {
    env: readServerEnv(process.env),
    store: new BlobsQuotaStore(),
    ip: context.ip,
  });
  return response ?? Response.json({ error: { message: 'Rota inexistente.' } }, { status: 404 });
};

export const config: Config = {
  path: API_PATHS,
};
