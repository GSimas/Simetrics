import type { Config, Context } from '@netlify/functions';

/**
 * Proxy para a API do Jev (TypeSafe).
 *
 * Existe porque a API do Jev recusa requisições com cabeçalho `Origin` (403): o navegador
 * não consegue chamá-la diretamente. Esta função repassa o corpo sem o `Origin` e devolve
 * a resposta como veio, com o mesmo status — o cliente trata 429/529 com novas tentativas.
 *
 * A chave vem do cabeçalho `Authorization` do cliente (BYOK, digitada pelo usuário e
 * guardada só no navegador dele) ou, na ausência dela, da `TYPESAFE_API_KEY` do servidor.
 * A chave nunca é registrada em log.
 */

const UPSTREAM = 'https://api.typesafe.ai/v1/systemone';
/** Uma requisição leva um documento; 200 kB é folga de sobra e barra abuso do proxy. */
const MAX_PAYLOAD_BYTES = 200_000;

export default async (request: Request, _context: Context): Promise<Response> => {
  if (request.method !== 'POST') {
    return Response.json({ error: 'Método não permitido. Use POST.' }, { status: 405 });
  }

  const authorization =
    request.headers.get('authorization')?.trim() ||
    (process.env['TYPESAFE_API_KEY']?.trim() ? `Bearer ${process.env['TYPESAFE_API_KEY'].trim()}` : '');

  if (!authorization) {
    return Response.json(
      { error: 'Chave do Jev ausente. Informe-a nas configurações da classificação híbrida ou defina TYPESAFE_API_KEY.' },
      { status: 401 },
    );
  }

  const body = await request.text();
  if (body.length > MAX_PAYLOAD_BYTES) {
    return Response.json({ error: 'Payload grande demais para uma avaliação.' }, { status: 413 });
  }

  try {
    const upstream = await fetch(UPSTREAM, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: authorization },
      body,
    });
    const headers = new Headers({ 'Content-Type': upstream.headers.get('content-type') ?? 'application/json' });
    const retryAfter = upstream.headers.get('retry-after');
    if (retryAfter) headers.set('Retry-After', retryAfter);
    return new Response(await upstream.text(), { status: upstream.status, headers });
  } catch (cause) {
    console.error('Falha ao contatar o Jev:', cause instanceof Error ? cause.message : cause);
    return Response.json({ error: 'Não foi possível contatar o Jev.' }, { status: 502 });
  }
};

export const config: Config = {
  path: '/api/jev/systemone',
};
