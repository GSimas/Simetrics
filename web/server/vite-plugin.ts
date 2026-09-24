import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import tls from 'node:tls';
import type { Plugin } from 'vite';

import { readServerEnv } from './env.ts';
import { API_PATHS, handleApi } from './handlers.ts';
import { MemoryQuotaStore, type QuotaRecord } from './quota.ts';

/**
 * Serve os endpoints /api/* no `vite dev`, com os mesmos handlers da Netlify Function.
 * As chaves vêm do `web/.env` (DEEPSEEK_API_KEY, TYPESAFE_API_KEY) e as cotas ficam num
 * JSON em node_modules/.cache — reiniciar o servidor não zera o contador, apagar o
 * arquivo sim.
 */

/**
 * Faz o Node confiar também nos certificados do sistema operacional, como o navegador.
 * Em redes com inspeção de SSL o Node, que só conhece a lista embutida, recusa as
 * conexões com "unable to get local issuer certificate".
 */
function trustSystemCertificates(): void {
  const api = tls as typeof tls & {
    getCACertificates?: (type: 'default' | 'system') => string[];
    setDefaultCACertificates?: (certs: string[]) => void;
  };
  if (!api.getCACertificates || !api.setDefaultCACertificates) return;
  try {
    api.setDefaultCACertificates([...new Set([...api.getCACertificates('default'), ...api.getCACertificates('system')])]);
  } catch {
    // Sem acesso ao repositório do sistema: segue só com a lista embutida.
  }
}

function fileStore(root: string): MemoryQuotaStore {
  const path = resolve(root, 'node_modules/.cache/simetrics/dev-quota.json');
  return new MemoryQuotaStore({
    load: () => {
      try {
        return JSON.parse(readFileSync(path, 'utf8')) as Record<string, QuotaRecord>;
      } catch {
        return null;
      }
    },
    save: (data) => {
      try {
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, JSON.stringify(data));
      } catch {
        // Cota só em memória nesta sessão.
      }
    },
  });
}

export function simetricsApi(envSource: Record<string, string | undefined>): Plugin {
  const env = readServerEnv(envSource);
  return {
    name: 'simetrics-api',
    configureServer(server) {
      trustSystemCertificates();
      const store = fileStore(server.config.root);
      const configured = [env.deepseekKey && 'DeepSeek', env.typesafeKey && 'Jev'].filter(Boolean);
      server.config.logger.info(
        `  simetrics-api: ${configured.length > 0 ? `chaves do servidor: ${configured.join(', ')}` : 'nenhuma chave de servidor no .env'}`,
      );

      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
        if (!API_PATHS.includes(url.pathname as `/${string}`)) return next();

        try {
          const chunks: Buffer[] = [];
          for await (const chunk of req) chunks.push(chunk as Buffer);
          const headers = new Headers();
          for (const [name, value] of Object.entries(req.headers)) {
            if (typeof value === 'string') headers.set(name, value);
            else if (Array.isArray(value)) headers.set(name, value.join(', '));
          }
          const method = req.method ?? 'GET';
          const request = new Request(url, {
            method,
            headers,
            ...(method === 'GET' || method === 'HEAD' ? {} : { body: Buffer.concat(chunks) }),
          });

          const response = await handleApi(request, { env, store, ip: req.socket.remoteAddress ?? '' });
          if (!response) return next();

          res.statusCode = response.status;
          response.headers.forEach((value, name) => res.setHeader(name, value));
          if (response.body) {
            const reader = response.body.getReader();
            for (;;) {
              const { done, value } = await reader.read();
              if (done) break;
              res.write(value);
            }
          }
          res.end();
        } catch (cause) {
          server.config.logger.error(`simetrics-api: ${cause instanceof Error ? cause.message : String(cause)}`);
          if (!res.headersSent) res.statusCode = 500;
          res.end();
        }
      });
    },
  };
}
