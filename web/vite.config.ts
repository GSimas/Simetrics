import { defineConfig } from 'vitest/config';
import { loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { visualizer } from 'rollup-plugin-visualizer';
import { Agent } from 'node:https';
import tls from 'node:tls';
import { fileURLToPath, URL } from 'node:url';

/**
 * Agente HTTPS do proxy do Jev que confia também nos certificados do sistema operacional,
 * como o navegador. Em redes corporativas com inspeção de SSL o Node, que por padrão só
 * conhece a lista embutida, recusa a conexão ("unable to get local issuer certificate").
 */
function systemTrustAgent(): Agent | undefined {
  if (typeof tls.getCACertificates !== 'function') return undefined;
  try {
    return new Agent({ ca: [...tls.getCACertificates('default'), ...tls.getCACertificates('system')], keepAlive: true });
  } catch {
    return undefined;
  }
}

export default defineConfig(({ mode }) => {
  // Só o proxy do Jev lê o .env, e só a chave dele — nada disto chega ao bundle do cliente.
  const typesafeKey = loadEnv(mode, process.cwd(), '')['TYPESAFE_API_KEY']?.trim();
  const jevAgent = systemTrustAgent();

  return {
    plugins: [
      react(),
      tailwindcss(),
      // `ANALYZE=1 npm run build` gera dist/stats.html (treemap) e dist/stats.json.
      ...(process.env['ANALYZE']
        ? [
            visualizer({ filename: 'dist/stats.html', gzipSize: true, brotliSize: true, template: 'treemap' }),
            visualizer({ filename: 'dist/stats.json', template: 'raw-data' }),
          ]
        : []),
    ],
    define: {
      // Dependências escritas para o Node (PapaParse, html2canvas, o SDK de IA, a lib dos
      // workers de grafo) referenciam `global`, um objeto que não existe no navegador. Sem
      // este mapeamento, qualquer uma delas que não se proteja com `typeof` quebra com
      // "global is not defined" — e apenas em execução: o build conclui sem reclamar.
      global: 'globalThis',
    },
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      // Respeita a porta atribuída pelo ambiente (harness/CI); cai em 5173 no uso local.
      port: process.env['PORT'] ? Number(process.env['PORT']) : 5173,
      // Em produção o Netlify serve as funções no mesmo domínio, então o cliente chama
      // /api/... direto. No desenvolvimento com `vite dev` puro não há função alguma; este
      // proxy encaminha para `netlify functions:serve`, quando ele estiver rodando.
      //
      //     npx netlify-cli functions:serve --port 9999
      //
      // Sem esse servidor, as chamadas falham com erro de conexão — e a interface trata
      // isso como qualquer outra indisponibilidade. Rodar `netlify dev` no lugar de
      // `vite dev` dispensa o proxy, porque ele já sobe os dois lados juntos.
      proxy: {
        // O Jev recusa requisições com `Origin` (403), então não pode ser chamado do
        // navegador. Em produção quem faz a ponte é a função netlify/functions/jev-systemone;
        // aqui é este proxy, que remove o `Origin` e usa a TYPESAFE_API_KEY do .env quando o
        // usuário não informou chave própria. Precisa vir antes do '/api' genérico.
        '/api/jev': {
          target: 'https://api.typesafe.ai',
          changeOrigin: true,
          ...(jevAgent ? { agent: jevAgent } : {}),
          rewrite: (path) => path.replace(/^\/api\/jev/, '/v1'),
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              proxyReq.removeHeader('origin');
              proxyReq.removeHeader('referer');
              if (!req.headers.authorization && typesafeKey) {
                proxyReq.setHeader('authorization', `Bearer ${typesafeKey}`);
              }
            });
          },
        },
        '/api': {
          target: process.env['NETLIFY_FUNCTIONS_URL'] ?? 'http://localhost:9999',
          changeOrigin: true,
        },
      },
    },
    worker: {
      // Os workers usam import/export; o formato ES evita o bundle IIFE legado.
      format: 'es',
    },
    build: {
      target: 'es2022',
      // As libs pesadas vão em chunks próprios: a UI fica num chunk pequeno e elas carregam
      // sob demanda, por aba. Os gráficos em si são SVG desenhado pelo Simetrics.
      // Vite 8 usa Rolldown: o `manualChunks` em objeto do Rollup virou `codeSplitting.groups`.
      rollupOptions: {
        output: {
          codeSplitting: {
            groups: [
              { name: 'echarts', test: /node_modules[\\/]echarts[\\/]/ },
              { name: 'graph', test: /node_modules[\\/](graphology|sigma)/ },
            ],
          },
        },
      },
      // O maior chunk, o principal, fica em ~1,8 MB. O limite fica logo acima dele para o
      // aviso continuar tendo função: sinalizar crescimento inesperado, e não repetir um
      // fato já conhecido.
      chunkSizeWarningLimit: 2000,
    },
    test: {
      globals: true,
      environment: 'node',
      include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    },
  };
});
