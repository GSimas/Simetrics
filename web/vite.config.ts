import { defineConfig } from 'vitest/config';
import { loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { visualizer } from 'rollup-plugin-visualizer';
import { fileURLToPath, URL } from 'node:url';

import { simetricsApi } from './server/vite-plugin.ts';

export default defineConfig(({ mode }) => {
  // Chaves de servidor (DEEPSEEK_API_KEY, TYPESAFE_API_KEY) lidas do .env só pelo plugin da
  // API — sem prefixo VITE_, nada disto chega ao bundle do cliente.
  const serverEnv = { ...loadEnv(mode, process.cwd(), ''), ...process.env };

  return {
    plugins: [
      react(),
      tailwindcss(),
      simetricsApi(serverEnv),
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
      // Os endpoints /api/* são atendidos pelo plugin simetricsApi (server/vite-plugin.ts),
      // com os mesmos handlers da Netlify Function — não há mais proxy nem servidor extra.
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
