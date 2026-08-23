import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    // O `plotly.js` de código-fonte (diferente do `dist-min` pré-empacotado) arrasta
    // dependências que referenciam `global`, um objeto do Node que não existe no
    // navegador. Sem este mapeamento a aplicação quebra em tempo de execução com
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
    // Plotly e ECharts sozinhos passam de 1MB. Separá-los mantém a UI em um chunk pequeno
    // e permite carregar as libs de gráfico sob demanda, por aba.
    // Vite 8 usa Rolldown: o `manualChunks` em objeto do Rollup virou `codeSplitting.groups`.
    rollupOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: 'plotly', test: /node_modules[\\/]plotly\.js[\\/]/ },
            { name: 'echarts', test: /node_modules[\\/]echarts[\\/]/ },
            { name: 'graph', test: /node_modules[\\/](graphology|sigma)/ },
          ],
        },
      },
    },
    // O chunk do Plotly com geo e 3D fica em ~1,9 MB e é irredutível sem abrir mão de
    // visualizações. O limite fica logo acima dele para o aviso continuar tendo função:
    // sinalizar crescimento inesperado, e não repetir um fato já conhecido.
    chunkSizeWarningLimit: 2000,
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
  },
});
