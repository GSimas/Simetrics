/**
 * Relatório do bundle de produção — rode depois de `npm run build`:
 *
 *     node scripts/bundle-report.mjs            # tabela no terminal
 *     node scripts/bundle-report.mjs --json     # JSON (para comparar antes/depois)
 *
 * "JS inicial" é o que o navegador baixa antes de a página funcionar: o script de entrada
 * do `dist/index.html` mais os chunks que ele pré-carrega (`modulepreload`) e os que ele
 * importa estaticamente. Chunks só alcançáveis por `import()` dinâmico ficam de fora —
 * são os carregados sob demanda.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { brotliCompressSync, gzipSync, constants } from 'node:zlib';

const DIST = new URL('../dist/', import.meta.url);
const distPath = decodeURIComponent(DIST.pathname).replace(/^\/([A-Za-z]:)/, '$1');
const assetsPath = join(distPath, 'assets');

const sizes = (buffer) => ({
  raw: buffer.length,
  gzip: gzipSync(buffer, { level: 9 }).length,
  brotli: brotliCompressSync(buffer, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length,
});

const files = readdirSync(assetsPath).filter((name) => statSync(join(assetsPath, name)).isFile());
const js = files.filter((name) => name.endsWith('.js'));
const css = files.filter((name) => name.endsWith('.css'));

const info = new Map(
  [...js, ...css].map((name) => {
    const buffer = readFileSync(join(assetsPath, name));
    return [name, { name, ...sizes(buffer), source: buffer.toString('utf8') }];
  }),
);

// Grafo de imports estáticos entre chunks: `import ... from "./x.js"` e `import "./x.js"`.
const staticImports = (source) => {
  const found = new Set();
  const pattern = /(?:^|[;\n}])\s*import\s*(?:[\w*{}\s,$]+from\s*)?["']\.\/([^"']+\.js)["']/g;
  for (const match of source.matchAll(pattern)) found.add(match[1]);
  return [...found];
};

const html = readFileSync(join(distPath, 'index.html'), 'utf8');
const entry = [...html.matchAll(/<script[^>]+src="\/assets\/([^"]+\.js)"/g)].map((m) => m[1]);
const preloaded = [...html.matchAll(/rel="modulepreload"[^>]*href="\/assets\/([^"]+\.js)"/g)].map((m) => m[1]);
const cssLinks = [...html.matchAll(/rel="stylesheet"[^>]*href="\/assets\/([^"]+\.css)"/g)].map((m) => m[1]);

const initial = new Set();
const queue = [...entry, ...preloaded];
while (queue.length) {
  const name = queue.shift();
  if (initial.has(name) || !info.has(name)) continue;
  initial.add(name);
  queue.push(...staticImports(info.get(name).source));
}

const sum = (names, key) => names.reduce((total, name) => total + (info.get(name)?.[key] ?? 0), 0);
const kb = (bytes) => `${(bytes / 1024).toFixed(1)} KB`;

const workers = js.filter((name) => /worker/i.test(name));
const lazy = js.filter((name) => !initial.has(name) && !workers.includes(name));

const report = {
  initialJs: { files: [...initial], raw: sum([...initial], 'raw'), gzip: sum([...initial], 'gzip'), brotli: sum([...initial], 'brotli') },
  initialCss: { raw: sum(cssLinks, 'raw'), gzip: sum(cssLinks, 'gzip') },
  totalJs: { count: js.length, raw: sum(js, 'raw'), gzip: sum(js, 'gzip'), brotli: sum(js, 'brotli') },
  lazyChunks: lazy.length,
  workerChunks: workers.length,
  largest: [...js]
    .sort((a, b) => info.get(b).raw - info.get(a).raw)
    .slice(0, 12)
    .map((name) => ({
      name,
      raw: info.get(name).raw,
      gzip: info.get(name).gzip,
      kind: initial.has(name) ? 'initial' : workers.includes(name) ? 'worker' : 'lazy',
    })),
};

if (process.argv.includes('--json')) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log('\nBundle de produção (dist/)\n');
  console.log(`JS inicial      ${kb(report.initialJs.raw).padStart(10)} bruto  ${kb(report.initialJs.gzip).padStart(10)} gzip  ${kb(report.initialJs.brotli).padStart(10)} brotli  (${report.initialJs.files.length} arquivos)`);
  console.log(`CSS inicial     ${kb(report.initialCss.raw).padStart(10)} bruto  ${kb(report.initialCss.gzip).padStart(10)} gzip`);
  console.log(`JS total        ${kb(report.totalJs.raw).padStart(10)} bruto  ${kb(report.totalJs.gzip).padStart(10)} gzip  ${kb(report.totalJs.brotli).padStart(10)} brotli  (${report.totalJs.count} arquivos)`);
  console.log(`Chunks sob demanda: ${report.lazyChunks} · workers: ${report.workerChunks}\n`);
  console.log('Maiores chunks:');
  for (const chunk of report.largest) {
    console.log(`  ${chunk.kind.padEnd(8)} ${kb(chunk.raw).padStart(10)} ${kb(chunk.gzip).padStart(10)} gz  ${chunk.name}`);
  }
}
