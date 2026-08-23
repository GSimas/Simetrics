import type { RisSource } from '@/core/parsers';
import type { Dataset } from '@/lib/types';
import { getIngestWorker } from '@/workers/client';

/**
 * Base de demonstração — ⇄ o "Modo de Demonstração" da barra lateral (Geral.py:317).
 *
 * Os arquivos ficam em `public/demo/` e são servidos como assets estáticos, no lugar da
 * leitura de disco que o Streamlit fazia na raiz do projeto.
 */

export const DEMO_FILES: readonly { name: string; database: string }[] = [
  { name: 'scopus.ris', database: 'Scopus' },
  { name: 'wos.ris', database: 'Web of Science' },
  { name: 'scielo.ris', database: 'SciELO' },
];

/** Baixa os arquivos de exemplo e processa tudo no worker de ingestão. */
export async function loadDemoDataset(): Promise<Dataset> {
  const sources: RisSource[] = await Promise.all(
    DEMO_FILES.map(async ({ name, database }) => {
      const response = await fetch(`${import.meta.env.BASE_URL}demo/${name}`);
      if (!response.ok) {
        throw new Error(`Falha ao carregar ${name}: HTTP ${response.status}`);
      }
      return { name, database, text: await response.text() };
    }),
  );

  const worker = getIngestWorker();

  // Os arquivos de demonstração são todos RIS, então vão direto ao pipeline de RIS —
  // o roteamento por extensão só importa quando o usuário envia os próprios arquivos.
  return worker.ingestRis(sources);
}
