import type { PDFDocumentProxy } from 'pdfjs-dist';

import { detectTextLayer } from '@/core/review/evidence';
import type { StudyDocument } from '@/core/review/types';
import { getPdf, putPdf, sha256, type StoredPdf } from './pdf-store';

/**
 * pdf.js sob demanda: a biblioteca (e o worker dela) só é baixada quando alguém abre ou
 * anexa um PDF — quem não usa a revisão não paga por ela.
 */

type PdfJs = typeof import('pdfjs-dist');
let pdfjsPromise: Promise<PdfJs> | null = null;

export function loadPdfJs(): Promise<PdfJs> {
  pdfjsPromise ??= Promise.all([import('pdfjs-dist'), import('pdfjs-dist/build/pdf.worker.min.mjs?url')]).then(
    ([pdfjs, worker]) => {
      pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
      return pdfjs;
    },
  );
  return pdfjsPromise;
}

/** Abre o PDF. O pdf.js toma posse do buffer que recebe, então vai uma cópia. */
export async function openPdf(data: Blob | ArrayBuffer): Promise<PDFDocumentProxy> {
  const pdfjs = await loadPdfJs();
  const bytes = new Uint8Array(data instanceof Blob ? await data.arrayBuffer() : data.slice(0));
  return pdfjs.getDocument({ data: bytes }).promise;
}

/** Texto de cada página, na ordem de leitura que o pdf.js dá. */
export async function extractPageTexts(doc: PDFDocumentProxy): Promise<string[]> {
  const texts: string[] = [];
  for (let number = 1; number <= doc.numPages; number += 1) {
    const page = await doc.getPage(number);
    const content = await page.getTextContent();
    texts.push(
      content.items
        .map((item) => ('str' in item ? item.str + (item.hasEOL ? '\n' : '') : ''))
        .join('')
        .replace(/[ \t]+/g, ' '),
    );
    page.cleanup();
  }
  return texts;
}

const DOI_PATTERN = /\b10\.\d{4,9}\/[^\s"<>]+/i;

export interface IngestedPdf {
  document: StudyDocument;
  /** DOI achado nas duas primeiras páginas — o que casa o arquivo com o registro no envio em lote. */
  doi: string | null;
  firstPageText: string;
}

/**
 * Lê o arquivo, extrai o texto, detecta se é escaneado e guarda no banco de PDFs. Um
 * arquivo que já está lá (mesmo hash) não é processado de novo.
 */
export async function ingestPdf(file: File): Promise<IngestedPdf> {
  const buffer = await file.arrayBuffer();
  const hash = await sha256(buffer);
  let stored = await getPdf(hash);
  let pages: number;
  if (stored) {
    pages = stored.pageTexts.length;
  } else {
    const doc = await openPdf(buffer);
    try {
      pages = doc.numPages;
      stored = {
        hash,
        name: file.name,
        data: new Blob([buffer], { type: 'application/pdf' }),
        pageTexts: await extractPageTexts(doc),
        storedAt: new Date().toISOString(),
      } satisfies StoredPdf;
    } finally {
      void doc.loadingTask.destroy();
    }
    await putPdf(stored);
  }
  const head = stored.pageTexts.slice(0, 2).join('\n');
  return {
    document: {
      hash,
      name: file.name,
      size: file.size,
      pages,
      textLayer: detectTextLayer(stored.pageTexts),
      addedAt: new Date().toISOString(),
    },
    doi: head.match(DOI_PATTERN)?.[0]?.replace(/[.,;)\]]+$/, '') ?? null,
    firstPageText: stored.pageTexts[0] ?? '',
  };
}
