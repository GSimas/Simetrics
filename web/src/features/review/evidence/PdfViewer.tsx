import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { PDFDocumentProxy, PDFPageProxy, RenderTask } from 'pdfjs-dist';
import { Loader2, Maximize2, ZoomIn, ZoomOut } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { quoteContext, relocate } from '@/core/review/evidence';
import type { PageRect } from '@/core/review/types';
import { loadPdfJs, openPdf } from '@/lib/pdf';
import { cn } from '@/lib/utils';
import type { EvidenceCopy } from './copy';

/**
 * Leitor de PDF com camada de texto selecionável, destaques das evidências e marcação de
 * área (tabelas, figuras, páginas escaneadas).
 *
 * As páginas são desenhadas só perto da área visível e descartadas ao se afastar: um
 * artigo de 30 páginas em tela de alta densidade passaria de 200 MB de canvas.
 */

export interface ViewerHighlight {
  id: string;
  page: number;
  quote: string;
  prefix: string;
  suffix: string;
  rects: PageRect[];
  tone: 'active' | 'muted' | 'warning';
}

export interface CapturedSelection {
  page: number;
  quote: string;
  prefix: string;
  suffix: string;
  rects: PageRect[];
}

interface PageSize {
  width: number;
  height: number;
}

const PAGE_GAP = 12;
const NO_PAGES: PDFPageProxy[] = [];
const NO_SIZES: PageSize[] = [];
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;

/** Texto da camada de texto de uma página, com a posição de cada nó — o que liga o texto ao DOM. */
function textIndex(root: Element): { text: string; nodes: Text[]; starts: number[] } {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  const starts: number[] = [];
  let text = '';
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    nodes.push(node as Text);
    starts.push(text.length);
    text += (node as Text).data;
  }
  return { text, nodes, starts };
}

/** Posição no texto da página de um ponto de seleção (nó + deslocamento). */
function offsetOf(index: ReturnType<typeof textIndex>, node: Node, offset: number): number | null {
  if (node.nodeType === Node.TEXT_NODE) {
    const at = index.nodes.indexOf(node as Text);
    return at >= 0 ? index.starts[at]! + offset : null;
  }
  // Ponto num elemento: o primeiro nó de texto a partir do filho indicado.
  const child = node.childNodes[offset] ?? null;
  for (let at = 0; at < index.nodes.length; at += 1) {
    const textNode = index.nodes[at]!;
    if (!child || child === textNode || child.contains(textNode) || child.compareDocumentPosition(textNode) & Node.DOCUMENT_POSITION_FOLLOWING) {
      return index.starts[at]!;
    }
  }
  return index.text.length;
}

function rangeAt(index: ReturnType<typeof textIndex>, start: number, end: number): Range | null {
  const locate = (position: number, isEnd: boolean): [Text, number] | null => {
    for (let at = index.nodes.length - 1; at >= 0; at -= 1) {
      const begin = index.starts[at]!;
      if (begin < position || (!isEnd && begin === position)) {
        return [index.nodes[at]!, Math.min(position - begin, index.nodes[at]!.data.length)];
      }
    }
    return index.nodes[0] ? [index.nodes[0], 0] : null;
  };
  const from = locate(start, false);
  const to = locate(end, true);
  if (!from || !to) return null;
  const range = document.createRange();
  range.setStart(from[0], from[1]);
  range.setEnd(to[0], to[1]);
  return range;
}

function normalizedRects(range: Range, box: DOMRect): PageRect[] {
  return [...range.getClientRects()]
    .filter((rect) => rect.width > 1 && rect.height > 1)
    .map((rect) => ({
      x: (rect.left - box.left) / box.width,
      y: (rect.top - box.top) / box.height,
      w: rect.width / box.width,
      h: rect.height / box.height,
    }));
}

const TONE: Record<ViewerHighlight['tone'], string> = {
  active: 'bg-highlight/30 ring-1 ring-highlight/70',
  muted: 'bg-highlight/12',
  warning: 'bg-amber-400/30 ring-1 ring-amber-500/70',
};

function PdfPage({
  page,
  number,
  size,
  scale,
  root,
  highlights,
  focusId,
  focusNonce,
  areaMode,
  onArea,
  registerElement,
}: {
  page: PDFPageProxy | undefined;
  number: number;
  size: PageSize;
  scale: number;
  root: HTMLElement | null;
  highlights: ViewerHighlight[];
  focusId: string | null;
  focusNonce: number;
  areaMode: boolean;
  onArea: (page: number, rect: PageRect) => void;
  registerElement: (page: number, element: HTMLDivElement | null) => void;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textRef = useRef<HTMLDivElement | null>(null);
  const [near, setNear] = useState(false);
  const [textVersion, setTextVersion] = useState(0);
  const [placed, setPlaced] = useState<Record<string, PageRect[]>>({});
  const [draft, setDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  const width = size.width * scale;
  const height = size.height * scale;

  useEffect(() => {
    const box = boxRef.current;
    if (!box || !root) return;
    const observer = new IntersectionObserver(([entry]) => setNear(!!entry?.isIntersecting), { root, rootMargin: '900px 0px' });
    observer.observe(box);
    return () => observer.disconnect();
  }, [root]);

  // Desenho da página e da camada de texto.
  useEffect(() => {
    const canvas = canvasRef.current;
    const textLayer = textRef.current;
    if (!page || !near || !canvas || !textLayer) return;
    let cancelled = false;
    let renderTask: RenderTask | null = null;
    let textTask: { cancel: () => void } | null = null;
    const viewport = page.getViewport({ scale });
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(viewport.width * ratio);
    canvas.height = Math.floor(viewport.height * ratio);
    renderTask = page.render({ canvas, viewport, ...(ratio !== 1 ? { transform: [ratio, 0, 0, ratio, 0, 0] } : {}) });
    renderTask.promise.catch(() => undefined);

    textLayer.replaceChildren();
    textLayer.style.setProperty('--total-scale-factor', String(scale));
    void loadPdfJs().then(({ TextLayer }) => {
      if (cancelled) return;
      const layer = new TextLayer({ textContentSource: page.streamTextContent(), container: textLayer, viewport });
      textTask = layer;
      layer
        .render()
        .then(() => !cancelled && setTextVersion((version) => version + 1))
        .catch(() => undefined);
    });
    return () => {
      cancelled = true;
      renderTask?.cancel();
      textTask?.cancel();
    };
  }, [page, near, scale]);

  // Longe da área visível, a página libera a memória do canvas.
  useEffect(() => {
    if (near) return;
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = 0;
      canvas.height = 0;
    }
    textRef.current?.replaceChildren();
  }, [near]);

  // Destaques: o trecho é reencontrado na camada de texto; sem ela, valem os retângulos salvos.
  useLayoutEffect(() => {
    const textLayer = textRef.current;
    const box = boxRef.current;
    if (!textLayer || !box) return;
    const index = textVersion > 0 ? textIndex(textLayer) : null;
    const bounds = box.getBoundingClientRect();
    const next: Record<string, PageRect[]> = {};
    for (const highlight of highlights) {
      let rects = highlight.rects;
      if (index && highlight.quote) {
        const match = relocate(index.text, highlight.quote, highlight.prefix, highlight.suffix);
        const range = match ? rangeAt(index, match.start, match.end) : null;
        const found = range ? normalizedRects(range, bounds) : [];
        if (found.length > 0) rects = found;
      }
      next[highlight.id] = rects;
    }
    setPlaced(next);
  }, [highlights, textVersion, scale]);

  // Ir ao trecho: centraliza o destaque e o faz pulsar — uma vez por pedido, não a cada
  // recálculo dos destaques.
  const handledFocus = useRef(-1);
  useEffect(() => {
    if (!focusId || !placed[focusId]?.length || handledFocus.current === focusNonce) return;
    handledFocus.current = focusNonce;
    const element = boxRef.current?.querySelector(`[data-highlight="${CSS.escape(focusId)}"]`);
    element?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    element?.animate(
      [{ boxShadow: '0 0 0 0 var(--highlight)' }, { boxShadow: '0 0 0 8px transparent' }],
      { duration: 900, iterations: 2 },
    );
  }, [focusId, focusNonce, placed]);

  const pointFrom = (event: React.PointerEvent): { x: number; y: number } => {
    const bounds = boxRef.current!.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    };
  };

  return (
    <div
      ref={(element) => {
        boxRef.current = element;
        registerElement(number, element);
      }}
      data-page-number={number}
      className="relative mx-auto bg-white shadow-sm ring-1 ring-black/10"
      style={{ width, height, marginBottom: PAGE_GAP }}
    >
      <canvas ref={canvasRef} className="absolute inset-0" style={{ width, height }} aria-hidden />
      <div className="pointer-events-none absolute inset-0 z-[1]" aria-hidden>
        {highlights.flatMap((highlight) =>
          (placed[highlight.id] ?? []).map((rect, index) => (
            <span
              key={`${highlight.id}-${index}`}
              data-highlight={index === 0 ? highlight.id : undefined}
              className={cn('absolute rounded-[2px] mix-blend-multiply transition-colors duration-300', TONE[highlight.tone])}
              style={{ left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` }}
            />
          )),
        )}
      </div>
      <div ref={textRef} className="textLayer z-[2]" />
      {areaMode && (
        <div
          className="absolute inset-0 z-[3] cursor-crosshair touch-none"
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            const point = pointFrom(event);
            setDraft({ x0: point.x, y0: point.y, x1: point.x, y1: point.y });
          }}
          onPointerMove={(event) => {
            if (!draft) return;
            const point = pointFrom(event);
            setDraft({ ...draft, x1: point.x, y1: point.y });
          }}
          onPointerUp={() => {
            if (!draft) return;
            const rect = {
              x: Math.min(draft.x0, draft.x1),
              y: Math.min(draft.y0, draft.y1),
              w: Math.abs(draft.x1 - draft.x0),
              h: Math.abs(draft.y1 - draft.y0),
            };
            setDraft(null);
            if (rect.w > 0.01 && rect.h > 0.01) onArea(number, rect);
          }}
        >
          {draft && (
            <span
              className="absolute border-2 border-dashed border-highlight bg-highlight/15"
              style={{
                left: `${Math.min(draft.x0, draft.x1) * 100}%`,
                top: `${Math.min(draft.y0, draft.y1) * 100}%`,
                width: `${Math.abs(draft.x1 - draft.x0) * 100}%`,
                height: `${Math.abs(draft.y1 - draft.y0) * 100}%`,
              }}
            />
          )}
        </div>
      )}
      <span className="pointer-events-none absolute -bottom-0.5 right-1 z-[4] translate-y-full text-[10px] tabular-nums text-muted-foreground">
        {number}
      </span>
    </div>
  );
}

export function PdfViewer({
  data,
  highlights,
  focus,
  areaMode,
  onArea,
  selectionAction,
  copy,
  className,
}: {
  data: Blob;
  highlights: ViewerHighlight[];
  /** Evidência a mostrar; o `nonce` repete o pulo quando a mesma é escolhida de novo. */
  focus: { id: string; page: number; nonce: number } | null;
  areaMode: boolean;
  onArea: (page: number, rect: PageRect) => void;
  /** O que oferecer sobre um trecho selecionado (o botão "usar como evidência"). */
  selectionAction: (selection: CapturedSelection, clear: () => void) => ReactNode;
  copy: EvidenceCopy;
  className?: string;
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [root, setRoot] = useState<HTMLDivElement | null>(null);
  // O documento aberto vale para o `data` que o abriu; outro arquivo começa do zero.
  const [opened, setOpened] = useState<{ data: Blob; doc: PDFDocumentProxy; pages: PDFPageProxy[]; sizes: PageSize[] } | null>(null);
  const [failure, setFailure] = useState<{ data: Blob; message: string } | null>(null);
  const current = opened?.data === data ? opened : null;
  const doc = current?.doc ?? null;
  const pages = current?.pages ?? NO_PAGES;
  const sizes = current?.sizes ?? NO_SIZES;
  const error = failure?.data === data ? failure.message : null;
  const [containerWidth, setContainerWidth] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [selection, setSelection] = useState<{ value: CapturedSelection; top: number; left: number } | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const elements = useRef(new Map<number, HTMLDivElement>());

  useEffect(() => {
    let cancelled = false;
    let proxy: PDFDocumentProxy | null = null;
    void (async () => {
      try {
        proxy = await openPdf(data);
        const loaded: PDFPageProxy[] = [];
        for (let number = 1; number <= proxy.numPages; number += 1) loaded.push(await proxy.getPage(number));
        if (cancelled) return;
        const pageSizes = loaded.map((page) => {
          const viewport = page.getViewport({ scale: 1 });
          return { width: viewport.width, height: viewport.height };
        });
        setOpened({ data, doc: proxy, pages: loaded, sizes: pageSizes });
      } catch (cause) {
        if (!cancelled) setFailure({ data, message: cause instanceof Error ? cause.message : String(cause) });
      }
    })();
    return () => {
      cancelled = true;
      void proxy?.loadingTask.destroy();
    };
  }, [data]);

  // Largura útil (sem o recuo) medida já na montagem — o ResizeObserver só avisa depois do
  // primeiro quadro desenhado, e não avisa enquanto a janela está oculta.
  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const measure = (): void => {
      const style = getComputedStyle(element);
      setContainerWidth(element.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const widest = useMemo(() => Math.max(1, ...sizes.map((size) => size.width)), [sizes]);
  const fit = containerWidth > 0 ? (containerWidth - 24) / widest : 1;
  const scale = Math.max(0.2, fit * zoom);

  const registerElement = useCallback((page: number, element: HTMLDivElement | null) => {
    if (element) elements.current.set(page, element);
    else elements.current.delete(page);
  }, []);

  // Ir a uma evidência: primeiro a página (que então é desenhada), depois o destaque.
  useEffect(() => {
    if (!focus) return;
    elements.current.get(focus.page)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, [focus]);

  const byPage = useMemo(() => {
    const map = new Map<number, ViewerHighlight[]>();
    for (const highlight of highlights) map.set(highlight.page, [...(map.get(highlight.page) ?? []), highlight]);
    return map;
  }, [highlights]);

  const captureSelection = (): void => {
    const current = window.getSelection();
    const scroller = scrollRef.current;
    if (!current || current.isCollapsed || current.rangeCount === 0 || !scroller) return setSelection(null);
    const range = current.getRangeAt(0);
    const startPage = (range.startContainer.parentElement ?? (range.startContainer as Element)).closest?.('[data-page-number]');
    const endPage = (range.endContainer.parentElement ?? (range.endContainer as Element)).closest?.('[data-page-number]');
    // Um trecho por página: seleções que atravessam páginas viram duas evidências, à mão.
    if (!startPage || startPage !== endPage || !scroller.contains(startPage)) return setSelection(null);
    const textLayer = startPage.querySelector('.textLayer');
    const quote = current.toString().replace(/\s+/g, ' ').trim();
    if (!textLayer || quote.length < 2) return setSelection(null);

    const index = textIndex(textLayer);
    const start = offsetOf(index, range.startContainer, range.startOffset);
    const end = offsetOf(index, range.endContainer, range.endOffset);
    const context = start !== null && end !== null ? quoteContext(index.text, start, end) : { prefix: '', suffix: '' };
    const rects = normalizedRects(range, startPage.getBoundingClientRect());
    const last = range.getClientRects()[range.getClientRects().length - 1] ?? range.getBoundingClientRect();
    const frame = scroller.getBoundingClientRect();
    setSelection({
      value: { page: Number(startPage.getAttribute('data-page-number')), quote, ...context, rects },
      top: last.bottom - frame.top + scroller.scrollTop + 6,
      left: Math.min(Math.max(8, last.left - frame.left), frame.width - 260),
    });
  };

  const clearSelection = useCallback(() => {
    window.getSelection()?.removeAllRanges();
    setSelection(null);
  }, []);

  return (
    <div className={cn('sm-pdf flex min-h-0 flex-col', className)}>
      <div className="flex flex-wrap items-center gap-1 border-b border-border/80 px-2 py-1.5 text-xs">
        <Button type="button" variant="ghost" size="icon" className="size-7" onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.2))} aria-label={copy.zoomOut} title={copy.zoomOut}>
          <ZoomOut aria-hidden />
        </Button>
        <span className="w-10 text-center tabular-nums text-muted-foreground">{Math.round(zoom * 100)}%</span>
        <Button type="button" variant="ghost" size="icon" className="size-7" onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.2))} aria-label={copy.zoomIn} title={copy.zoomIn}>
          <ZoomIn aria-hidden />
        </Button>
        <Button type="button" variant="ghost" size="icon" className="size-7" onClick={() => setZoom(1)} aria-label={copy.fitWidth} title={copy.fitWidth}>
          <Maximize2 aria-hidden />
        </Button>
        {doc && (
          <span className="ml-auto tabular-nums text-muted-foreground">
            {copy.page} {currentPage} / {doc.numPages}
          </span>
        )}
      </div>
      <div
        ref={(element) => {
          scrollRef.current = element;
          setRoot(element);
        }}
        className="relative min-h-0 flex-1 overflow-auto bg-muted/40 px-3 py-3"
        onMouseUp={() => !areaMode && window.setTimeout(captureSelection, 0)}
        onScroll={(event) => {
          const top = event.currentTarget.scrollTop;
          let page = 1;
          for (const [number, element] of elements.current) if (element.offsetTop - 40 <= top) page = Math.max(page, number);
          setCurrentPage(page);
        }}
      >
        {error ? (
          <p className="p-6 text-center text-sm text-exclude">{copy.readFailed.replace('{error}', error)}</p>
        ) : !doc ? (
          <p className="flex items-center justify-center gap-2 p-10 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" aria-hidden />
            {copy.reading}
          </p>
        ) : (
          sizes.map((size, index) => (
            <PdfPage
              key={index + 1}
              page={pages[index]}
              number={index + 1}
              size={size}
              scale={scale}
              root={root}
              highlights={byPage.get(index + 1) ?? []}
              focusId={focus?.page === index + 1 ? focus.id : null}
              focusNonce={focus?.nonce ?? 0}
              areaMode={areaMode}
              onArea={onArea}
              registerElement={registerElement}
            />
          ))
        )}
        {selection && (
          <div
            className="absolute z-10 animate-in fade-in-0 zoom-in-95 duration-150"
            style={{ top: selection.top, left: selection.left }}
            onMouseUp={(event) => event.stopPropagation()}
          >
            {selectionAction(selection.value, clearSelection)}
          </div>
        )}
      </div>
    </div>
  );
}
