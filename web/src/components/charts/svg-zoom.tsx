import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent, type RefObject } from 'react';
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';

import { CHART_BUTTON_CLASS } from '@/components/charts/ExportImageButton';
import { useLocale } from '@/state/locale.store';

/**
 * Zoom (roda do mouse e botões) e arraste para mover num SVG com `viewBox`. O conteúdo
 * vai num `<g>` com o `style` devolvido; as coordenadas são as do viewBox, via
 * `getScreenCTM`, então funcionam com o SVG em qualquer tamanho na tela.
 *
 * Arrastar não dispara clique: o clique que encerra um arraste é descartado, para não
 * selecionar o país ou nó sob o cursor ao soltar.
 */
interface View {
  k: number;
  x: number;
  y: number;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const BUTTON_STEP = 1.6;
/** Pixels de movimento até um clique virar arraste. */
const DRAG_THRESHOLD = 4;

/** Fração do quadro que o conteúdo precisa continuar cobrindo ao ser arrastado. */
const MIN_VISIBLE = 0.2;

// O pan é livre em qualquer zoom, inclusive 1×, mas para antes de o conteúdo sair do
// quadro: sempre sobra ao menos MIN_VISIBLE dele à vista.
function clamp(next: View, width: number, height: number): View {
  const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next.k));
  const marginX = width * MIN_VISIBLE;
  const marginY = height * MIN_VISIBLE;
  return {
    k,
    x: Math.min(width - marginX, Math.max(marginX - width * k, next.x)),
    y: Math.min(height - marginY, Math.max(marginY - height * k, next.y)),
  };
}

/** Zoom mantendo fixo o ponto (px, py) do viewBox — o que está sob o cursor. */
function zoomView(current: View, factor: number, px: number, py: number, width: number, height: number): View {
  const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.k * factor));
  return clamp(
    { k, x: px - (px - current.x) * (k / current.k), y: py - (py - current.y) * (k / current.k) },
    width,
    height,
  );
}

export function useSvgZoom(ref: RefObject<SVGSVGElement | null>, width: number, height: number) {
  const [view, setView] = useState<View>({ k: 1, x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const viewRef = useRef(view);
  useLayoutEffect(() => {
    viewRef.current = view;
  }, [view]);
  const drag = useRef<{ id: number; x: number; y: number; view: View; moved: boolean } | null>(null);
  const suppressClick = useRef(false);

  const toSvg = (clientX: number, clientY: number): DOMPoint | null => {
    const ctm = ref.current?.getScreenCTM();
    return ctm ? new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse()) : null;
  };

  const zoomAt = (factor: number, px: number, py: number): void =>
    setView((current) => zoomView(current, factor, px, py, width, height));

  // `wheel` precisa de listener não passivo para impedir a rolagem da página.
  useEffect(() => {
    const svg = ref.current;
    if (!svg) return;
    const onWheel = (event: WheelEvent): void => {
      const zoomingOut = event.deltaY > 0;
      // Já no tamanho original, rolar para baixo continua rolando a página.
      if (zoomingOut && viewRef.current.k <= MIN_ZOOM) return;
      event.preventDefault();
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : 1);
      const factor = Math.exp(-delta * 0.002);
      setView((current) => zoomView(current, factor, point.x, point.y, width, height));
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, [ref, width, height]);

  const handlers = {
    onPointerDown: (event: PointerEvent<SVGSVGElement>): void => {
      suppressClick.current = false;
      if (event.button !== 0) return;
      drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, view: viewRef.current, moved: false };
    },
    onPointerMove: (event: PointerEvent<SVGSVGElement>): void => {
      const current = drag.current;
      if (!current || current.id !== event.pointerId) return;
      if (!current.moved) {
        if (Math.hypot(event.clientX - current.x, event.clientY - current.y) < DRAG_THRESHOLD) return;
        current.moved = true;
        setDragging(true);
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      const from = toSvg(current.x, current.y);
      const to = toSvg(event.clientX, event.clientY);
      if (!from || !to) return;
      setView(
        clamp({ k: current.view.k, x: current.view.x + to.x - from.x, y: current.view.y + to.y - from.y }, width, height),
      );
    },
    onPointerUp: (): void => {
      if (drag.current?.moved) suppressClick.current = true;
      drag.current = null;
      setDragging(false);
    },
    onPointerCancel: (): void => {
      drag.current = null;
      setDragging(false);
    },
    onClickCapture: (event: { stopPropagation: () => void }): void => {
      if (!suppressClick.current) return;
      suppressClick.current = false;
      event.stopPropagation();
    },
  };

  const style: CSSProperties = {
    transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`,
    transformOrigin: '0 0',
    // Arrastar segue o cursor sem atraso; roda e botões deslizam até o novo zoom.
    transition: dragging ? 'none' : 'transform 200ms ease-out',
  };

  const cursor = dragging ? 'grabbing' : 'grab';
  const moved = view.k > MIN_ZOOM || view.x !== 0 || view.y !== 0;

  return {
    /** Fator de zoom atual — para afinar traços e marcadores, que senão engrossariam. */
    k: view.k,
    style,
    svgProps: { ...handlers, style: { cursor, touchAction: view.k > MIN_ZOOM ? 'none' : 'pan-y' } as CSSProperties },
    zoomIn: () => zoomAt(BUTTON_STEP, width / 2, height / 2),
    zoomOut: () => zoomAt(1 / BUTTON_STEP, width / 2, height / 2),
    reset: () => setView({ k: 1, x: 0, y: 0 }),
    canZoomIn: view.k < MAX_ZOOM,
    canZoomOut: view.k > MIN_ZOOM,
    /** Há zoom ou deslocamento a desfazer. */
    canReset: moved,
  };
}

export type SvgZoom = ReturnType<typeof useSvgZoom>;

/** Botões de zoom sobre o gráfico, no canto inferior direito. */
export function ZoomControls({ zoom }: { zoom: SvgZoom }) {
  const t = useLocale((state) => state.t);
  const buttons = [
    { Icon: ZoomIn, label: t('chart_zoom_in'), onClick: zoom.zoomIn, disabled: !zoom.canZoomIn },
    { Icon: ZoomOut, label: t('chart_zoom_out'), onClick: zoom.zoomOut, disabled: !zoom.canZoomOut },
    { Icon: RotateCcw, label: t('chart_zoom_reset'), onClick: zoom.reset, disabled: !zoom.canReset },
  ];
  return (
    <div className="absolute bottom-2 right-2 z-10 flex flex-col gap-1" title={t('chart_zoom_hint')}>
      {buttons.map(({ Icon, label, onClick, disabled }) => (
        <button
          key={label}
          type="button"
          onClick={onClick}
          disabled={disabled}
          title={label}
          aria-label={label}
          className={CHART_BUTTON_CLASS}
        >
          <Icon className="size-4" aria-hidden />
        </button>
      ))}
    </div>
  );
}
