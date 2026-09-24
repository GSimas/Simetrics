import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent,
  type ReactNode,
  type RefObject,
} from 'react';

/** Hooks dos gráficos SVG: largura do contêiner, dica flutuante e zoom por seleção. */

/**
 * Largura real do contêiner. O SVG é desenhado nessa largura (e não esticado a partir
 * de um viewBox fixo): assim o texto mantém o tamanho legível também no celular.
 */
export function useElementWidth<T extends HTMLElement>(fallback = 800): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = (): void => {
      const next = Math.round(element.clientWidth);
      if (next > 0) setWidth(next);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** Dica posicionada em relação ao contêiner (o mesmo que `useElementWidth` mede). */
export function useTooltip(containerRef: RefObject<HTMLElement | null>) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const show = useCallback(
    (event: { clientX: number; clientY: number }, content: ReactNode) => {
      const box = containerRef.current?.getBoundingClientRect();
      if (!box) return;
      setTooltip({ x: event.clientX - box.left, y: event.clientY - box.top, content });
    },
    [containerRef],
  );
  const hide = useCallback(() => setTooltip(null), []);
  return { tooltip, show, hide };
}


export interface PlotArea {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface BrushRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Arrastar com o mouse dentro da área do gráfico desenha uma seleção; soltar chama
 * `onSelect` com o retângulo (coordenadas do SVG). No toque o arraste continua rolando a
 * página — o zoom fica nos botões e no duplo toque.
 */
export function useBrush(
  svgRef: RefObject<SVGSVGElement | null>,
  area: PlotArea,
  axis: 'x' | 'xy',
  onSelect: (rect: BrushRect) => void,
) {
  const [selection, setSelection] = useState<BrushRect | null>(null);
  const start = useRef<{ x: number; y: number; id: number } | null>(null);
  const moved = useRef(false);

  const toSvg = (event: { clientX: number; clientY: number }): { x: number; y: number } | null => {
    const ctm = svgRef.current?.getScreenCTM();
    if (!ctm) return null;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    return { x: point.x, y: point.y };
  };

  const clampX = (x: number): number => Math.min(area.left + area.width, Math.max(area.left, x));
  const clampY = (y: number): number => Math.min(area.top + area.height, Math.max(area.top, y));

  const handlers = {
    onPointerDown: (event: PointerEvent<SVGSVGElement>): void => {
      moved.current = false;
      if (event.pointerType === 'touch' || event.button !== 0) return;
      const point = toSvg(event);
      if (!point) return;
      if (point.x < area.left || point.x > area.left + area.width || point.y < area.top || point.y > area.top + area.height) return;
      start.current = { ...point, id: event.pointerId };
    },
    onPointerMove: (event: PointerEvent<SVGSVGElement>): void => {
      const origin = start.current;
      if (!origin || origin.id !== event.pointerId) return;
      const point = toSvg(event);
      if (!point) return;
      if (!moved.current && Math.hypot(point.x - origin.x, point.y - origin.y) < 5) return;
      if (!moved.current) event.currentTarget.setPointerCapture(event.pointerId);
      moved.current = true;
      setSelection({
        x0: clampX(Math.min(origin.x, point.x)),
        x1: clampX(Math.max(origin.x, point.x)),
        y0: axis === 'x' ? area.top : clampY(Math.min(origin.y, point.y)),
        y1: axis === 'x' ? area.top + area.height : clampY(Math.max(origin.y, point.y)),
      });
    },
    onPointerUp: (): void => {
      if (selection && moved.current && selection.x1 - selection.x0 > 6 && (axis === 'x' || selection.y1 - selection.y0 > 6)) {
        onSelect(selection);
      }
      start.current = null;
      setSelection(null);
    },
    onPointerCancel: (): void => {
      start.current = null;
      setSelection(null);
    },
    /** Um clique que encerrou um arraste não seleciona o ponto sob o cursor. */
    onClickCapture: (event: { stopPropagation: () => void }): void => {
      if (!moved.current) return;
      moved.current = false;
      event.stopPropagation();
    },
  };

  const overlay = selection ? (
    <rect
      x={selection.x0}
      y={selection.y0}
      width={selection.x1 - selection.x0}
      height={selection.y1 - selection.y0}
      fill="var(--highlight)"
      fillOpacity={0.12}
      stroke="var(--highlight)"
      strokeDasharray="4 3"
      pointerEvents="none"
    />
  ) : null;

  return { handlers, overlay };
}
