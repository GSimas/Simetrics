/**
 * Barras de rolagem sobrepostas do Simetrics. As nativas ficam escondidas (`index.css`) e
 * não ocupam faixa; no lugar delas, uma alça reta desenhada numa camada fixa, por cima do
 * conteúdo, que só aparece enquanto a área rola (roda, teclado, arrasto) ou com o mouse
 * perto da borda onde a barra fica. A alça arrasta e repassa a roda do mouse à área.
 *
 * Vale para qualquer elemento rolável, sem marcar nada no JSX: a rolagem é ouvida na fase
 * de captura da janela e a borda é achada subindo pelos ancestrais do ponteiro.
 *
 * ponytail: a alça não é recortada por ancestrais com `overflow` (só pela tela) e fica
 * abaixo da camada superior (popovers nativos); se um dia incomodar, recortar pela
 * interseção dos retângulos dos ancestrais roláveis.
 */

const EDGE = 28; // px da borda em que a barra aparece
const LINGER = 900; // ms visível depois da última rolagem
const THICK = 8; // espessura da alça
const INSET = 3; // folga entre a alça e a borda
const MIN = 32; // comprimento mínimo da alça

type Axis = 'x' | 'y';

interface Bar {
  el: Element;
  thumbs: Record<Axis, HTMLDivElement>;
  timer: number;
  scrolling: boolean;
  edge: boolean;
  drag: boolean;
}

const root = (): Element => document.scrollingElement ?? document.documentElement;
const isRoot = (el: Element): boolean => el === root() || el === document.documentElement;

function scrolls(el: Element, axis: Axis): boolean {
  if (isRoot(el)) return true;
  const overflow = getComputedStyle(el)[axis === 'y' ? 'overflowY' : 'overflowX'];
  return overflow === 'auto' || overflow === 'scroll';
}

/** Caixa visível (sem bordas) e medidas de rolagem; a da página é a própria janela. */
function measure(el: Element) {
  if (isRoot(el)) {
    const page = root();
    return {
      top: 0,
      left: 0,
      width: window.innerWidth,
      height: window.innerHeight,
      scrollTop: page.scrollTop,
      scrollLeft: page.scrollLeft,
      scrollHeight: page.scrollHeight,
      scrollWidth: page.scrollWidth,
    };
  }
  const r = el.getBoundingClientRect();
  return {
    top: r.top + el.clientTop,
    left: r.left + el.clientLeft,
    width: el.clientWidth,
    height: el.clientHeight,
    scrollTop: el.scrollTop,
    scrollLeft: el.scrollLeft,
    scrollHeight: el.scrollHeight,
    scrollWidth: el.scrollWidth,
  };
}

/** Tamanho e posição da alça no eixo, ou `null` se o eixo não rola. */
function thumbFor(el: Element, axis: Axis) {
  const m = measure(el);
  const view = axis === 'y' ? m.height : m.width;
  const full = axis === 'y' ? m.scrollHeight : m.scrollWidth;
  if (full <= view + 1 || !scrolls(el, axis)) return null;
  const length = Math.max(MIN, (view * view) / full);
  const travel = view - length;
  const offset = ((axis === 'y' ? m.scrollTop : m.scrollLeft) / (full - view)) * travel;
  return { m, length, travel, offset, range: full - view };
}

/** O ponteiro está na faixa da barra (vertical ou horizontal) do elemento? */
function nearEdge(el: Element, x: number, y: number): boolean {
  if (!isRoot(el) && el.scrollHeight <= el.clientHeight && el.scrollWidth <= el.clientWidth) return false;
  const m = measure(el);
  if (x < m.left || x > m.left + m.width || y < m.top || y > m.top + m.height) return false;
  return (
    (x >= m.left + m.width - EDGE && thumbFor(el, 'y') !== null) ||
    (y >= m.top + m.height - EDGE && thumbFor(el, 'x') !== null)
  );
}

export function installOverlayScrollbars(): void {
  const layer = document.createElement('div');
  layer.className = 'sm-scroll-layer';
  layer.setAttribute('aria-hidden', 'true');
  document.body.append(layer);

  const bars = new Map<Element, Bar>();

  const place = (bar: Bar): void => {
    for (const axis of ['y', 'x'] as const) {
      const thumb = bar.thumbs[axis];
      const t = thumbFor(bar.el, axis);
      if (!t || !bar.el.isConnected) {
        thumb.hidden = true;
        continue;
      }
      thumb.hidden = false;
      const { m } = t;
      // Presa à tela: uma área maior que a janela não joga a alça para fora dela.
      if (axis === 'y') {
        const top = Math.max(0, m.top + t.offset);
        const bottom = Math.min(window.innerHeight, m.top + t.offset + t.length);
        Object.assign(thumb.style, {
          left: `${Math.min(window.innerWidth, m.left + m.width) - THICK - INSET}px`,
          top: `${top}px`,
          width: `${THICK}px`,
          height: `${Math.max(0, bottom - top)}px`,
        });
      } else {
        const left = Math.max(0, m.left + t.offset);
        const right = Math.min(window.innerWidth, m.left + t.offset + t.length);
        Object.assign(thumb.style, {
          top: `${Math.min(window.innerHeight, m.top + m.height) - THICK - INSET}px`,
          left: `${left}px`,
          height: `${THICK}px`,
          width: `${Math.max(0, right - left)}px`,
        });
      }
    }
  };

  const refresh = (bar: Bar): void => {
    const on = bar.scrolling || bar.edge || bar.drag;
    for (const thumb of Object.values(bar.thumbs)) thumb.toggleAttribute('data-on', on);
    if (on) {
      place(bar);
      return;
    }
    // Some com o esmaecimento do CSS e depois sai da camada.
    window.setTimeout(() => {
      if (bar.scrolling || bar.edge || bar.drag || bars.get(bar.el) !== bar) return;
      for (const thumb of Object.values(bar.thumbs)) thumb.remove();
      bars.delete(bar.el);
    }, 300);
  };

  const scrollTarget = (el: Element): Element | Window => (isRoot(el) ? window : el);

  const makeThumb = (bar: Bar, axis: Axis): HTMLDivElement => {
    const thumb = document.createElement('div');
    thumb.className = 'sm-scroll-thumb';
    thumb.hidden = true;
    // Sobre a alça, a roda continua rolando a área de baixo.
    thumb.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
        scrollTarget(bar.el).scrollBy(event.deltaX * unit, event.deltaY * unit);
      },
      { passive: false },
    );
    thumb.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      // Sem isso, diálogos e menus do Radix entendem o clique como "fora" e fecham.
      event.preventDefault();
      event.stopPropagation();
      const t = thumbFor(bar.el, axis);
      if (!t || t.travel <= 0) return;
      thumb.setPointerCapture(event.pointerId);
      bar.drag = true;
      thumb.setAttribute('data-drag', '');
      const start = axis === 'y' ? event.clientY : event.clientX;
      const from = axis === 'y' ? t.m.scrollTop : t.m.scrollLeft;
      const move = (e: PointerEvent): void => {
        const to = from + (((axis === 'y' ? e.clientY : e.clientX) - start) * t.range) / t.travel;
        const target = isRoot(bar.el) ? root() : bar.el;
        if (axis === 'y') target.scrollTop = to;
        else target.scrollLeft = to;
      };
      const end = (): void => {
        bar.drag = false;
        thumb.removeAttribute('data-drag');
        thumb.removeEventListener('pointermove', move);
        refresh(bar);
      };
      thumb.addEventListener('pointermove', move);
      thumb.addEventListener('pointerup', end, { once: true });
      thumb.addEventListener('pointercancel', end, { once: true });
    });
    layer.append(thumb);
    return thumb;
  };

  const barFor = (el: Element): Bar => {
    const key = isRoot(el) ? root() : el;
    let bar = bars.get(key);
    if (!bar) {
      const created: Bar = { el: key, timer: 0, scrolling: false, edge: false, drag: false } as Bar;
      created.thumbs = { y: makeThumb(created, 'y'), x: makeThumb(created, 'x') };
      bars.set(key, created);
      bar = created;
    }
    return bar;
  };

  window.addEventListener(
    'scroll',
    (event) => {
      // A rolagem da página chega com `document` como alvo.
      const bar = barFor(event.target instanceof Element ? event.target : root());
      bar.scrolling = true;
      clearTimeout(bar.timer);
      bar.timer = window.setTimeout(() => {
        bar.scrolling = false;
        refresh(bar);
      }, LINGER);
      refresh(bar);
      // A área que rolou pode mover as alças das áreas de dentro dela.
      for (const other of bars.values()) if (other !== bar) place(other);
    },
    { capture: true, passive: true },
  );

  const setEdges = (next: Element[]): void => {
    for (const bar of bars.values()) {
      const edge = next.includes(bar.el);
      if (bar.edge !== edge) {
        bar.edge = edge;
        refresh(bar);
      }
    }
    for (const el of next) {
      const bar = barFor(el);
      if (!bar.edge) {
        bar.edge = true;
        refresh(bar);
      }
    }
  };

  let frame = 0;
  window.addEventListener(
    'pointermove',
    (event) => {
      if (event.pointerType !== 'mouse' || frame) return;
      const { clientX: x, clientY: y, target } = event;
      // Sobre uma alça, ela fica como está (a área de baixo não é ancestral dela).
      if (target instanceof Element && target.classList.contains('sm-scroll-thumb')) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const next: Element[] = [];
        for (let el = target instanceof Element ? target : null; el; el = el.parentElement) {
          if (nearEdge(el, x, y)) next.push(isRoot(el) ? root() : el);
        }
        setEdges(next);
      });
    },
    { passive: true },
  );
  document.documentElement.addEventListener('mouseleave', () => setEdges([]));
  window.addEventListener('resize', () => {
    for (const bar of bars.values()) place(bar);
  });
}
