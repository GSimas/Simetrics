import { useEffect, useRef } from 'react';

/**
 * Dicas de texto do Simetrics no lugar das do navegador.
 *
 * Uma camada só, montada uma vez na raiz, cobre o app inteiro: qualquer elemento com
 * `title` (ou `data-tip`) ganha a dica no visual do app, sem precisar trocar cada um dos
 * ~140 `title` espalhados pelos componentes — nem os que ainda vão ser escritos. Ao passar
 * o mouse, o `title` vira `data-tip` antes que o navegador mostre a dica dele; um
 * MutationObserver faz o mesmo quando o React volta a escrever um `title`.
 *
 * Acessibilidade: o texto do `title` era o nome ou a descrição do elemento. Sem nome
 * próprio (ícone sem `aria-label` nem texto), ele vira `aria-label`; com nome, vira
 * `aria-description`. No teclado a dica também aparece, no foco visível.
 *
 * O painel é um popover manual (camada superior, acima de modais e de outros popovers),
 * posicionado acima do elemento — abaixo quando não cabe — e sem eventos de ponteiro.
 */

const SHOW_DELAY = 450;
/** Passando de um elemento com dica a outro logo em seguida, a próxima aparece na hora. */
const SKIP_WINDOW = 400;
const GAP = 8;

function adopt(element: Element): string | null {
  const title = element.getAttribute('title');
  if (title !== null) {
    element.removeAttribute('title');
    const tip = title.trim();
    if (tip) {
      element.setAttribute('data-tip', tip);
      const text = (element.textContent ?? '').trim();
      const named = element.hasAttribute('aria-label') || element.hasAttribute('aria-labelledby') || text !== '';
      if (!named) element.setAttribute('aria-label', tip);
      else if (text !== tip && !element.hasAttribute('aria-description')) element.setAttribute('aria-description', tip);
    }
  }
  return element.getAttribute('data-tip');
}

function findTipped(start: Element | null): Element | null {
  for (let element = start; element && element !== document.documentElement; element = element.parentElement) {
    if (element.hasAttribute('title') || element.hasAttribute('data-tip')) return element;
  }
  return null;
}

export function TooltipLayer() {
  const panelRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const panel = panelRef.current;
    const label = textRef.current;
    if (!panel || !label) return;

    let current: Element | null = null;
    // Depois de um clique, a dica daquele elemento não volta até o ponteiro sair dele.
    let suppressed: Element | null = null;
    let visible = false;
    let lastHidden = 0;
    let showTimer = 0;
    let frame = 0;

    const place = (anchor: Element): void => {
      const rect = anchor.getBoundingClientRect();
      const width = panel.offsetWidth;
      const height = panel.offsetHeight;
      let top = rect.top - GAP - height;
      if (top < GAP) top = rect.bottom + GAP;
      const left = Math.min(Math.max(GAP, rect.left + rect.width / 2 - width / 2), window.innerWidth - width - GAP);
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
    };

    const hide = (): void => {
      window.clearTimeout(showTimer);
      current = null;
      if (!visible) return;
      visible = false;
      lastHidden = Date.now();
      if (panel.matches(':popover-open')) panel.hidePopover();
    };

    const show = (anchor: Element, tip: string): void => {
      label.textContent = tip;
      if (!panel.matches(':popover-open')) panel.showPopover();
      place(anchor);
      visible = true;
    };

    const point = (start: Element | null): void => {
      const tipped = findTipped(start);
      if (tipped !== suppressed) suppressed = null;
      if (tipped === current) return;
      window.clearTimeout(showTimer);
      const tip = tipped && tipped !== suppressed ? adopt(tipped) : null;
      if (!tipped || !tip) {
        hide();
        return;
      }
      current = tipped;
      const instant = visible || Date.now() - lastHidden < SKIP_WINDOW;
      showTimer = window.setTimeout(() => {
        if (current === tipped && tipped.isConnected) show(tipped, tip);
      }, instant ? 0 : SHOW_DELAY);
    };

    // `pointermove` + `elementFromPoint`, e não `pointerover`: botões desabilitados (as dicas
    // de "indisponível no exemplo") nem sempre recebem eventos do mouse.
    let last = { x: 0, y: 0 };
    const onMove = (event: PointerEvent): void => {
      if (event.pointerType !== 'mouse') return;
      last = { x: event.clientX, y: event.clientY };
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        point(document.elementFromPoint(last.x, last.y));
      });
    };
    const onDown = (): void => {
      suppressed = current;
      hide();
    };
    const onFocus = (event: FocusEvent): void => {
      const target = event.target;
      if (!(target instanceof Element) || !target.matches(':focus-visible')) return;
      point(target);
    };
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') hide();
    };
    const onLeave = (): void => hide();

    // O React pode reescrever um `title` num elemento já montado: vira dica do app na hora.
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        const element = mutation.target as Element;
        if (!element.hasAttribute('title')) continue;
        const tip = adopt(element);
        if (element === current && visible && tip) label.textContent = tip;
      }
    });
    observer.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['title'] });

    document.addEventListener('pointermove', onMove, { passive: true });
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('focusin', onFocus);
    document.addEventListener('focusout', onLeave);
    document.addEventListener('keydown', onKey);
    document.documentElement.addEventListener('pointerleave', onLeave);
    window.addEventListener('scroll', onLeave, true);
    window.addEventListener('blur', onLeave);

    return () => {
      observer.disconnect();
      window.clearTimeout(showTimer);
      window.cancelAnimationFrame(frame);
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('focusin', onFocus);
      document.removeEventListener('focusout', onLeave);
      document.removeEventListener('keydown', onKey);
      document.documentElement.removeEventListener('pointerleave', onLeave);
      window.removeEventListener('scroll', onLeave, true);
      window.removeEventListener('blur', onLeave);
    };
  }, []);

  return (
    <div
      ref={panelRef}
      popover="manual"
      role="tooltip"
      aria-hidden
      className="pointer-events-none fixed inset-auto m-0 max-w-xs rounded-md border border-highlight/30 bg-popover/95 px-2.5 py-1.5 text-xs leading-snug text-popover-foreground shadow-[0_10px_28px_-14px_rgb(0_0_0/0.6),0_0_20px_-12px_var(--highlight)] backdrop-blur-sm"
    >
      <span ref={textRef} className="block break-words" />
    </div>
  );
}
