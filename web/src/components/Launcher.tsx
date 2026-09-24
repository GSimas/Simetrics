import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ArrowUpRight, X, type LucideIcon } from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { InfoTip } from '@/components/InfoTip';
import { usePresence } from '@/lib/use-presence';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

export interface LauncherProps {
  Icon: LucideIcon;
  title: string;
  /** Uma linha no cartão dizendo o que há dentro — de preferência com um número real. */
  summary: ReactNode;
  /** Como ler / como usar: vai para o "i" do cabeçalho do modal, não para a tela. */
  info?: ReactNode;
  /** Conteúdo do modal: gráficos, tabelas, controles. */
  children: ReactNode;
  /**
   * Mantém o conteúdo montado depois de fechar. Por padrão ele é desmontado — o que
   * adia o cálculo até o usuário pedir —, mas um formulário perderia o que foi digitado.
   */
  keepMounted?: boolean;
  disabled?: boolean;
  className?: string;
  /** Âncora do tour guiado (`data-tour`). */
  tour?: string;
}

/**
 * Bloco que abre um modal, no padrão do EcoGrad: a aba mostra só um índice de cartões e
 * o usuário vai abrindo, um de cada vez, o que quer consultar.
 *
 * O modal é próprio (não Radix) porque vive fora de `#root` e torna `#root` inerte: os
 * menus do Radix (Select, Dialog de confirmação) continuam portados para `body` e seguem
 * clicáveis por cima dele, o que um Dialog modal do Radix bloquearia.
 */
export function Launcher({
  Icon,
  title,
  summary,
  info,
  children,
  keepMounted = false,
  disabled = false,
  className,
  tour,
}: LauncherProps) {
  const t = useLocale((state) => state.t);
  const titleId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  const { mounted, closing } = usePresence(open);

  useEffect(() => {
    if (!open) return;

    const root = document.getElementById('root');
    const trigger = triggerRef.current;
    const previousOverflow = document.body.style.overflow;
    if (root) root.inert = true;
    document.body.style.overflow = 'hidden';
    panelRef.current?.focus();
    // Um gráfico que ficou montado e escondido não acompanhou mudanças de tamanho.
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      // Esc pertence ao que estiver por cima: uma dica aberta, um Select, um diálogo.
      if (document.querySelector(':popover-open')) return;
      const focused = document.activeElement;
      if (focused && focused !== document.body && !panelRef.current?.contains(focused)) return;
      setOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);

    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', onKeyDown);
      if (root) root.inert = false;
      document.body.style.overflow = previousOverflow;
      trigger?.focus();
    };
  }, [open]);

  const show = (): void => {
    setEverOpened(true);
    setOpen(true);
  };

  const renderBody = mounted || (keepMounted && everOpened);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        data-tour={tour}
        onClick={show}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        className={cn(
          'group flex h-full cursor-pointer flex-col items-start gap-3 border border-border bg-card p-5 text-left transition-colors duration-200 animate-in fade-in-0 hover:border-highlight/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
          className,
        )}
      >
        <span className="flex w-full items-center justify-between text-highlight">
          <Icon className="size-5" aria-hidden />
          <ArrowUpRight
            className="size-4 text-muted-foreground transition-colors group-hover:text-highlight"
            aria-hidden
          />
        </span>
        <span className="text-base font-semibold tracking-tight text-foreground">{title}</span>
        <span className="text-xs leading-relaxed text-muted-foreground">{summary}</span>
      </button>

      {renderBody &&
        createPortal(
          <div
            // Fundo: o clique fora fecha (atalho de mouse); no teclado, o Esc.
            role="presentation"
            hidden={!mounted}
            className={cn(
              'fixed inset-0 z-50 flex items-center justify-center bg-ink/75 p-2 backdrop-blur-sm duration-200 sm:p-4',
              closing ? 'animate-out fade-out-0 fill-mode-forwards' : 'animate-in fade-in-0',
            )}
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) setOpen(false);
            }}
          >
            <div
              ref={panelRef}
              role="dialog"
              aria-modal="true"
              aria-labelledby={titleId}
              tabIndex={-1}
              className={cn(
                'flex max-h-[92dvh] w-full max-w-6xl flex-col overflow-hidden border border-border bg-background text-foreground shadow-2xl outline-none duration-200',
                closing
                  ? 'animate-out fade-out-0 zoom-out-[0.97] fill-mode-forwards'
                  : 'animate-in fade-in-0 zoom-in-[0.97] slide-in-from-bottom-2',
              )}
            >
              <div className="flex items-center gap-3 border-b border-border px-5 py-4">
                <Icon className="size-5 shrink-0 text-highlight" aria-hidden />
                <h2 id={titleId} className="min-w-0 flex-1 truncate text-lg font-medium tracking-tight">
                  {title}
                </h2>
                {info && <InfoTip label={`${t('info_how_to_read')}: ${title}`}>{info}</InfoTip>}
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  aria-label={`${t('modal_close')} ${title}`}
                  className="inline-flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full border border-border transition-colors hover:border-highlight hover:text-highlight"
                >
                  <X className="size-4" aria-hidden />
                </button>
              </div>
              <div className="flex-1 space-y-4 overflow-y-auto p-5">
                <ErrorBoundary variant="page" label={title}>
                  {children}
                </ErrorBoundary>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

/**
 * Grade de blocos com o rótulo "— Aprofundar a análise". `intro` fica entre o rótulo e a
 * grade, para controles que precisam estar sempre à vista.
 */
export function LauncherGrid({
  label,
  intro,
  children,
}: {
  label: string;
  intro?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section data-tour="launchers" className="space-y-3" aria-label={label}>
      <p className="eyebrow">— {label}</p>
      {intro}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">{children}</div>
    </section>
  );
}
