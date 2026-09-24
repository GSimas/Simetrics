import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';

import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

export interface InfoTipProps {
  /** Rótulo acessível do botão — "Como ler: Produção ao longo do tempo". */
  label: string;
  children: ReactNode;
  className?: string;
}

const GAP = 8;

/**
 * Botão "i" que abre uma dica curta, no padrão do EcoGrad: as instruções de leitura e uso
 * saem da tela e ficam a um clique.
 *
 * Usa o popover nativo do navegador: ele vai para a camada superior — acima dos modais e
 * fora de qualquer `overflow` que o cortaria — e já fecha com Esc e clique fora. Só o
 * posicionamento é manual, ancorado ao botão e mantido dentro da janela.
 */
export function InfoTip({ label, children, className }: InfoTipProps) {
  const id = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  const place = (): void => {
    const button = buttonRef.current;
    const panel = panelRef.current;
    if (!button || !panel) return;

    const anchor = button.getBoundingClientRect();
    // Ainda fechado (chamada no clique), o painel não tem medida: usa a largura fixa e
    // assume que cabe abaixo; a chamada no `toggle` corrige com as medidas reais.
    const width = panel.offsetWidth || 288;
    const height = panel.offsetHeight;
    const left = Math.min(
      Math.max(GAP, anchor.left + anchor.width / 2 - width / 2),
      window.innerWidth - width - GAP,
    );
    const fitsBelow = anchor.bottom + GAP + height <= window.innerHeight - GAP;
    const top = fitsBelow ? anchor.bottom + GAP : Math.max(GAP, anchor.top - GAP - height);

    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  };

  // Fixo na tela, o painel descolaria do botão ao rolar — então fecha.
  useEffect(() => {
    if (!open) return;
    const close = (): void => panelRef.current?.hidePopover();
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        popoverTarget={id}
        aria-label={label}
        title={label}
        onClick={(event) => {
          event.stopPropagation();
          place();
        }}
        className={cn(
          'inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:border-highlight hover:text-highlight focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          open && 'border-highlight text-highlight',
          className,
        )}
      >
        <Info className="size-3.5" aria-hidden />
      </button>
      <div
        ref={panelRef}
        id={id}
        popover="auto"
        role="tooltip"
        onToggle={(event) => {
          const isOpen = event.newState === 'open';
          setOpen(isOpen);
          if (isOpen) place();
        }}
        className="fixed inset-auto m-0 w-72 max-w-[calc(100vw-16px)] border border-border bg-popover p-3.5 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-popover-foreground shadow-xl"
      >
        <p className="eyebrow mb-1.5 text-highlight">{label}</p>
        <div className="space-y-2 text-muted-foreground [&_strong]:text-foreground">{children}</div>
      </div>
    </>
  );
}

export interface SectionTitleProps {
  title: ReactNode;
  /** Texto de "como ler / como usar", escondido atrás do "i". */
  info?: ReactNode;
  /** Rótulo da dica, quando o título não é texto puro. */
  infoLabel?: string;
  /** Ações alinhadas à direita (botões, contadores). */
  children?: ReactNode;
  className?: string;
}

/** Título de seção com o "i" ao lado — substitui o par título + parágrafo descritivo. */
export function SectionTitle({ title, info, infoLabel, children, className }: SectionTitleProps) {
  return (
    <div className={cn('flex flex-wrap items-center gap-x-2 gap-y-2', className)}>
      <h3 className="text-base font-semibold tracking-tight text-foreground">{title}</h3>
      {info && (
        <InfoTip label={infoLabel ?? (typeof title === 'string' ? title : '')}>{info}</InfoTip>
      )}
      {children && <div className="ml-auto flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

/** "i" seguido do texto "Como ler", para explicações de leitura de um gráfico. */
export function ReadingTip({ children, label }: { children: ReactNode; label?: string }) {
  const t = useLocale((state) => state.t);
  const text = label ?? t('info_how_to_read');
  return (
    <div className="flex items-center gap-2">
      <InfoTip label={text}>{children}</InfoTip>
      <span className="eyebrow">{text}</span>
    </div>
  );
}
