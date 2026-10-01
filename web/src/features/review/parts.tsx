import type { ReactNode } from 'react';
import { Trash2 } from 'lucide-react';

import { cn } from '@/lib/utils';
import { BLOCK, PRESS } from './motion';

/** Bloco com título e dica — a unidade visual dos painéis da revisão. */
export function Block({
  title,
  hint,
  tour,
  target,
  className,
  children,
}: {
  title: string;
  hint?: string;
  /** Alvo do tour guiado (`data-tour`). */
  tour?: string;
  /** Alvo do "próximo passo" (`data-review-target`), que o destaca ao ser escolhido. */
  target?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn('space-y-3 p-4', BLOCK, className)} data-tour={tour} data-review-target={target}>
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">{title}</h3>
        {hint && <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>}
      </div>
      {children}
    </section>
  );
}

/** Remover um item: o ícone acende em vermelho sob o cursor, como toda exclusão da revisão. */
export function RemoveButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex size-9 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-exclude/10 hover:text-exclude hover:shadow-[0_0_16px_-8px_var(--glow-exclude)] disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4',
        PRESS,
      )}
    >
      <Trash2 aria-hidden />
    </button>
  );
}

/**
 * Trecho de formulário que o exemplo (só visualização) mostra sem deixar editar: o
 * `fieldset` desabilitado trava de uma vez todos os campos e botões dentro dele.
 */
export function ReadOnlyScope({ readOnly, className, children }: { readOnly: boolean; className?: string; children: ReactNode }) {
  return (
    <fieldset disabled={readOnly} className={cn('m-0 min-w-0 border-0 p-0', className)}>
      {children}
    </fieldset>
  );
}
