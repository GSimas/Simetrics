import * as React from 'react';
import { Check } from 'lucide-react';

import { cn } from '@/lib/utils';

export interface CheckboxProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onChange' | 'value'> {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  /** Cor quando marcada: o destaque da marca, ou o vermelho de exclusão da revisão. */
  tone?: 'highlight' | 'exclude';
}

/**
 * Caixa de seleção do Simetrics, no lugar da do sistema operacional — a nativa muda de
 * cara em cada navegador e celular. Um `button` com `role="checkbox"`: dentro de um
 * `<label>`, o clique no texto também alterna (o botão é o controle rotulável do label).
 */
const Checkbox = React.forwardRef<HTMLButtonElement, CheckboxProps>(
  ({ checked, onCheckedChange, tone = 'highlight', className, onClick, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      role="checkbox"
      aria-checked={checked}
      data-state={checked ? 'checked' : 'unchecked'}
      onClick={(event) => {
        onClick?.(event);
        if (!event.defaultPrevented) onCheckedChange(!checked);
      }}
      className={cn(
        'inline-flex size-4 shrink-0 cursor-pointer items-center justify-center rounded-[3px] border',
        'transition-[background-color,border-color,box-shadow,transform] duration-200 ease-out active:scale-90 motion-reduce:transform-none',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
        checked
          ? tone === 'exclude'
            ? 'border-exclude bg-exclude text-white shadow-[0_0_12px_-3px_var(--glow-exclude)]'
            : 'border-highlight bg-highlight text-primary-foreground shadow-[0_0_12px_-3px_var(--highlight)]'
          : 'border-input bg-transparent hover:border-highlight/70',
        className,
      )}
      {...props}
    >
      {checked && <Check className="size-3 animate-in zoom-in-50 duration-150" strokeWidth={3} aria-hidden />}
    </button>
  ),
);
Checkbox.displayName = 'Checkbox';

export { Checkbox };
