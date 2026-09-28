import * as React from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Input } from './input';

export interface NumberInputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type' | 'min' | 'max' | 'step'> {
  value: number | null;
  onValueChange: (value: number | null) => void;
  step?: number;
  min?: number | undefined;
  max?: number | undefined;
  /** Classe do contêiner (largura). A do campo vai em `inputClassName`. */
  className?: string;
  inputClassName?: string;
}

function decimals(step: number): number {
  const text = String(step);
  return text.includes('.') ? text.length - text.indexOf('.') - 1 : 0;
}

/**
 * Campo numérico do Simetrics: as setinhas do sistema somem (index.css) e dão lugar a um
 * par de botões na mesma linguagem do app. O teclado continua valendo — ↑/↓ no campo — e
 * no celular o teclado numérico abre pelo `inputMode`.
 */
const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ value, onValueChange, step = 1, min, max, className, inputClassName, disabled, ...props }, ref) => {
    const places = decimals(step);
    const clamp = (next: number): number => {
      const bounded = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, next));
      return Number(bounded.toFixed(places));
    };
    const nudge = (direction: 1 | -1): void => onValueChange(clamp((value ?? min ?? 0) + direction * step));

    const stepper =
      'flex h-1/2 w-6 items-center justify-center text-muted-foreground transition-[color,background-color,transform] duration-150 hover:bg-highlight/10 hover:text-highlight active:scale-90 disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-3';

    return (
      <div className={cn('relative', className)}>
        <Input
          ref={ref}
          type="number"
          inputMode={places > 0 || (min ?? 0) < 0 ? 'decimal' : 'numeric'}
          value={value ?? ''}
          step={step}
          min={min}
          max={max}
          disabled={disabled}
          onChange={(event) => {
            const raw = event.target.value;
            const parsed = Number(raw);
            onValueChange(raw === '' || !Number.isFinite(parsed) ? null : parsed);
          }}
          className={cn('pr-7 tabular-nums', inputClassName)}
          {...props}
        />
        {/* Só para o mouse e o toque: no teclado, ↑/↓ no próprio campo fazem o mesmo. */}
        <div className="absolute inset-y-px right-px flex flex-col overflow-hidden rounded-r-md border-l border-input">
          <button
            type="button"
            tabIndex={-1}
            aria-hidden
            disabled={disabled || (max !== undefined && (value ?? -Infinity) >= max)}
            onClick={() => nudge(1)}
            className={stepper}
          >
            <ChevronUp />
          </button>
          <button
            type="button"
            tabIndex={-1}
            aria-hidden
            disabled={disabled || (min !== undefined && (value ?? Infinity) <= min)}
            onClick={() => nudge(-1)}
            className={cn(stepper, 'border-t border-input')}
          >
            <ChevronDown />
          </button>
        </div>
      </div>
    );
  },
);
NumberInput.displayName = 'NumberInput';

export { NumberInput };
