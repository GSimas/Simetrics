import * as React from 'react';

import { cn } from '@/lib/utils';

/**
 * `autoComplete="off"` por padrão: sem ele, o navegador abre por cima do campo a própria
 * lista de valores já digitados — um dropdown do sistema, fora da linguagem do app. Quem
 * precisa do autopreenchimento passa outro valor.
 */
const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<'input'>>(
  ({ className, type, autoComplete = 'off', ...props }, ref) => (
    <input
      type={type}
      ref={ref}
      autoComplete={autoComplete}
      className={cn(
        'flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors',
        'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
        'disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = 'Input';

export { Input };
