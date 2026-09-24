import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { isChunkLoadError } from '@/lib/lazy';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

/**
 * Isola falhas de renderização: um gráfico que quebra com um dado atípico (ou um chunk
 * que não baixou) mostra um aviso no próprio lugar, com "Tentar novamente", em vez de
 * desmontar o app inteiro — o comportamento padrão do React 19 sem boundary.
 *
 * `resetKeys`: quando um desses valores muda (outra base, outra aba), o boundary sai do
 * estado de erro sozinho e tenta renderizar de novo.
 */
export interface ErrorBoundaryProps {
  children: ReactNode;
  /** `compact` para gráficos e painéis; `page` para a aba ou o app inteiro. */
  variant?: 'compact' | 'page';
  /** Nome do que falhou, no aviso ("este gráfico", "a aba Redes"). */
  label?: string;
  resetKeys?: readonly unknown[];
  className?: string;
  /** Ação extra no aviso de página (ex.: voltar à lista de projetos). */
  extraAction?: ReactNode;
}

interface State {
  error: Error | null;
  keys: readonly unknown[] | undefined;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, State> {
  override state: State = { error: null, keys: this.props.resetKeys };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  static getDerivedStateFromProps(props: ErrorBoundaryProps, state: State): Partial<State> | null {
    const next = props.resetKeys;
    const previous = state.keys;
    const changed =
      next !== previous &&
      (!next || !previous || next.length !== previous.length || next.some((value, index) => !Object.is(value, previous[index])));
    return changed ? { keys: next, error: null } : null;
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[Simetrics] Falha isolada por ErrorBoundary:', error, info.componentStack);
  }

  private readonly retry = (): void => {
    if (this.state.error && isChunkLoadError(this.state.error)) {
      window.location.reload();
      return;
    }
    this.setState({ error: null });
  };

  override render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <ErrorFallback
        error={error}
        variant={this.props.variant ?? 'compact'}
        label={this.props.label}
        onRetry={this.retry}
        className={this.props.className}
        extraAction={this.props.extraAction}
      />
    );
  }
}

function ErrorFallback({
  error,
  variant,
  label,
  onRetry,
  className,
  extraAction,
}: {
  error: Error;
  variant: 'compact' | 'page';
  label: string | undefined;
  onRetry: () => void;
  className: string | undefined;
  extraAction: ReactNode;
}) {
  const t = useLocale((state) => state.t);
  const chunk = isChunkLoadError(error);
  return (
    <div
      role="alert"
      className={cn(
        'flex flex-col items-center justify-center gap-3 border border-dashed border-border p-6 text-center',
        variant === 'page' ? 'min-h-[40vh]' : 'min-h-40',
        className,
      )}
    >
      <AlertTriangle className="size-6 text-destructive" aria-hidden />
      <div className="space-y-1">
        <p className="text-sm font-semibold text-foreground">
          {label ? t('error_title_named').replace('{name}', label) : t('error_title')}
        </p>
        <p className="max-w-md text-xs text-muted-foreground">
          {chunk ? t('error_chunk_desc') : t('error_desc')}
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button size="sm" variant="outline" onClick={onRetry} className="gap-1.5">
          <RotateCcw className="size-3.5" aria-hidden />
          {chunk ? t('error_reload') : t('error_retry')}
        </Button>
        {extraAction}
      </div>
    </div>
  );
}
