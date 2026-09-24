import type { ComponentType } from 'react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { useLocale } from '@/state/locale.store';

/**
 * Envolve um gráfico num ErrorBoundary: um dado atípico que quebre o desenho mostra um
 * aviso no lugar do gráfico, com "Tentar novamente", sem levar a aba junto.
 *
 * O boundary se reinicia sozinho quando alguma prop muda (comparação rasa): trocar o
 * filtro ou a base já tenta de novo.
 */
export function withChartBoundary<P extends object>(Chart: ComponentType<P>): ComponentType<P> {
  function Guarded(props: P) {
    const t = useLocale((state) => state.t);
    return (
      <ErrorBoundary label={t('error_chart')} resetKeys={Object.values(props)}>
        <Chart {...props} />
      </ErrorBoundary>
    );
  }
  Guarded.displayName = `withChartBoundary(${Chart.displayName ?? Chart.name})`;
  return Guarded;
}
