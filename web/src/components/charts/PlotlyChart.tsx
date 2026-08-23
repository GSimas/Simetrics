import { useEffect, useRef } from 'react';
import Plotly, { type Config, type Data, type Layout } from './plotly';

import { cn } from '@/lib/utils';

/**
 * Envelope do Plotly para React.
 *
 * O tema é aplicado a partir dos tokens CSS resolvidos em tempo de execução, e não de
 * cores fixas: o Plotly desenha em canvas e não enxerga as variáveis do Tailwind, então
 * elas precisam ser lidas do DOM e injetadas no layout.
 */

export interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  height?: number;
  className?: string;
  /** Nome do arquivo ao exportar PNG pela barra de ferramentas. */
  exportName?: string;
}

/** Lê um token de cor do tema, com reserva para quando o CSS ainda não aplicou. */
function readToken(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export default function PlotlyChart({
  data,
  layout,
  config,
  height = 420,
  className,
  exportName = 'grafico',
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const foreground = readToken('--foreground', '#1e293b');
    const muted = readToken('--muted-foreground', '#64748b');
    const border = readToken('--border', '#e2e8f0');

    const themedLayout: Partial<Layout> = {
      // Fundo transparente deixa o cartão do tema aparecer por baixo, em vez de um
      // retângulo branco que destoaria no modo escuro.
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: foreground, family: 'inherit', size: 12 },
      margin: { l: 60, r: 24, t: 32, b: 48 },
      height,
      xaxis: { gridcolor: border, zerolinecolor: border, tickfont: { color: muted } },
      yaxis: { gridcolor: border, zerolinecolor: border, tickfont: { color: muted } },
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: muted } },
      hoverlabel: { font: { family: 'inherit' } },
      ...layout,
    };

    const themedConfig: Partial<Config> = {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: { format: 'png', filename: exportName, scale: 2 },
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      locale: 'pt-br',
      ...config,
    };

    void Plotly.newPlot(container, data, themedLayout, themedConfig);

    return () => {
      Plotly.purge(container);
    };
  }, [data, layout, config, height, exportName]);

  return <div ref={containerRef} className={cn('w-full', className)} />;
}
