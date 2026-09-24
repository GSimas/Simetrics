import { useEffect, useRef, useState } from 'react';
import Plotly, { type Config, type Data, type Layout } from './plotly';

import { ExpandChartButton, expandedHeight } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import { resolveCssVariables, type ChartImage } from '@/lib/export-image';
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
  /** Clique num ponto, barra ou nó — recebe o ponto do Plotly. */
  onPointClick?: ((point: PlotPoint) => void) | undefined;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

/** Campos do ponto clicado que os painéis usam para descobrir a entidade. */
export interface PlotPoint {
  x?: unknown;
  y?: unknown;
  text?: unknown;
  label?: unknown;
  location?: unknown;
  customdata?: unknown;
  data?: { name?: string };
}

const FONT_SANS = 'Manrope, Arial, sans-serif';
const FONT_MONO = '"DM Mono", ui-monospace, monospace';

/** Lê um token de cor do tema, com reserva para quando o CSS ainda não aplicou. */
function readToken(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export default function PlotlyChart(props: PlotlyChartProps) {
  const { data, layout, config, height = 420, className, exportName = 'grafico', onPointClick, expanded } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onPointClick);
  const isDark = useIsDark();

  useEffect(() => {
    clickRef.current = onPointClick;
  }, [onPointClick]);

  // Desmonta o gráfico só ao sair da tela; atualizações passam pelo `Plotly.react`.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    return () => {
      Plotly.purge(container);
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const foreground = readToken('--foreground', '#07110f');
    const muted = readToken('--muted-foreground', '#56625d');
    const border = readToken('--border', '#d5d2c6');
    const popover = readToken('--popover', '#f7f6f1');

    // Os eixos do chamador são mesclados ao tema, não substituídos: um `xaxis: { title }`
    // passado de fora apagaria a cor da grade e o gráfico voltaria às linhas brancas
    // padrão do Plotly.
    const axisTheme = {
      gridcolor: border,
      linecolor: border,
      zerolinecolor: border,
      tickfont: { color: muted, family: FONT_MONO, size: 10.5 },
    };

    const themedLayout: Partial<Layout> = {
      // Fundo transparente deixa o cartão do tema aparecer por baixo, em vez de um
      // retângulo branco que destoaria no modo escuro.
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: foreground, family: FONT_SANS, size: 12 },
      margin: { l: 60, r: 24, t: 32, b: 48 },
      height,
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: muted } },
      hoverlabel: {
        bgcolor: popover,
        bordercolor: border,
        font: { family: FONT_SANS, color: foreground },
      },
      ...layout,
      // Ampliado, a altura da janela vence a que o painel fixou no layout.
      ...(expanded ? { height } : {}),
      xaxis: { ...axisTheme, ...layout?.xaxis },
      yaxis: { ...axisTheme, ...layout?.yaxis },
    };

    const themedConfig: Partial<Config> = {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: { format: 'png', filename: exportName, scale: 2 },
      // A exportação é o botão padrão do Simetrics (SVG / JPG / PNG), acima do gráfico.
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
      locale: 'pt-br',
      ...config,
    };

    // `react` compara com o desenho anterior e aplica só a diferença. `newPlot` apagava e
    // redesenhava o gráfico inteiro a cada render do painel — o que piscava a tela sempre
    // que qualquer estado vizinho mudava.
    //
    // Fechar o modal (ou trocar de aba) no meio do desenho faz o `purge` apagar o estado
    // que o Plotly ainda ia usar, e a promessa rejeita. Não há o que recuperar: o gráfico
    // já saiu da tela.
    Plotly.react(container, data, themedLayout, themedConfig)
      .then((plot) => {
        const target = plot as unknown as {
          removeAllListeners?: (event: string) => void;
          on?: (event: string, handler: (event: { points?: PlotPoint[] }) => void) => void;
        };
        target.removeAllListeners?.('plotly_click');
        target.on?.('plotly_click', (event) => {
          const point = event.points?.[0];
          if (point) clickRef.current?.(point);
        });
      })
      .catch(() => undefined);
    // `isDark` refaz o tema quando o usuário alterna claro/escuro.
  }, [data, layout, config, height, exportName, isDark, expanded]);

  const getImage = async (): Promise<ChartImage | null> => {
    const container = containerRef.current;
    if (!container) return null;
    const width = Math.max(1, Math.round(container.clientWidth));
    const url = await Plotly.toImage(container, { format: 'svg', width, height });
    // O Plotly devolve "data:image/svg+xml,<svg url-encoded>".
    const svg = decodeURIComponent(url.slice(url.indexOf(',') + 1));
    return { svg: resolveCssVariables(svg), width, height };
  };

  return (
    <div className={cn('w-full space-y-2', className)}>
      <div className="flex justify-end gap-1.5">
        {!expanded && (
          <ExpandChartButton>
            <PlotlyChart {...props} height={Math.max(height, expandedHeight())} expanded />
          </ExpandChartButton>
        )}
        <ExportImageButton filename={exportName} getImage={getImage} />
      </div>
      <div
        ref={containerRef}
        className={cn('w-full', onPointClick && '[&_.cursor-crosshair]:cursor-pointer [&_.nsewdrag]:cursor-pointer')}
      />
    </div>
  );
}

/** Acompanha a classe `dark` do `<html>`, para os gráficos seguirem a troca de tema. */
function useIsDark(): boolean {
  const [isDark, setIsDark] = useState(() =>
    typeof document === 'undefined' ? true : document.documentElement.classList.contains('dark'),
  );
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains('dark')),
    );
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);
  return isDark;
}
