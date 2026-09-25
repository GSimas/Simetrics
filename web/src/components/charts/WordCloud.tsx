import { useEffect, useMemo, useRef, useState } from 'react';
import cloud from 'd3-cloud';

import type { WordFrequency } from '@/core/wordcloud';
import { ExpandChartButton, expandedHeight } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import { imageFromSvgElement } from '@/lib/export-image';
import { numberLocale } from '@/lib/i18n/labels';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Paleta da nuvem: tons médios da família Scientata, legíveis sobre tinta e papel.
 */
const PALETTE = [
  '#3FAE8F', // pinho claro
  '#8FCE2A', // sinal
  '#2FBAB3', // ciano
  '#E56D45', // laranja Scientata
  '#6A7DFF', // índigo Simetrics
  '#5E9E88', // musgo
  '#C9A13A', // âmbar
  '#8F9B95', // sálvia
] as const;

const MIN_FONT = 13;
const MAX_FONT = 48;

// O d3-cloud preenche x0/x1/y0/y1 (limites do sprite) no layout, mas os tipos não os declaram.
type CloudWord = cloud.Word & {
  value: number;
  color: string;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
};

interface PlacedWord {
  text: string;
  size: number;
  value: number;
  x: number;
  y: number;
  rotate: number;
  color: string;
  /** Limites do sprite relativos ao centro da palavra, devolvidos pelo d3-cloud. */
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** Ampliação máxima ao enquadrar — evita que uma palavra solitária vire um cartaz. */
const MAX_ZOOM = 4;
const FIT_PADDING = 12;

/**
 * Recorta o viewBox à área que as palavras ocupam, para a nuvem preencher o quadro
 * inteiro seja qual for o número de termos. Mantém a proporção do quadro para o
 * `preserveAspectRatio` não distorcer nada.
 */
function fitViewBox(placed: readonly PlacedWord[], width: number, height: number): string {
  if (placed.length === 0) return `${-width / 2} ${-height / 2} ${width} ${height}`;

  const minX = Math.min(...placed.map((word) => word.x + word.x0)) - FIT_PADDING;
  const maxX = Math.max(...placed.map((word) => word.x + word.x1)) + FIT_PADDING;
  const minY = Math.min(...placed.map((word) => word.y + word.y0)) - FIT_PADDING;
  const maxY = Math.max(...placed.map((word) => word.y + word.y1)) + FIT_PADDING;

  const zoom = Math.min(MAX_ZOOM, width / (maxX - minX), height / (maxY - minY));
  const boxWidth = width / zoom;
  const boxHeight = height / zoom;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  return `${centerX - boxWidth / 2} ${centerY - boxHeight / 2} ${boxWidth} ${boxHeight}`;
}

interface HoverState {
  text: string;
  value: number;
  x: number;
  y: number;
}

export interface WordCloudProps {
  words: readonly WordFrequency[];
  width?: number;
  height?: number;
  className?: string;
  exportName?: string;
  /** Clique numa palavra. Só recebe as palavras para as quais `isClickable` é verdadeiro. */
  onWordClick?: (text: string) => void;
  isClickable?: (text: string) => boolean;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

function WordCloud(props: WordCloudProps) {
  const {
    words,
    width: fallbackWidth = 900,
    height = 420,
    className,
    exportName: exportNameProp,
    onWordClick,
    isClickable,
    expanded,
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { locale } = useLocale();
  const isEn = locale === 'en';
  const exportName = exportNameProp ?? (isEn ? 'word-cloud' : 'nuvem-de-palavras');

  const [hovered, setHovered] = useState<HoverState | null>(null);
  // O layout usa a largura real do quadro: com uma largura fixa, numa coluna estreita o
  // SVG encolheria tudo para caber e as palavras ficariam minúsculas.
  const [measuredWidth, setMeasuredWidth] = useState<number | null>(null);
  const width = measuredWidth ?? fallbackWidth;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = Math.round(entry?.contentRect.width ?? 0);
      if (next > 0) setMeasuredWidth((previous) => (previous === next ? previous : next));
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);
  const [layout, setLayout] = useState<{ source: readonly CloudWord[]; placed: PlacedWord[] } | null>(
    null,
  );

  // Escala de fonte suavizada e palavras ordenadas por relevância
  const scaled = useMemo<CloudWord[]>(() => {
    if (words.length === 0) return [];

    // Limita às 80 principais palavras para evitar saturação e sobreposição
    const topWords = [...words]
      .sort((a, b) => b.value - a.value)
      .slice(0, 80);

    // Escala relativa ao mínimo e máximo desta nuvem, não da base inteira. Com todas as
    // frequências iguais não há o que ordenar: todas ficam no meio da escala, e o
    // enquadramento (fitViewBox) as amplia até ocupar o quadro.
    // A faixa de fontes acompanha a área do quadro: num quadro estreito, fontes de
    // tamanho fixo não cabem e o d3-cloud descarta as palavras que sobram. O
    // enquadramento (fitViewBox) amplia o resultado depois.
    const areaScale = Math.min(1, Math.sqrt((width * height) / (900 * 420)));
    const minFont = Math.max(9, MIN_FONT * areaScale);
    const maxFont = Math.max(minFont + 8, MAX_FONT * areaScale);

    const maxValue = Math.max(...topWords.map((word) => word.value));
    const minValue = Math.min(...topWords.map((word) => word.value));
    const range = Math.sqrt(maxValue) - Math.sqrt(minValue);

    return topWords.map((word, index) => ({
      text: word.text,
      value: word.value,
      size:
        range === 0
          ? (minFont + maxFont) / 2
          : minFont + ((Math.sqrt(word.value) - Math.sqrt(minValue)) / range) * (maxFont - minFont),
      color: PALETTE[index % PALETTE.length] as string,
    }));
  }, [words, width, height]);

  useEffect(() => {
    if (scaled.length === 0) return;

    let cancelled = false;

    // Layout com orientação horizontal limpa e padding generoso para evitar sobreposição
    const instance = cloud<CloudWord>()
      .size([width, height])
      .words(scaled.map((word) => ({ ...word })))
      .padding(6)
      // Rotação estritamente horizontal para máxima legibilidade e organização
      .rotate(() => 0)
      .font('system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif')
      .fontSize((word) => word.size ?? MIN_FONT)
      .spiral('archimedean')
      // Posiciona em fatias de 12 ms: sem isso o layout inteiro roda de uma vez e
      // congela a troca de aba que monta a nuvem.
      .timeInterval(12)
      .on('end', (output) => {
        if (cancelled) return;
        setLayout({
          source: scaled,
          placed: output.map((word) => {
            const raw = word as CloudWord;
            return {
              text: String(word.text),
              size: word.size ?? MIN_FONT,
              value: raw.value ?? 1,
              x: word.x ?? 0,
              y: word.y ?? 0,
              rotate: word.rotate ?? 0,
              color: raw.color ?? PALETTE[0],
              x0: raw.x0 ?? 0,
              x1: raw.x1 ?? 0,
              y0: raw.y0 ?? 0,
              y1: raw.y1 ?? 0,
            };
          }),
        });
      });

    instance.start();

    return () => {
      cancelled = true;
      instance.stop();
    };
  }, [scaled, width, height]);

  const placed = useMemo(
    () => (layout?.source === scaled ? layout.placed : []),
    [layout, scaled],
  );
  const viewBox = useMemo(() => fitViewBox(placed, width, height), [placed, width, height]);

  const handleWordMouseMove = (e: React.MouseEvent, word: PlacedWord) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setHovered({
      text: word.text,
      value: word.value,
      // O tooltip é centrado no cursor; perto das bordas seria cortado pelo quadro.
      x: Math.min(Math.max(e.clientX - rect.left, 90), rect.width - 90),
      y: e.clientY - rect.top,
    });
  };

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex justify-end gap-1.5">
        {!expanded && (
          <ExpandChartButton>
            <WordCloud {...props} height={Math.max(height, expandedHeight())} expanded />
          </ExpandChartButton>
        )}
        <ExportImageButton
          filename={exportName}
          getImage={() => (svgRef.current ? imageFromSvgElement(svgRef.current) : null)}
        />
      </div>

      <div
        ref={containerRef}
        className="relative overflow-x-auto rounded-xl border border-border/80 bg-card p-2 shadow-2xs"
      >
        <svg
          ref={svgRef}
          viewBox={viewBox}
          width="100%"
          height={height}
          role="img"
          aria-label={
            isEn
              ? `Cloud of the ${placed.length} most frequent words`
              : `Nuvem com ${placed.length} palavras mais frequentes`
          }
          xmlns="http://www.w3.org/2000/svg"
          className="mx-auto select-none"
        >
          <g>
            {placed.map((word) => {
              const clickable = Boolean(onWordClick) && (isClickable?.(word.text) ?? true);
              return (
              <text
                key={`${word.text}-${word.x}-${word.y}`}
                textAnchor="middle"
                transform={`translate(${word.x},${word.y})`}
                style={{
                  fontSize: word.size,
                  fontWeight: 600,
                  fill: word.color,
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                }}
                className={cn(
                  'transition-opacity duration-150 hover:opacity-75',
                  clickable ? 'cursor-pointer' : 'cursor-default',
                )}
                onClick={clickable ? () => onWordClick?.(word.text) : undefined}
                onMouseEnter={(e) => handleWordMouseMove(e, word)}
                onMouseMove={(e) => handleWordMouseMove(e, word)}
                onMouseLeave={() => setHovered(null)}
              >
                {word.text}
              </text>
              );
            })}
          </g>
        </svg>

        {/* Tooltip flutuante interativo */}
        {hovered && (
          <div
            className="pointer-events-none absolute z-50 -translate-x-1/2 -translate-y-full rounded-lg border border-border/80 bg-popover/95 px-3 py-1.5 text-xs font-medium text-popover-foreground shadow-lg backdrop-blur-xs transition-transform animate-in fade-in-0 zoom-in-95"
            style={{
              left: `${hovered.x}px`,
              top: `${Math.max(10, hovered.y - 12)}px`,
            }}
          >
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-primary">&ldquo;{hovered.text}&rdquo;:</span>
              <span className="tabular-nums font-semibold">
                {hovered.value.toLocaleString(numberLocale(locale))}
              </span>
              <span className="text-muted-foreground">
                {hovered.value === 1
                  ? isEn
                    ? 'occurrence'
                    : 'ocorrência'
                  : isEn
                    ? 'occurrences'
                    : 'ocorrências'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(WordCloud);
