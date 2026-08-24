import { useEffect, useMemo, useRef, useState } from 'react';
import cloud from 'd3-cloud';
import { Download, FileImage } from 'lucide-react';

import type { WordFrequency } from '@/core/wordcloud';
import { downloadBlob, timestampedFilename } from '@/core/export';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

/**
 * Paleta refinada da nuvem em tons contrastantes e elegantes.
 */
const PALETTE = [
  '#0284c7', // Sky 600
  '#2563eb', // Blue 600
  '#4f46e5', // Indigo 600
  '#7c3aed', // Violet 600
  '#0d9488', // Teal 600
  '#0891b2', // Cyan 600
  '#1d4ed8', // Blue 700
  '#3b82f6', // Blue 500
] as const;

const MIN_FONT = 13;
const MAX_FONT = 48;

type CloudWord = cloud.Word & { value: number; color: string };

interface PlacedWord {
  text: string;
  size: number;
  value: number;
  x: number;
  y: number;
  rotate: number;
  color: string;
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
}

export default function WordCloud({
  words,
  width = 900,
  height = 420,
  className,
  exportName = 'nuvem-de-palavras',
}: WordCloudProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const { locale } = useLocale();
  const isEn = locale === 'en';

  const [hovered, setHovered] = useState<HoverState | null>(null);
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

    const maxValue = Math.max(...topWords.map((word) => word.value));
    const minValue = Math.min(...topWords.map((word) => word.value));
    const range = Math.sqrt(maxValue) - Math.sqrt(minValue) || 1;

    return topWords.map((word, index) => ({
      text: word.text,
      value: word.value,
      size:
        MIN_FONT +
        ((Math.sqrt(word.value) - Math.sqrt(minValue)) / range) * (MAX_FONT - MIN_FONT),
      color: PALETTE[index % PALETTE.length] as string,
    }));
  }, [words]);

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

  const placed = layout?.source === scaled ? layout.placed : [];

  const exportSvg = (): void => {
    const svg = svgRef.current;
    if (!svg) return;

    const source = new XMLSerializer().serializeToString(svg);
    downloadBlob(
      timestampedFilename(exportName, 'svg'),
      new Blob([source], { type: 'image/svg+xml;charset=utf-8' }),
    );
  };

  const exportPng = (): void => {
    const svg = svgRef.current;
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      const scale = 2; // Alta resolução (DPI 2x)
      const canvas = document.createElement('canvas');
      canvas.width = width * scale;
      canvas.height = height * scale;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Fundo branco limpo para PNG
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);

      canvas.toBlob((blob) => {
        if (blob) {
          downloadBlob(timestampedFilename(exportName, 'png'), blob);
        }
      }, 'image/png');
    };
    img.src = url;
  };

  const handleWordMouseMove = (e: React.MouseEvent, word: PlacedWord) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setHovered({
      text: word.text,
      value: word.value,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={exportPng}
          disabled={placed.length === 0}
          className="gap-1.5 text-xs font-semibold"
        >
          <FileImage className="size-3.5 text-blue-600" />
          <span>{isEn ? 'Download PNG' : 'Baixar PNG'}</span>
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={exportSvg}
          disabled={placed.length === 0}
          className="gap-1.5 text-xs font-medium"
        >
          <Download className="size-3.5" />
          <span>{isEn ? 'Download SVG' : 'Baixar SVG'}</span>
        </Button>
      </div>

      <div
        ref={containerRef}
        className="relative overflow-x-auto rounded-xl border border-border/80 bg-card p-2 shadow-2xs"
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          role="img"
          aria-label={`Nuvem com ${placed.length} palavras mais frequentes`}
          xmlns="http://www.w3.org/2000/svg"
          className="mx-auto select-none"
        >
          <g transform={`translate(${width / 2},${height / 2})`}>
            {placed.map((word) => (
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
                className="cursor-pointer transition-opacity duration-150 hover:opacity-75"
                onMouseEnter={(e) => handleWordMouseMove(e, word)}
                onMouseMove={(e) => handleWordMouseMove(e, word)}
                onMouseLeave={() => setHovered(null)}
              >
                {word.text}
                <title>
                  {word.text}: {word.value}{' '}
                  {word.value === 1
                    ? isEn
                      ? 'occurrence'
                      : 'ocorrência'
                    : isEn
                      ? 'occurrences'
                      : 'ocorrências'}
                </title>
              </text>
            ))}
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
                {hovered.value.toLocaleString('pt-BR')}
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
