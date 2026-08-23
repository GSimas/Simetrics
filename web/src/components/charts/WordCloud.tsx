import { useEffect, useMemo, useRef, useState } from 'react';
import cloud from 'd3-cloud';

import type { WordFrequency } from '@/core/wordcloud';
import { downloadBlob, timestampedFilename } from '@/core/export';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

/**
 * Nuvem de palavras em SVG, com layout do `d3-cloud`.
 *
 * Substitui o `echarts-wordcloud`, cujo peer está preso ao ECharts 5 — versão com um
 * advisory de XSS em aberto. O `d3-cloud` é justamente o algoritmo de posicionamento em
 * que o plugin do ECharts se baseia, então o resultado visual é equivalente.
 *
 * Renderizar em SVG (e não em canvas) traz dois ganhos: o texto continua selecionável e
 * legível por leitor de tela, e a exportação sai vetorial.
 */

/** Paleta da nuvem — ⇄ a paleta padrão de utils.py:2696. */
const PALETTE = ['#0077b6', '#00b4d8', '#90e0ef', '#03045e', '#023e8a'] as const;

const MIN_FONT = 14;
const MAX_FONT = 80;

/** Palavra do layout do d3-cloud, acrescida dos campos nossos. */
type CloudWord = cloud.Word & { value: number; color: string };

interface PlacedWord {
  text: string;
  size: number;
  x: number;
  y: number;
  rotate: number;
  color: string;
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
  height = 480,
  className,
  exportName = 'nuvem-de-palavras',
}: WordCloudProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  // O resultado guarda a entrada que o gerou.
  //
  // O layout do d3-cloud é assíncrono, então entre a troca das palavras e o fim do novo
  // cálculo existe uma janela em que o estado ainda tem a nuvem anterior. Amarrar o
  // resultado à sua origem faz essa janela renderizar vazio em vez de mostrar palavras
  // que não pertencem mais ao recorte — e evita o `setState` síncrono dentro do efeito,
  // que dispararia renderizações em cascata.
  const [layout, setLayout] = useState<{ source: readonly CloudWord[]; placed: PlacedWord[] } | null>(
    null,
  );

  // Escala de fonte por raiz quadrada da frequência.
  //
  // Escala linear é a escolha errada aqui: a distribuição de frequências é fortemente
  // enviesada (lei de Zipf), e a palavra mais comum acabaria dezenas de vezes maior que
  // as demais, deixando o resto ilegível. A raiz comprime a cauda longa.
  const scaled = useMemo<CloudWord[]>(() => {
    if (words.length === 0) return [];

    const maxValue = Math.max(...words.map((word) => word.value));
    const minValue = Math.min(...words.map((word) => word.value));
    const range = Math.sqrt(maxValue) - Math.sqrt(minValue) || 1;

    return words.map((word, index) => ({
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

    const instance = cloud<CloudWord>()
      .size([width, height])
      .words(scaled.map((word) => ({ ...word })))
      .padding(3)
      // Rotação em passos de 45°, como a configuração do ECharts no Python.
      .rotate(() => (Math.random() < 0.65 ? 0 : Math.random() < 0.5 ? 45 : -45))
      .font('inherit')
      .fontSize((word) => word.size ?? MIN_FONT)
      .on('end', (output) => {
        if (cancelled) return;
        setLayout({
          source: scaled,
          placed: output.map((word) => ({
            text: String(word.text),
            size: word.size ?? MIN_FONT,
            x: word.x ?? 0,
            y: word.y ?? 0,
            rotate: word.rotate ?? 0,
            color: word.color,
          })),
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

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={exportSvg} disabled={placed.length === 0}>
          Baixar SVG
        </Button>
      </div>

      <div className="overflow-x-auto rounded-md border">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          role="img"
          aria-label={`Nuvem com as ${placed.length} palavras mais frequentes`}
          xmlns="http://www.w3.org/2000/svg"
        >
          <g transform={`translate(${width / 2},${height / 2})`}>
            {placed.map((word) => (
              <text
                key={`${word.text}-${word.x}-${word.y}`}
                textAnchor="middle"
                transform={`translate(${word.x},${word.y}) rotate(${word.rotate})`}
                style={{ fontSize: word.size, fontWeight: 700, fill: word.color }}
              >
                {word.text}
              </text>
            ))}
          </g>
        </svg>
      </div>
    </div>
  );
}
