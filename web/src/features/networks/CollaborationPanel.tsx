import { lazy, Suspense, useState } from 'react';

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { circularPositions, type CollaborationNetwork } from '@/core/viz/collaboration';
import type { Dataset } from '@/lib/types';
import { useAsyncResult } from '@/lib/use-async-result';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { PALETTE, chartMessage } from '@/features/overview/viz-shared';

const PlotlyChart = lazy(() => import('@/components/charts/PlotlyChart'));

const MIN_EDGE_WIDTH = 1;
const MAX_EDGE_WIDTH = 6;

export interface CollaborationPanelProps {
  dataset: Dataset;
}

export function CollaborationPanel({ dataset }: CollaborationPanelProps) {
  const [topN, setTopN] = useState(30);
  const [view, setView] = useState<'mapa' | 'circular'>('mapa');
  const t = useLocale((state) => state.t);

  const { data: network } = useAsyncResult<CollaborationNetwork>(`collab ${topN}`, () =>
    getAnalyticsWorker().collaboration(dataset, topN),
  );

  return (
    <Card className="border-t-4 border-t-indigo-500 shadow-xs">
      <CardHeader>
        <CardTitle className="text-base font-bold text-foreground">{t('network_collab_title')}</CardTitle>
        <CardDescription>
          {t('network_collab_desc')}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="collab-top">{t('network_top_label')}</Label>
            <Select value={String(topN)} onValueChange={(value) => setTopN(Number(value))}>
              <SelectTrigger id="collab-top">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[15, 20, 30, 40, 50].map((option) => (
                  <SelectItem key={option} value={String(option)}>
                    Top {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {!network ? (
          chartMessage('Calculando rede de colaboração…')
        ) : network.nodes.length === 0 ? (
          chartMessage(
            'A base não traz informação de país. Isso depende do campo de afiliação, que ' +
              'nem toda exportação inclui.',
          )
        ) : (
          <Tabs value={view} onValueChange={(value) => setView(value as 'mapa' | 'circular')}>
            <TabsList className="bg-slate-100 dark:bg-slate-800/80 p-1">
              <TabsTrigger value="mapa">{t('network_map_tab')}</TabsTrigger>
              <TabsTrigger value="circular">{t('network_circular_tab')}</TabsTrigger>
            </TabsList>

            <TabsContent value="mapa">
              <GeoView network={network} />
            </TabsContent>
            <TabsContent value="circular">
              <CircularView network={network} />
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
}

/** Escala a espessura da aresta entre os limites visuais. */
function edgeWidth(documents: number, min: number, max: number): number {
  if (max === min) return (MIN_EDGE_WIDTH + MAX_EDGE_WIDTH) / 2;
  return (
    MIN_EDGE_WIDTH + ((documents - min) / (max - min)) * (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH)
  );
}

function GeoView({ network }: { network: CollaborationNetwork }) {
  const byCountry = new Map(network.nodes.map((node) => [node.country, node]));
  const weights = network.edges.map((edge) => edge.documents);
  const minWeight = weights.length > 0 ? Math.min(...weights) : 0;
  const maxWeight = weights.length > 0 ? Math.max(...weights) : 0;

  const traces: Record<string, unknown>[] = [
    {
      type: 'choropleth',
      locationmode: 'country names',
      locations: network.nodes.map((node) => node.plotlyName),
      z: network.nodes.map((node) => node.documents),
      text: network.nodes.map((node) => {
        const partners = node.partners
          .slice(0, 8)
          .map((partner) => `  • ${partner.country}: ${partner.documents}`)
          .join('<br>');
        return (
          `<b>${node.label}</b><br>Documentos: ${node.documents}<br>` +
          `<b>Principais parceiros:</b><br>${partners || '  (sem colaborações diretas)'}`
        );
      }),
      hoverinfo: 'text',
      colorscale: 'Teal',
      showscale: false,
      marker: { line: { color: 'rgba(255,255,255,0.6)', width: 0.5 } },
    },
  ];

  for (const edge of network.edges) {
    const source = byCountry.get(edge.source);
    const target = byCountry.get(edge.target);
    // Sem coordenada não há como traçar o arco; o país continua pintado no choropleth.
    if (!source?.latitude || !target?.latitude) continue;

    traces.push({
      type: 'scattergeo',
      mode: 'lines',
      lat: [source.latitude, target.latitude],
      lon: [source.longitude, target.longitude],
      line: { width: edgeWidth(edge.documents, minWeight, maxWeight), color: 'rgba(232,115,74,0.5)' },
      hoverinfo: 'text',
      text: `${edge.sourceLabel} ↔ ${edge.targetLabel}: ${edge.documents} documentos`,
      showlegend: false,
    });
  }

  return (
    <Suspense fallback={chartMessage('Carregando mapa…')}>
      <PlotlyChart
        exportName="colaboracao-internacional"
        height={520}
        data={traces as never}
        layout={{
          showlegend: false,
          geo: {
            projection: { type: 'natural earth' },
            showland: true,
            landcolor: 'rgba(200,206,212,0.35)',
            coastlinecolor: 'rgba(150,160,170,0.6)',
            showframe: false,
            bgcolor: 'rgba(0,0,0,0)',
          },
          margin: { l: 0, r: 0, t: 10, b: 0 },
        }}
      />
    </Suspense>
  );
}

function CircularView({ network }: { network: CollaborationNetwork }) {
  const positions = circularPositions(network.nodes);
  const weights = network.edges.map((edge) => edge.documents);
  const minWeight = weights.length > 0 ? Math.min(...weights) : 0;
  const maxWeight = weights.length > 0 ? Math.max(...weights) : 0;

  const traces: Record<string, unknown>[] = network.edges.map((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);

    return {
      type: 'scatter',
      mode: 'lines',
      x: [source?.x ?? 0, target?.x ?? 0],
      y: [source?.y ?? 0, target?.y ?? 0],
      line: {
        width: edgeWidth(edge.documents, minWeight, maxWeight),
        color: 'rgba(18,115,185,0.28)',
      },
      hoverinfo: 'text',
      text: `${edge.sourceLabel} ↔ ${edge.targetLabel}: ${edge.documents} documentos`,
      showlegend: false,
    };
  });

  const maxDocuments = Math.max(...network.nodes.map((node) => node.documents), 1);

  traces.push({
    type: 'scatter',
    mode: 'text+markers',
    x: network.nodes.map((node) => positions.get(node.country)?.x ?? 0),
    y: network.nodes.map((node) => positions.get(node.country)?.y ?? 0),
    text: network.nodes.map((node) => node.label),
    // Rótulos para fora do círculo, para não colidirem com as arestas do centro.
    textposition: network.nodes.map((node) => {
      const position = positions.get(node.country);
      return `${(position?.y ?? 0) >= 0 ? 'top' : 'bottom'} ${
        (position?.x ?? 0) >= 0 ? 'right' : 'left'
      }`;
    }),
    textfont: { size: 10 },
    marker: {
      size: network.nodes.map(
        (node) => 10 + (node.documents / maxDocuments) * 30,
      ),
      color: PALETTE[0],
      opacity: 0.85,
      line: { width: 1, color: 'white' },
    },
    customdata: network.nodes.map((node) => node.documents) as never,
    hovertemplate: '<b>%{text}</b><br>%{customdata} documentos<extra></extra>',
    showlegend: false,
  });

  return (
    <Suspense fallback={chartMessage('Carregando grafo…')}>
      <PlotlyChart
        exportName="colaboracao-circular"
        height={560}
        data={traces as never}
        layout={{
          showlegend: false,
          xaxis: { visible: false, range: [-1.35, 1.35] },
          // Escala travada ao eixo X para o círculo não virar elipse ao redimensionar.
          yaxis: { visible: false, range: [-1.35, 1.35], scaleanchor: 'x', scaleratio: 1 },
          margin: { l: 10, r: 10, t: 10, b: 10 },
        }}
      />
    </Suspense>
  );
}
