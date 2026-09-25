import { lazy, Suspense, useState } from 'react';

import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CONTINENTS, continentOf, type Continent } from '@/core/continents';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import type { Dataset } from '@/lib/types';
import { useAsyncResult } from '@/lib/use-async-result';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { PALETTE, chartMessage } from '@/features/overview/viz-shared';
import { openInSearch } from '@/state/navigation.store';

const WorldMap = lazy(() => import('@/components/charts/WorldMap'));
const RadialGraph = lazy(() => import('@/components/charts/RadialGraph'));

export interface CollaborationPanelProps {
  dataset: Dataset;
}

export function CollaborationPanel({ dataset }: CollaborationPanelProps) {
  const [topN, setTopN] = useState(30);
  const [view, setView] = useState<'mapa' | 'radial'>('mapa');
  const t = useLocale((state) => state.t);
  const en = useLocale((state) => state.locale === 'en');

  const { data: network } = useAsyncResult<CollaborationNetwork>(`collab ${topN}`, () =>
    getAnalyticsWorker().collaboration(dataset, topN),
  );

  return (
    <div className="space-y-4">
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
          chartMessage(en ? 'Computing the collaboration network…' : 'Calculando rede de colaboração…')
        ) : network.nodes.length === 0 ? (
          chartMessage(
            en
              ? 'The dataset has no country information. That depends on the affiliation field, ' +
                  'which not every export includes.'
              : 'A base não traz informação de país. Isso depende do campo de afiliação, que ' +
                  'nem toda exportação inclui.',
          )
        ) : (
          <Tabs value={view} onValueChange={(value) => setView(value as 'mapa' | 'radial')}>
            <TabsList className="w-full justify-start">
              <TabsTrigger value="mapa">{t('network_map_tab')}</TabsTrigger>
              <TabsTrigger value="radial">{t('network_radial_tab')}</TabsTrigger>
            </TabsList>

            <TabsContent value="mapa">
              <GeoView network={network} />
            </TabsContent>
            <TabsContent value="radial">
              <RadialView network={network} />
            </TabsContent>
          </Tabs>
        )}
    </div>
  );
}

/**
 * Mapa-múndi em SVG (componente WorldMap): primeiro clique num país destaca ele e suas
 * colaborações; o segundo abre o perfil do país no Motor de Busca.
 */
function GeoView({ network }: { network: CollaborationNetwork }) {
  const en = useLocale((state) => state.locale === 'en');
  const labelOf = new Map(network.nodes.map((node) => [node.country, node.label]));
  return (
    <Suspense fallback={chartMessage(en ? 'Loading map…' : 'Carregando mapa…')}>
      <WorldMap
        nodes={network.nodes.map((node) => ({
          key: node.country,
          label: node.label,
          documents: node.documents,
          latitude: node.latitude,
          longitude: node.longitude,
        }))}
        edges={network.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          documents: edge.documents,
        }))}
        onOpenProfile={(key) => openInSearch(labelOf.get(key) ?? key, ['País'])}
        exportName={en ? 'international-collaboration' : 'colaboracao-internacional'}
      />
    </Suspense>
  );
}

const CONTINENT_KEYS = {
  Africa: 'continent_Africa',
  'North America': 'continent_North_America',
  'South America': 'continent_South_America',
  Asia: 'continent_Asia',
  Europe: 'continent_Europe',
  Oceania: 'continent_Oceania',
} as const satisfies Record<Continent, string>;

/**
 * Grafo radial (diagrama de cordas): cada país num ponto do círculo, agrupado e colorido
 * por continente, do que mais publica para o que menos dentro de cada grupo; as
 * colaborações são cordas — mais grossas quanto mais documentos em comum.
 */
function RadialView({ network }: { network: CollaborationNetwork }) {
  const t = useLocale((state) => state.t);
  const en = useLocale((state) => state.locale === 'en');
  const labelOf = new Map(network.nodes.map((node) => [node.country, node.label]));
  // Índice do continente = grupo no círculo e cor; país sem continente conhecido fica no fim.
  const groupOf = (country: string): number => {
    const continent = continentOf(country);
    return continent ? CONTINENTS.indexOf(continent) : CONTINENTS.length;
  };
  // Sem continente: o cinza, último tom da paleta.
  const colorOf = (group: number): string => (group < CONTINENTS.length && PALETTE[group]) || PALETTE[7];
  const groups = [...new Set(network.nodes.map((node) => groupOf(node.country)))].sort((a, b) => a - b);

  return (
    <Suspense fallback={chartMessage(en ? 'Loading graph…' : 'Carregando grafo…')}>
      <RadialGraph
        nodes={network.nodes.map((node) => ({
          key: node.country,
          label: node.label,
          weight: node.documents,
          group: groupOf(node.country),
          color: colorOf(groupOf(node.country)),
        }))}
        edges={network.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          weight: edge.documents,
        }))}
        weightLabel={t('radial_documents')}
        legend={groups.map((group) => {
          const continent = CONTINENTS[group];
          return {
            label: t(continent ? CONTINENT_KEYS[continent] : 'continent_unknown'),
            color: colorOf(group),
          };
        })}
        onNodeClick={(key) => openInSearch(labelOf.get(key) ?? key, ['País'])}
        exportName={en ? 'radial-collaboration' : 'colaboracao-radial'}
      />
    </Suspense>
  );
}
