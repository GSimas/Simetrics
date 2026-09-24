import { useMemo, useRef, useState } from 'react';
import { geoGraticule10, geoNaturalEarth1, geoPath, type GeoPermissibleObjects } from 'd3-geo';
import { feature } from 'topojson-client';
import type { Feature, FeatureCollection, Geometry } from 'geojson';
import type { GeometryCollection, Topology } from 'topojson-specification';
import world from 'world-atlas/countries-110m.json';

import { ExpandChartButton } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import { useSvgZoom, ZoomControls } from '@/components/charts/svg-zoom';
import { imageFromSvgElement } from '@/lib/export-image';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Mapa-múndi de colaboração em SVG (d3-geo + world-atlas 1:110m).
 *
 * Países preenchidos pela produção, colaborações como arcos curvos entre os países e um
 * marcador por país. Primeiro clique num país fixa o destaque — ele, os parceiros e as
 * linhas entre eles —; segundo clique no mesmo país chama `onOpenProfile`. Clicar no
 * oceano desfaz o destaque. Roda do mouse ou botões aproximam; arrastar move o mapa.
 */
export interface WorldMapNode {
  key: string;
  label: string;
  documents: number;
  latitude: number | null;
  longitude: number | null;
}

export interface WorldMapEdge {
  source: string;
  target: string;
  documents: number;
}

export interface WorldMapProps {
  nodes: readonly WorldMapNode[];
  edges: readonly WorldMapEdge[];
  /** País destacado ao abrir (perfil de país no Motor de Busca). */
  focus?: string | undefined;
  onOpenProfile?: (key: string) => void;
  exportName?: string;
  className?: string;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

const WIDTH = 960;
const HEIGHT = 500;

type CountryFeature = Feature<Geometry, { name: string }>;

// A geometria não muda: decodifica o TopoJSON uma única vez por sessão.
const COUNTRIES = (
  feature(
    world as unknown as Topology,
    (world as unknown as Topology).objects.countries as GeometryCollection<{ name: string }>,
  ) as FeatureCollection<Geometry, { name: string }>
).features as CountryFeature[];

const projection = geoNaturalEarth1().fitExtent(
  [
    [8, 8],
    [WIDTH - 8, HEIGHT - 8],
  ],
  { type: 'Sphere' },
);
const path = geoPath(projection);
const SPHERE = path({ type: 'Sphere' }) ?? '';
const GRATICULE = path(geoGraticule10()) ?? '';
const COUNTRY_PATHS = COUNTRIES.map((country) => ({
  name: country.properties.name,
  d: path(country as GeoPermissibleObjects) ?? '',
}));

/**
 * Nome como vem das bases → nome no world-atlas. Só as divergências; o resto casa
 * ignorando maiúsculas.
 */
const ALIASES: Record<string, string> = {
  usa: 'united states of america',
  'united states': 'united states of america',
  us: 'united states of america',
  england: 'united kingdom',
  scotland: 'united kingdom',
  wales: 'united kingdom',
  'northern ireland': 'united kingdom',
  uk: 'united kingdom',
  'czech republic': 'czechia',
  'south korea': 'south korea',
  'korea, republic of': 'south korea',
  'republic of korea': 'south korea',
  'dominican republic': 'dominican rep.',
  'bosnia and herzegovina': 'bosnia and herz.',
  'democratic republic of the congo': 'dem. rep. congo',
  'central african republic': 'central african rep.',
  'ivory coast': "côte d'ivoire",
  'cote d’ivoire': "côte d'ivoire",
  "cote d'ivoire": "côte d'ivoire",
  'north macedonia': 'macedonia',
  'russian federation': 'russia',
  'viet nam': 'vietnam',
  'iran, islamic republic of': 'iran',
  'syrian arab republic': 'syria',
  'south sudan': 's. sudan',
  'equatorial guinea': 'eq. guinea',
  'solomon islands': 'solomon is.',
  'east timor': 'timor-leste',
};

const normalize = (name: string): string => {
  const lower = name.trim().toLowerCase();
  return ALIASES[lower] ?? lower;
};

function WorldMap(props: WorldMapProps) {
  const { nodes, edges, focus, onOpenProfile, exportName = 'mapa-mundi', className, expanded } = props;
  const t = useLocale((state) => state.t);
  const svgRef = useRef<SVGSVGElement>(null);
  const zoom = useSvgZoom(svgRef, WIDTH, HEIGHT);
  const [hovered, setHovered] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string | null>(focus ?? null);

  const data = useMemo(() => {
    const byName = new Map<string, WorldMapNode>();
    for (const node of nodes) {
      byName.set(normalize(node.label), node);
      byName.set(normalize(node.key), node);
    }
    const maxDocuments = Math.max(...nodes.map((node) => node.documents), 1);
    const maxEdge = Math.max(...edges.map((edge) => edge.documents), 1);

    const points = new Map(
      nodes
        .filter((node) => node.latitude !== null && node.longitude !== null)
        .map((node) => {
          const [x, y] = projection([node.longitude!, node.latitude!]) ?? [0, 0];
          return [node.key, { node, x, y, r: 2.5 + Math.sqrt(node.documents / maxDocuments) * 9 }];
        }),
    );

    const arcs = edges.flatMap((edge) => {
      const from = points.get(edge.source);
      const to = points.get(edge.target);
      if (!from || !to) return [];
      // Curva no plano do mapa, arqueada para cima e proporcional à distância. O arco de
      // grande círculo cruzaria a borda da projeção (China ↔ EUA pelo Ártico) e sairia
      // partido em dois traços.
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const distance = Math.hypot(dx, dy);
      let nx = -dy / (distance || 1);
      let ny = dx / (distance || 1);
      if (ny > 0) {
        nx = -nx;
        ny = -ny;
      }
      const bend = distance * 0.22;
      const cx = (from.x + to.x) / 2 + nx * bend;
      const cy = (from.y + to.y) / 2 + ny * bend;
      const d = `M${from.x.toFixed(1)},${from.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${to.x.toFixed(1)},${to.y.toFixed(1)}`;
      return [{ ...edge, d, width: 0.6 + (edge.documents / maxEdge) * 3.4 }];
    });

    return { byName, points, arcs, maxDocuments };
  }, [nodes, edges]);

  const selected = chosen !== null && nodes.some((node) => node.key === chosen) ? chosen : null;
  const active = selected ?? hovered;

  const partners = useMemo(() => {
    if (!active) return null;
    const keys = new Set([active]);
    for (const edge of edges) {
      if (edge.source === active) keys.add(edge.target);
      if (edge.target === active) keys.add(edge.source);
    }
    return keys;
  }, [active, edges]);

  const handleClick = (key: string): void => {
    if (selected === key) onOpenProfile?.(key);
    else setChosen(key);
  };

  const activeNode = active ? nodes.find((node) => node.key === active) : undefined;

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex justify-end gap-1.5">
        {!expanded && (
          <ExpandChartButton>
            <WorldMap {...props} focus={selected ?? focus} expanded />
          </ExpandChartButton>
        )}
        <ExportImageButton
          filename={exportName}
          getImage={() => (svgRef.current ? imageFromSvgElement(svgRef.current) : null)}
        />
      </div>

      <div className="relative overflow-hidden border border-border">
        {activeNode && (
          <div className="pointer-events-none absolute left-3 top-3 z-10 max-w-xs border border-border bg-popover/95 p-3 text-xs shadow-lg animate-in fade-in-0">
            <p className="mb-1 font-semibold">{activeNode.label}</p>
            <p className="text-muted-foreground">
              <span className="tabular-nums text-foreground">{activeNode.documents.toLocaleString('pt-BR')}</span>{' '}
              {t('radial_documents')} ·{' '}
              <span className="tabular-nums text-foreground">{(partners?.size ?? 1) - 1}</span> {t('map_partners')}
            </p>
            {onOpenProfile && (
              <p className="eyebrow mt-2 text-highlight">
                {selected === activeNode.key ? t('map_click_open') : t('map_click_highlight')}
              </p>
            )}
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className={cn('block w-full select-none', expanded ? 'h-[74dvh]' : 'h-auto')}
          role="img"
          aria-label={t('map_aria').replace('{count}', String(nodes.length))}
          xmlns="http://www.w3.org/2000/svg"
          {...zoom.svgProps}
        >
          {/* Oceano: clicar nele desfaz o destaque. */}
          <g style={zoom.style}>
          <path d={SPHERE} fill="var(--card)" stroke="var(--border)" strokeWidth={1 / zoom.k} onClick={() => setChosen(null)} />
          <path d={GRATICULE} fill="none" stroke="var(--border)" strokeOpacity={0.5} strokeWidth={0.5 / zoom.k} />

          <g>
            {COUNTRY_PATHS.map((country, index) => {
              const node = data.byName.get(country.name.toLowerCase());
              const related = !partners || (node !== undefined && partners.has(node.key));
              const intensity = node ? 0.25 + 0.7 * (Math.log1p(node.documents) / Math.log1p(data.maxDocuments)) : 0;
              return (
                <path
                  key={`${country.name}-${index}`}
                  d={country.d}
                  fill={node ? 'var(--highlight)' : 'var(--muted)'}
                  fillOpacity={node ? (related ? intensity : 0.12) : partners ? 0.5 : 1}
                  stroke="var(--background)"
                  strokeWidth={0.5 / zoom.k}
                  className={cn('transition-[fill-opacity] duration-300', node && onOpenProfile && 'cursor-pointer')}
                  onMouseEnter={node ? () => setHovered(node.key) : undefined}
                  onMouseLeave={node ? () => setHovered(null) : undefined}
                  onClick={node ? () => handleClick(node.key) : () => setChosen(null)}
                />
              );
            })}
          </g>

          <g fill="none" pointerEvents="none">
            {data.arcs.map((arc) => {
              const on = !active || arc.source === active || arc.target === active;
              return (
                <path
                  key={`${arc.source}→${arc.target}`}
                  d={arc.d}
                  stroke="var(--cyan)"
                  strokeWidth={arc.width / zoom.k}
                  strokeLinecap="round"
                  strokeOpacity={active ? (on ? 0.9 : 0.04) : 0.35}
                  className="transition-[stroke-opacity] duration-300"
                />
              );
            })}
          </g>

          <g>
            {[...data.points.values()].map(({ node, x, y, r }) => {
              const related = !partners || partners.has(node.key);
              return (
                <circle
                  key={node.key}
                  cx={x}
                  cy={y}
                  // Marcadores e traços crescem menos que o mapa: com zoom alto cobririam
                  // os países vizinhos.
                  r={r / Math.sqrt(zoom.k)}
                  fill="var(--highlight)"
                  fillOpacity={related ? 0.9 : 0.15}
                  stroke={selected === node.key ? 'var(--foreground)' : 'var(--background)'}
                  strokeWidth={(selected === node.key ? 2 : 1) / zoom.k}
                  className={cn('transition-[fill-opacity] duration-300', onOpenProfile && 'cursor-pointer')}
                  onMouseEnter={() => setHovered(node.key)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => handleClick(node.key)}
                />
              );
            })}
          </g>
          </g>
        </svg>
        <ZoomControls zoom={zoom} />
      </div>
    </div>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(WorldMap);
