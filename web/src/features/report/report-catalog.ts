/**
 * Catálogo do relatório: o que pode entrar nele, em ordem de documento.
 *
 * Uma lista só para o seletor, a prévia e os dois geradores (PDF e DOCX): a ordem em
 * que os blocos aparecem na tela é a ordem em que saem no arquivo.
 */

export type ReportSectionId =
  | 'summary'
  | 'kpis'
  | 'authors'
  | 'countries'
  | 'venues'
  | 'keywords'
  | 'themes'
  | 'networkTopology'
  | 'topDocuments';

export type ReportChartId =
  | 'production'
  | 'rankingAuthors'
  | 'rankingCountries'
  | 'worldMap'
  | 'collabChord'
  | 'rankingVenues'
  | 'wordCloud'
  | 'themesChart'
  | 'network'
  | 'networkChord'
  | 'rankingDocuments'
  | 'sankey'
  | 'boxplot'
  | 'genetics'
  | 'concept2d'
  | 'concept3d'
  | 'thematicMap'
  | 'historiograph'
  | 'lotka';

export type ReportItemId = ReportSectionId | ReportChartId;
export type ReportSelection = Record<ReportItemId, boolean>;

/** Gráfico já rasterizado para o documento (PNG em data URL, tamanho em pixels). */
export interface ReportImage {
  dataUrl: string;
  width: number;
  height: number;
}
export type ReportImages = Partial<Record<ReportChartId, ReportImage>>;

type Text = { pt: string; en: string };

export type ReportGroup = 'sections' | 'production' | 'geography' | 'lexicon' | 'networks' | 'advanced';

export const REPORT_GROUPS: Record<ReportGroup, Text> = {
  sections: { pt: 'Seções e tabelas', en: 'Sections and tables' },
  production: { pt: 'Produção e impacto', en: 'Production and impact' },
  geography: { pt: 'Geografia e colaboração', en: 'Geography and collaboration' },
  lexicon: { pt: 'Léxico e temas', en: 'Lexicon and themes' },
  networks: { pt: 'Redes', en: 'Networks' },
  advanced: { pt: 'Análises visuais avançadas', en: 'Advanced visual analyses' },
};

export interface ReportItem {
  id: ReportItemId;
  kind: 'section' | 'chart';
  group: ReportGroup;
  label: Text;
  /** Selecionado ao abrir a aba. */
  byDefault: boolean;
}

const section = (id: ReportSectionId, pt: string, en: string): ReportItem => ({
  id,
  kind: 'section',
  group: 'sections',
  label: { pt, en },
  byDefault: true,
});
const chart = (id: ReportChartId, group: ReportGroup, pt: string, en: string, byDefault = false): ReportItem => ({
  id,
  kind: 'chart',
  group,
  label: { pt, en },
  byDefault,
});

/** Ordem de documento. Os gráficos que já existiam no relatório abrem marcados. */
export const REPORT_ITEMS: readonly ReportItem[] = [
  section('summary', 'Resumo executivo e escopo', 'Executive summary and scope'),
  section('kpis', 'Indicadores cientométricos globais', 'Core scientometric indicators'),
  chart('production', 'production', 'Produção ao longo do tempo', 'Production over time', true),
  section('authors', 'Ranking de autores e produtividade', 'Author ranking and impact'),
  chart('rankingAuthors', 'production', 'Top 10 autores', 'Top 10 authors', true),
  section('countries', 'Geografia da produção', 'Geographic distribution'),
  chart('rankingCountries', 'geography', 'Top 10 países', 'Top 10 countries', true),
  chart('worldMap', 'geography', 'Mapa-múndi de colaboração', 'World collaboration map', true),
  chart('collabChord', 'geography', 'Colaboração entre países (grafo radial)', 'Collaboration between countries (radial graph)'),
  section('venues', 'Veículos de publicação', 'Publishing venues'),
  chart('rankingVenues', 'production', 'Top 10 venues', 'Top 10 venues'),
  section('keywords', 'Palavras-chave e lexicometria', 'Keywords and lexicometrics'),
  chart('wordCloud', 'lexicon', 'Nuvem de palavras-chave', 'Keyword cloud', true),
  section('themes', 'Estrutura temática por IA', 'AI thematic structure'),
  chart('themesChart', 'lexicon', 'Distribuição dos temas por IA', 'AI theme distribution', true),
  section('networkTopology', 'Topologia da rede', 'Network topology'),
  chart('network', 'networks', 'Rede de coocorrência (grafo)', 'Co-occurrence network (graph)', true),
  chart('networkChord', 'networks', 'Rede de coocorrência (grafo radial)', 'Co-occurrence network (radial graph)'),
  section('topDocuments', 'Documentos mais citados', 'Most cited documents'),
  chart('rankingDocuments', 'production', 'Top 10 documentos', 'Top 10 documents'),
  chart('sankey', 'advanced', 'Evolução temática', 'Thematic evolution'),
  chart('boxplot', 'advanced', 'Distribuição comparativa', 'Comparative distribution'),
  chart('genetics', 'advanced', 'Genética das ideias', 'Genetics of ideas'),
  chart('concept2d', 'advanced', 'Mapa conceitual 2D', 'Concept map 2D'),
  chart('concept3d', 'advanced', 'Mapa conceitual 3D', 'Concept map 3D'),
  chart('thematicMap', 'advanced', 'Mapa temático', 'Thematic map'),
  chart('historiograph', 'advanced', 'Historiógrafo', 'Historiograph'),
  chart('lotka', 'advanced', 'Lei de Lotka', 'Lotka’s law'),
];

/** Os gráficos que formam a seção final, "Análises visuais avançadas". */
export const ADVANCED_CHARTS: readonly ReportChartId[] = REPORT_ITEMS.filter(
  (item): item is ReportItem & { id: ReportChartId } => item.group === 'advanced',
).map((item) => item.id);

export function reportLabel(id: ReportItemId, locale: 'pt' | 'en'): string {
  return REPORT_ITEMS.find((item) => item.id === id)?.label[locale] ?? id;
}

export function selectionOf(pick: (item: ReportItem) => boolean): ReportSelection {
  return Object.fromEntries(REPORT_ITEMS.map((item) => [item.id, pick(item)])) as ReportSelection;
}

export const DEFAULT_SELECTION: ReportSelection = selectionOf((item) => item.byDefault);
