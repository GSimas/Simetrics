/**
 * Textos das sub-abas de visualização (VisualAnalyses). Ficam aqui, e não em
 * i18n/translations.ts, pelo mesmo critério da classificação híbrida: são muitos e só
 * estes painéis os usam. Marcadores entre chaves são trocados com `.replace`.
 */
import type { Locale } from '@/lib/i18n/translations';

const pt = {
  // Boxplot
  compareBy: 'Comparar por',
  metric: 'Métrica',
  yScale: 'Escala do eixo Y',
  linear: 'Linear',
  logarithmic: 'Logarítmica',
  noThemes: 'Nenhum tema disponível. Use o mapeamento temático por IA acima para gerá-los.',
  noDimensionData: 'A base não traz dados de {dimension}.',
  selectUpTo: 'Selecione até {max} itens ({count} selecionados)',
  selectAtLeastOne: 'Selecione ao menos um item para comparar.',
  boxplotExport: 'distribuicao-comparativa',
  // Sankey
  computingFlows: 'Calculando fluxos temáticos…',
  sankeyNeedsData: 'A base precisa de anos e palavras-chave para montar o fluxo.',
  termsPerPeriod: 'Termos por período',
  period: 'Período {n}',
  sankeyTip:
    'As linhas mais grossas são termos que sobreviveram de um período ao seguinte; as ' +
    'finas, termos distintos que costumam aparecer nos mesmos documentos.',
  noFlow: 'Nenhum fluxo para os períodos selecionados — tente um recorte mais amplo.',
  sankeyExport: 'evolucao-tematica',
  continuity: 'Continuidade',
  intersection: 'Intersecção',
  // Genética dos termos
  computingGenetics: 'Calculando ciclo de vida dos termos…',
  geneticsNeedsData: 'A base precisa de palavras-chave e anos para esta análise.',
  geneticsTip:
    'Cada ponto é uma palavra-chave. O eixo X mostra quando ela apareceu pela primeira ' +
    'vez; o Y, por quantos anos permaneceu em uso; o tamanho, quantas vezes se replicou. ' +
    'Termos no alto e à esquerda são o núcleo estável da área; à direita e embaixo, as ' +
    'fronteiras recentes.',
  termBirthYear: 'Ano de nascimento do termo',
  longevityYears: 'Longevidade (anos)',
  citations: 'Citações',
  bornIn: 'Nasceu em',
  longevity: 'Longevidade',
  years: '{n} anos',
  replications: 'Replicações',
  geneticsExport: 'genetica-das-ideias',
  // Mapa conceitual
  projectingTerms: 'Projetando termos…',
  conceptNeedsData: 'A base precisa de palavras-chave suficientes para o mapa conceitual.',
  clusterN: 'Agrupamento {n}',
  cluster: 'Agrupamento',
  frequency: 'Frequência',
  projection: 'Projeção',
  dimensions: '{n} dimensões',
  clusters: 'Agrupamentos',
  clustersN: '{n} agrupamentos',
  conceptTip:
    'Termos próximos aparecem nos mesmos documentos. As ilhas são escolas de pensamento; ' +
    'os termos entre elas são pontes conceituais.',
  dimensionN: 'Dimensão {n}',
  conceptExport: 'mapa-conceitual',
  // Mapa temático
  buildingNetwork: 'Construindo a rede de coocorrência…',
  thematicNeedsData: 'Não há texto suficiente para montar o mapa temático.',
  textSource: 'Fonte do texto',
  abstracts: 'Resumos',
  keywords: 'Palavras-chave',
  centralityAxis: 'Centralidade (relevância externa)',
  densityAxis: 'Densidade (desenvolvimento interno)',
  niches: 'Nichos',
  motors: 'Motores',
  emerging: 'Emergentes / em declínio',
  basic: 'Básicos / transversais',
  centrality: 'Centralidade',
  density: 'Densidade',
  thematicExport: 'mapa-tematico',
  // Historiógrafo
  tracingCitations: 'Rastreando citações diretas…',
  noReferences:
    'Esta base não traz referências citadas, e sem elas não há como rastrear quais ' +
    'documentos citam quais. No Web of Science, exporte com "Full Record and Cited ' +
    'References"; no Scopus, marque "References" na exportação.',
  documents: 'Documentos',
  historiographTip:
    '{edges} citações diretas entre os {nodes} documentos mais citados. A detecção casa ' +
    'sobrenome do primeiro autor e ano dentro do texto das referências, então erra em ' +
    'homônimos e em grafias divergentes.',
  timeline: 'Linha do tempo',
  year: 'Ano',
  // Lotka
  computingLotka: 'Calculando a distribuição de produtividade…',
  loadingChart: 'Carregando gráfico…',
};

export const VISUAL_COPY: Record<Locale, typeof pt> = {
  pt,
  en: {
    compareBy: 'Compare by',
    metric: 'Metric',
    yScale: 'Y-axis scale',
    linear: 'Linear',
    logarithmic: 'Logarithmic',
    noThemes: 'No themes available. Use the AI thematic mapping above to generate them.',
    noDimensionData: 'The dataset has no {dimension} data.',
    selectUpTo: 'Select up to {max} items ({count} selected)',
    selectAtLeastOne: 'Select at least one item to compare.',
    boxplotExport: 'comparative-distribution',
    computingFlows: 'Computing thematic flows…',
    sankeyNeedsData: 'The dataset needs years and keywords to build the flow.',
    termsPerPeriod: 'Terms per period',
    period: 'Period {n}',
    sankeyTip:
      'Thicker lines are terms that survived from one period to the next; thinner ones are ' +
      'different terms that tend to appear in the same documents.',
    noFlow: 'No flows for the selected periods — try a wider range.',
    sankeyExport: 'thematic-evolution',
    continuity: 'Continuity',
    intersection: 'Intersection',
    computingGenetics: 'Computing the life cycle of terms…',
    geneticsNeedsData: 'The dataset needs keywords and years for this analysis.',
    geneticsTip:
      'Each point is a keyword. The X axis shows when it first appeared; the Y axis, how many ' +
      'years it stayed in use; the size, how many times it was replicated. Terms at the top ' +
      'left are the stable core of the field; at the bottom right, the recent frontiers.',
    termBirthYear: 'Year the term first appeared',
    longevityYears: 'Longevity (years)',
    citations: 'Citations',
    bornIn: 'First appeared',
    longevity: 'Longevity',
    years: '{n} years',
    replications: 'Replications',
    geneticsExport: 'genetics-of-ideas',
    projectingTerms: 'Projecting terms…',
    conceptNeedsData: 'The dataset needs enough keywords for the concept map.',
    clusterN: 'Cluster {n}',
    cluster: 'Cluster',
    frequency: 'Frequency',
    projection: 'Projection',
    dimensions: '{n} dimensions',
    clusters: 'Clusters',
    clustersN: '{n} clusters',
    conceptTip:
      'Nearby terms appear in the same documents. The islands are schools of thought; the ' +
      'terms between them are conceptual bridges.',
    dimensionN: 'Dimension {n}',
    conceptExport: 'concept-map',
    buildingNetwork: 'Building the co-occurrence network…',
    thematicNeedsData: 'There is not enough text to build the thematic map.',
    textSource: 'Text source',
    abstracts: 'Abstracts',
    keywords: 'Keywords',
    centralityAxis: 'Centrality (external relevance)',
    densityAxis: 'Density (internal development)',
    niches: 'Niche themes',
    motors: 'Motor themes',
    emerging: 'Emerging / declining',
    basic: 'Basic / transversal',
    centrality: 'Centrality',
    density: 'Density',
    thematicExport: 'thematic-map',
    tracingCitations: 'Tracing direct citations…',
    noReferences:
      'This dataset has no cited references, and without them there is no way to trace which ' +
      'documents cite which. In Web of Science, export with "Full Record and Cited ' +
      'References"; in Scopus, check "References" when exporting.',
    documents: 'Documents',
    historiographTip:
      '{edges} direct citations among the {nodes} most cited documents. Detection matches the ' +
      'first author’s surname and year within the reference text, so it misses homonyms and ' +
      'divergent spellings.',
    timeline: 'Timeline',
    year: 'Year',
    computingLotka: 'Computing the productivity distribution…',
    loadingChart: 'Loading chart…',
  },
};
