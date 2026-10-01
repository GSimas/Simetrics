import {
  BarChart3,
  Bot,
  CircleCheckBig,
  ClipboardList,
  Compass,
  Copy,
  Database,
  FileText,
  Filter,
  FolderOpen,
  Gauge,
  GitBranch,
  Globe2,
  ListFilter,
  HelpCircle,
  LayoutGrid,
  Layers,
  ListOrdered,
  Network,
  Radar,
  Search,
  Settings,
  Share2,
  ShieldCheck,
  Sparkles,
  Table2,
  TextSearch,
  TrendingUp,
  UserRound,
  Users,
  type LucideIcon,
} from 'lucide-react';

import { optionsForType } from '@/core/search';
import { collectColumns, pickColumn, splitTokens } from '@/core/text';
import { FIELD_CANDIDATES } from '@/lib/schema';
import { useDataset } from '@/state/dataset.store';
import { resolveEntity, useNavigation } from '@/state/navigation.store';
import { useReviewNav, type ReviewStep } from '@/state/review.store';

export type TourTab = 'data' | 'overview' | 'networks' | 'advanced' | 'search' | 'report' | 'review';

/** Abre uma etapa da revisão sistematizada antes de procurar o alvo. */
function openReviewStep(step: ReviewStep): () => void {
  return () => useReviewNav.getState().setStep(step);
}

interface TourCopy {
  title: string;
  body: string;
  bullets?: string[];
}

export interface TourStep {
  id: string;
  Icon: LucideIcon;
  /** Valor de `data-tour` do elemento destacado. Sem alvo, o cartão fica centralizado. */
  target?: string;
  /** Aba que precisa estar aberta para o alvo existir. */
  tab?: TourTab;
  /** O passo só faz sentido com uma base carregada — o tour carrega a de exemplo. */
  needsData?: boolean;
  /** Pulado em silêncio se o alvo não aparecer (bloco condicional). */
  optional?: boolean;
  /** Ajuste do app antes de procurar o alvo (ex.: abrir um perfil no Motor de Busca). */
  prepare?: () => void;
  pt: TourCopy;
  en: TourCopy;
}

/** Abre no Motor de Busca o autor de maior destaque, para o dossiê existir. */
function openTopAuthor(): void {
  const { tables, searchOptions } = useDataset.getState();
  const navigation = useNavigation.getState();
  if (navigation.searchTerm) {
    useNavigation.setState({ activeTab: 'search', chatOpen: false });
    return;
  }
  const candidate =
    tables?.authors[0]?.entity ?? (searchOptions ? optionsForType(searchOptions, 'Autor')[0] : undefined);
  const match = resolveEntity(candidate, ['Autor']);
  if (match) navigation.selectEntity(match.type, match.term);
  useNavigation.setState({ activeTab: 'search' });
}

/** A palavra-chave mais frequente da base — uma consulta que certamente traz resultados. */
function topKeyword(): string {
  const active = useDataset.getState().active ?? [];
  const column = pickColumn(collectColumns(active), FIELD_CANDIDATES.keywords);
  if (!column) return '';
  const counts = new Map<string, { term: string; count: number }>();
  for (const doc of active) {
    for (const term of splitTokens(doc[column])) {
      const key = term.toLowerCase();
      const entry = counts.get(key) ?? { term, count: 0 };
      entry.count += 1;
      counts.set(key, entry);
    }
  }
  let best = { term: '', count: 0 };
  for (const entry of counts.values()) if (entry.count > best.count) best = entry;
  return best.term;
}

/** Abre no Motor de Busca os resultados de uma consulta de exemplo. */
function showSampleResults(): void {
  const { searchQuery } = useNavigation.getState();
  useNavigation.setState({
    activeTab: 'search',
    searchQuery: searchQuery || topKeyword(),
    searchTerm: null,
    chatOpen: false,
  });
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: 'welcome',
    Icon: Compass,
    pt: {
      title: 'Tour guiado pelo Simetrics',
      body: 'Vamos percorrer o Simetrics inteiro, bloco a bloco: dados, Motor de Busca e conversa com a IA, indicadores, gráficos, redes, relatório e revisão sistematizada. Cada parada destaca uma parte da tela e explica como ler e usar.',
      bullets: [
        'Sem base aberta, o tour carrega a base de exemplo (973 documentos reais).',
        'Navegue com os botões, com as setas ← → do teclado, ou saia com Esc.',
        'A página continua clicável: experimente o que estiver destacado.',
      ],
    },
    en: {
      title: 'Guided tour of Simetrics',
      body: 'We will walk through all of Simetrics, block by block: data, Search Engine and AI conversation, indicators, charts, networks, report and systematized review. Each stop highlights part of the screen and explains how to read and use it.',
      bullets: [
        'With no dataset open, the tour loads the demo dataset (973 real documents).',
        'Move with the buttons or the ← → keys, and leave with Esc.',
        'The page stays clickable: try whatever is highlighted.',
      ],
    },
  },
  {
    id: 'tabs',
    Icon: LayoutGrid,
    target: 'tabs',
    tab: 'data',
    pt: {
      title: 'A barra lateral',
      body: 'O trabalho segue uma jornada, de cima para baixo nesta barra: primeiro os dados, depois a exploração e as análises, todas sobre a mesma base. Recolha a barra para ganhar espaço.',
      bullets: [
        '01 Dados — importação e deduplicação. Os módulos abaixo abrem com uma base carregada; a Revisão já abre sem ela, pelo protocolo.',
        '02 Motor de Busca — busca nos documentos, dossiês e conversa com a IA sobre a base.',
        '03 Análise Bibliométrica — Informações Principais, Redes, Análises Avançadas e Relatório.',
        '04 Revisão Sistematizada — protocolo, triagem, qualidade, extração, síntese e PRISMA, com o progresso de cada etapa.',
      ],
    },
    en: {
      title: 'The sidebar',
      body: 'Work follows a journey, top to bottom in this bar: data first, then exploration and analyses, all over the same dataset. Collapse the bar to gain space.',
      bullets: [
        '01 Data — import and deduplication. The modules below open once a dataset is loaded; the Review opens without one, starting with the protocol.',
        '02 Search Engine — document search, dossiers and AI conversation about the dataset.',
        '03 Bibliometric Analysis — Main Information, Networks, Advanced Analyses and Report.',
        '04 Systematized Review — protocol, screening, quality, extraction, synthesis and PRISMA, with the progress of each stage.',
      ],
    },
  },
  {
    id: 'upload',
    Icon: Database,
    target: 'upload',
    tab: 'data',
    pt: {
      title: 'Base de dados',
      body: 'Tudo começa aqui. Envie um ou vários arquivos de uma vez: o Simetrics reconhece a origem de cada um e une as bases num só acervo, processado 100% no seu navegador.',
      bullets: [
        'RIS (SciELO, WoS, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (WoS) e TXT/NBIB (PubMed).',
        'Confira a base sugerida para cada arquivo antes de processar.',
        '"Carregar exemplo" abre uma base pronta; "Limpar base" recomeça do zero.',
      ],
    },
    en: {
      title: 'Dataset',
      body: 'Everything starts here. Upload one or several files at once: Simetrics detects the source of each and merges them into a single collection, processed 100% in your browser.',
      bullets: [
        'RIS (SciELO, WoS, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (WoS) and TXT/NBIB (PubMed).',
        'Check the suggested database for each file before processing.',
        '"Load demo" opens a ready dataset; "Clear dataset" starts over.',
      ],
    },
  },
  {
    id: 'dedup',
    Icon: Copy,
    target: 'dedup',
    tab: 'data',
    needsData: true,
    pt: {
      title: 'Deduplicação',
      body: 'Ao juntar bases diferentes, o mesmo artigo costuma aparecer mais de uma vez. Escolha a estratégia e execute: todos os indicadores são recalculados sobre a base limpa.',
      bullets: [
        'Por DOI — remove registros com o mesmo DOI.',
        'Por similaridade — compara títulos (Jaccard) e pega duplicatas sem DOI.',
        'Ambos — aplica os dois critérios. O relatório lista o que saiu e o que ficou.',
      ],
    },
    en: {
      title: 'Deduplication',
      body: 'When merging different databases, the same paper often shows up more than once. Pick a strategy and run it: every indicator is recomputed on the clean dataset.',
      bullets: [
        'By DOI — removes records sharing a DOI.',
        'By similarity — compares titles (Jaccard) and catches duplicates without a DOI.',
        'Both — applies both criteria. The report lists what was removed and what was kept.',
      ],
    },
  },
  {
    id: 'search-picker',
    Icon: Search,
    target: 'search-picker',
    tab: 'search',
    needsData: true,
    prepare: showSampleResults,
    pt: {
      title: 'Motor de Busca',
      body: 'Uma caixa só, como num buscador: enquanto você digita, ela sugere autores, países, venues, palavras-chave, temas e títulos; Enter busca o texto nos documentos, do mais ao menos relevante. Todo nome clicável no Simetrics (barras, nós, países, linhas de tabela) também traz você para cá. Buscamos a palavra-chave mais frequente da base como exemplo.',
    },
    en: {
      title: 'Search Engine',
      body: 'A single box, like a search engine: as you type it suggests authors, countries, venues, keywords, themes and titles; Enter searches the text of the documents, most relevant first. Every clickable name in Simetrics (bars, nodes, countries, table rows) also brings you here. We searched the most frequent keyword in the dataset as an example.',
    },
  },
  {
    id: 'search-results',
    Icon: ListFilter,
    target: 'search-results',
    tab: 'search',
    needsData: true,
    prepare: showSampleResults,
    pt: {
      title: 'Resultados da busca',
      body: 'Enter traz os documentos mais relevantes para a frase, como num buscador: autores, ano e periódico, o título e um trecho do resumo com os termos em destaque. Acima, as entidades com esse nome. Qualquer título abre o dossiê do documento.',
    },
    en: {
      title: 'Search results',
      body: 'Enter brings the documents most relevant to the phrase, like a search engine: authors, year and venue, the title and an abstract excerpt with the terms highlighted. Above them, the entities with that name. Any title opens the document dossier.',
    },
  },
  {
    id: 'search-ask',
    Icon: Sparkles,
    target: 'search-ask',
    tab: 'search',
    needsData: true,
    prepare: showSampleResults,
    pt: {
      title: 'Perguntar à IA',
      body: 'Quando a busca não basta, a mesma frase vira uma pergunta à Simi, sobre a base inteira. A conversa abre em tela cheia e cada resposta lista os documentos em que se apoiou.',
      bullets: [
        'Só os documentos mais relevantes para a pergunta e um panorama agregado vão ao provedor de IA; o restante fica no navegador.',
        'Há perguntas gratuitas; depois, entre com o OpenRouter ou informe sua chave.',
      ],
    },
    en: {
      title: 'Ask AI',
      body: 'When search is not enough, the same phrase becomes a question to Simi about the whole dataset. The conversation opens full screen, and each answer lists the documents it relied on.',
      bullets: [
        'Only the documents most relevant to the question and an aggregate overview go to the AI provider; everything else stays in the browser.',
        'A few questions are free; after that, sign in with OpenRouter or enter your key.',
      ],
    },
  },
  {
    id: 'search-dossier',
    Icon: UserRound,
    target: 'search-dossier',
    tab: 'search',
    needsData: true,
    prepare: openTopAuthor,
    pt: {
      title: 'Dossiê da entidade',
      body: 'O perfil reúne documentos, coautores, citações, índices h, g, i10 e m, período de atividade e a especialização temática. Os coautores e países também são clicáveis — dá para navegar de perfil em perfil. Abrimos o autor de maior destaque da base como exemplo.',
    },
    en: {
      title: 'Entity dossier',
      body: 'The profile gathers documents, co-authors, citations, h, g, i10 and m indices, active period and thematic specialisation. Co-authors and countries are clickable too — you can hop from profile to profile. We opened the most prominent author in the dataset as an example.',
    },
  },
  {
    id: 'search-lexico',
    Icon: TextSearch,
    target: 'search-lexico',
    tab: 'search',
    needsData: true,
    optional: true,
    prepare: openTopAuthor,
    pt: {
      title: 'Léxico e trajetória',
      body: 'A nuvem de palavras mostra o vocabulário da entidade; a linha do tempo, como a produção se distribui nos anos; e, quando há países, o mapa das suas colaborações.',
    },
    en: {
      title: 'Lexicon and trajectory',
      body: "The word cloud shows the entity's vocabulary; the timeline, how its output spreads over the years; and, when countries are present, the map of its collaborations.",
    },
  },
  {
    id: 'search-similar',
    Icon: Users,
    target: 'search-similar',
    tab: 'search',
    needsData: true,
    optional: true,
    prepare: openTopAuthor,
    pt: {
      title: 'Entidades semelhantes',
      body: 'Perfis com "DNA acadêmico" parecido, pela similaridade de Jaccard entre coautores, venues e vocabulário. Ótimo para descobrir pares, revisores ou grupos que você ainda não conhecia.',
    },
    en: {
      title: 'Similar entities',
      body: 'Profiles with a similar "academic DNA", by Jaccard similarity across co-authors, venues and vocabulary. Great for discovering peers, reviewers or groups you did not know yet.',
    },
  },
  {
    id: 'search-docs',
    Icon: FileText,
    target: 'search-docs',
    tab: 'search',
    needsData: true,
    optional: true,
    prepare: openTopAuthor,
    pt: {
      title: 'Documentos da entidade',
      body: 'Todos os documentos ligados ao perfil, com ano, citações e link do DOI, prontos para filtrar, ordenar e exportar.',
    },
    en: {
      title: "Entity's documents",
      body: 'Every document linked to the profile, with year, citations and DOI link, ready to filter, sort and export.',
    },
  },
  {
    id: 'bibliometric-views',
    Icon: BarChart3,
    target: 'bibliometric-views',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Análise Bibliométrica',
      body: 'A análise bibliométrica tem quatro vistas, aninhadas na barra lateral.',
      bullets: [
        '1 Informações Principais — indicadores, rankings e tabelas analíticas.',
        '2 Redes — coocorrência, comunidades e colaboração internacional.',
        '3 Análises Avançadas — Sankey, boxplot, mapas conceitual e temático, historiograph e Lotka.',
        '4 Relatório — um documento PDF ou Word com o que você escolher.',
      ],
    },
    en: {
      title: 'Bibliometric Analysis',
      body: 'The bibliometric analysis has four views, nested in the sidebar.',
      bullets: [
        '1 Main Information — indicators, rankings and analytical tables.',
        '2 Networks — co-occurrence, communities and international collaboration.',
        '3 Advanced Analyses — Sankey, boxplot, concept and thematic maps, historiograph and Lotka.',
        '4 Report — a PDF or Word document with whatever you select.',
      ],
    },
  },
  {
    id: 'launchers',
    Icon: Layers,
    target: 'launchers',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Aprofundar a análise',
      body: 'Estes blocos abrem numa janela própria — e só são calculadas quando você abre. Vamos ver cada um.',
    },
    en: {
      title: 'Deeper analysis',
      body: 'These blocks open in their own window — and are only computed when you open them. Let us look at each one.',
    },
  },
  {
    id: 'launcher-meta',
    Icon: ShieldCheck,
    target: 'launcher-meta',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Qualidade dos metadados',
      body: 'Mostra, campo a campo (autores, resumo, palavras-chave, DOI, citações…), quantos registros estão sem informação e classifica a completude de Excelente a Ruim. Confira aqui antes de confiar numa análise que dependa de um campo incompleto.',
    },
    en: {
      title: 'Metadata quality',
      body: 'Shows, field by field (authors, abstract, keywords, DOI, citations…), how many records are missing data, rated from Excellent to Poor. Check it before trusting an analysis that depends on an incomplete field.',
    },
  },
  {
    id: 'launcher-theme',
    Icon: Sparkles,
    target: 'launcher-theme',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Mapeamento temático por IA',
      body: 'Agrupa os documentos por similaridade e pede à IA que dê nome a cada grupo, criando a categoria "Temas (IA)" usada no gráfico de produção, nas tabelas (Quociente Locacional) e no relatório. Requer uma chave de IA própria (BYOK), configurada na assistente Simi.',
    },
    en: {
      title: 'AI thematic mapping',
      body: 'Clusters documents by similarity and asks the AI to name each cluster, creating the "Themes (AI)" category used in the production chart, the tables (Location Quotient) and the report. Requires your own AI key (BYOK), set up in the Simi assistant.',
    },
  },
  {
    id: 'ticker',
    Icon: Gauge,
    target: 'ticker',
    needsData: true,
    pt: {
      title: 'Faixa de indicadores',
      body: 'Os números-chave da base ficam sempre à vista no cabeçalho, em qualquer tela: bases de origem, documentos, período, autores, países, venues e crescimento. Clique na faixa para pausar a rolagem.',
    },
    en: {
      title: 'Indicator strip',
      body: 'The key numbers of the dataset stay visible in the header on every screen: source databases, documents, period, authors, countries, venues and growth. Click the strip to pause the scrolling.',
    },
  },
  {
    id: 'kpis',
    Icon: TrendingUp,
    target: 'kpis',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Indicadores principais',
      body: 'Oito indicadores resumem a base no padrão Bibliometrix.',
      bullets: [
        'Documentos, autores, países e venues — o tamanho do acervo.',
        'Crescimento anual — taxa composta de publicações entre o primeiro e o último ano.',
        'Citações por ano — média de citações de cada documento dividida pela sua idade.',
        'Colaboração internacional — documentos com autores de mais de um país.',
        'Autores por documento — o índice de coautoria.',
      ],
    },
    en: {
      title: 'Key indicators',
      body: 'Eight indicators summarise the dataset, following Bibliometrix.',
      bullets: [
        'Documents, authors, countries and venues — the size of the collection.',
        'Annual growth — compound rate of publications between the first and last year.',
        'Citations per year — average citations of each document divided by its age.',
        'International collaboration — documents with authors from more than one country.',
        'Authors per document — the co-authorship index.',
      ],
    },
  },
  {
    id: 'production',
    Icon: BarChart3,
    target: 'production',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Produção ao longo do tempo',
      body: 'Documentos publicados por ano. Use "Categorizar por" para quebrar a série por país, base de dados, tipo de trabalho ou tema de IA, e "Visualização" para alternar entre barras agrupadas, empilhadas ou linhas.',
      bullets: [
        'Passe o mouse para ver os valores; arraste para dar zoom num período.',
        'Nas categorias país e tema, clicar numa série abre o perfil no Motor de Busca.',
        'Os botões acima do gráfico ampliam em tela cheia e exportam em SVG, PNG ou JPG.',
      ],
    },
    en: {
      title: 'Production over time',
      body: 'Documents published per year. Use "Categorize by" to split the series by country, database, document type or AI theme, and "View" to switch between grouped bars, stacked bars or lines.',
      bullets: [
        'Hover to read values; drag to zoom into a period.',
        'For countries and themes, clicking a series opens its profile in the Search Engine.',
        'The buttons above the chart expand it to full screen and export SVG, PNG or JPG.',
      ],
    },
  },
  {
    id: 'top10',
    Icon: ListOrdered,
    target: 'top10',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Top 10',
      body: 'Os dez autores, documentos, países e venues de maior destaque. Cada ranking tem sua própria métrica.',
      bullets: [
        'Citações em soma, média ou mediana, ou número de documentos.',
        'Países também por documentos por autor (média ou mediana).',
        'Médias e medianas favorecem quem tem poucos trabalhos muito citados.',
        'Clique num item para abrir o perfil completo no Motor de Busca.',
      ],
    },
    en: {
      title: 'Top 10',
      body: 'The ten most prominent authors, documents, countries and venues. Each ranking has its own metric.',
      bullets: [
        'Citations as sum, mean or median, or number of documents.',
        'Countries also by documents per author (mean or median).',
        'Means and medians favour entities with few, highly cited works.',
        'Click an item to open its full profile in the Search Engine.',
      ],
    },
  },
  {
    id: 'tables',
    Icon: Table2,
    target: 'tables',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Tabelas analíticas',
      body: 'A base completa e os rankings de autores, países, venues e palavras-chave, com índices h, g, i10 e m, citações (soma, média, mediana, desvio padrão), especialização temática e coautores.',
      bullets: [
        'Filtre e ordene qualquer coluna na própria tabela.',
        'Clique num nome para abrir o perfil; exporte tudo em CSV.',
      ],
    },
    en: {
      title: 'Analytical tables',
      body: 'The full dataset and the rankings of authors, countries, venues and keywords, with h, g, i10 and m indices, citations (sum, mean, median, standard deviation), thematic specialisation and co-authors.',
      bullets: [
        'Filter and sort any column right in the table.',
        'Click a name to open its profile; export everything as CSV.',
      ],
    },
  },
  {
    id: 'net-ecology',
    Icon: Network,
    target: 'net-ecology',
    tab: 'networks',
    needsData: true,
    pt: {
      title: 'Ecologia da rede',
      body: 'Métricas globais da rede de coocorrência: quantos nós, arestas e componentes, a densidade, o coeficiente de clustering, a entropia e a eficiência. Em "Ver todas as métricas" há o restante, cada uma com sua explicação no "i".',
    },
    en: {
      title: 'Network ecology',
      body: 'Global metrics of the co-occurrence network: nodes, edges and components, density, clustering coefficient, entropy and efficiency. "See all metrics" shows the rest, each explained in its "i".',
    },
  },
  {
    id: 'net-cooccurrence',
    Icon: Users,
    target: 'net-cooccurrence',
    tab: 'networks',
    needsData: true,
    pt: {
      title: 'Rede de coocorrência',
      body: 'Escolha o tipo de rede — coautoria, palavras-chave ou países —, quantas entidades exibir e qual métrica define o tamanho dos nós (grau, intermediação, proximidade…). As cores marcam as comunidades detectadas pelo algoritmo de Louvain.',
    },
    en: {
      title: 'Co-occurrence network',
      body: 'Choose the network type — co-authorship, keywords or countries —, how many entities to show and which metric sizes the nodes (degree, betweenness, closeness…). Colours mark the communities found by the Louvain algorithm.',
    },
  },
  {
    id: 'net-graph',
    Icon: Share2,
    target: 'net-graph',
    tab: 'networks',
    needsData: true,
    pt: {
      title: 'Grafo de rede e grafo radial',
      body: 'Duas formas de ver a mesma rede. O grafo de forças aproxima quem colabora; o radial (de cordas) dispõe os nós num círculo, agrupados por comunidade.',
      bullets: [
        'Role para dar zoom e arraste para mover o gráfico.',
        'Passe o mouse num nó para destacar suas conexões; clique para fixar e clique de novo para abrir o perfil.',
        'Amplie em tela cheia ou exporte como imagem pelos botões do canto.',
      ],
    },
    en: {
      title: 'Network graph and radial graph',
      body: 'Two ways to see the same network. The force graph pulls collaborators together; the radial (chord) graph lays nodes on a circle, grouped by community.',
      bullets: [
        'Scroll to zoom and drag to pan the chart.',
        'Hover a node to highlight its links; click to pin it and click again to open its profile.',
        'Expand to full screen or export as an image with the corner buttons.',
      ],
    },
  },
  {
    id: 'net-collab',
    Icon: Globe2,
    target: 'net-collab',
    tab: 'networks',
    needsData: true,
    pt: {
      title: 'Colaboração internacional',
      body: 'O mapa-múndi pinta cada país pela sua produção e traça arcos entre os que publicam juntos; o grafo radial mostra as mesmas parcerias em círculo.',
      bullets: [
        'Primeiro clique num país destaca ele e seus parceiros; o segundo abre o perfil.',
        'Zoom pela roda do mouse ou pelos botões; arraste para mover o mapa.',
      ],
    },
    en: {
      title: 'International collaboration',
      body: 'The world map shades each country by its output and draws arcs between countries that publish together; the radial graph shows the same partnerships on a circle.',
      bullets: [
        'The first click on a country highlights it and its partners; the second opens its profile.',
        'Zoom with the mouse wheel or the buttons; drag to pan the map.',
      ],
    },
  },
  {
    id: 'net-metrics',
    Icon: Table2,
    target: 'net-metrics',
    tab: 'networks',
    needsData: true,
    pt: {
      title: 'Métricas por nó',
      body: 'A tabela de cada nó da rede: grau absoluto, centralidade de grau, autovetor, intermediação (betweenness) e proximidade (closeness). Ordene para achar os atores centrais e as pontes entre comunidades, e exporte em CSV.',
    },
    en: {
      title: 'Per-node metrics',
      body: 'The table of every node in the network: absolute degree, degree centrality, eigenvector, betweenness and closeness. Sort to find the central actors and the bridges between communities, and export as CSV.',
    },
  },
  {
    id: 'advanced',
    Icon: Radar,
    target: 'visual',
    tab: 'advanced',
    needsData: true,
    pt: {
      title: 'Análises visuais avançadas',
      body: 'Sete visualizações clássicas da bibliometria, numa vista própria, cada uma com seu "Como ler".',
      bullets: [
        'Sankey — evolução dos temas entre períodos (ajuste os cortes nos controles deslizantes).',
        'Boxplot — distribuição de citações entre grupos.',
        'Genética dos termos, mapa conceitual (PCA 2D/3D) e mapa temático em quadrantes.',
        'Historiograph — a linhagem de citações entre documentos.',
        'Lei de Lotka — a produtividade dos autores.',
      ],
    },
    en: {
      title: 'Advanced visual analyses',
      body: 'Seven classic bibliometric visualisations, in their own view, each with its own "How to read".',
      bullets: [
        'Sankey — how themes evolve across periods (adjust the cuts with the sliders).',
        'Boxplot — citation distribution across groups.',
        'Term genetics, concept map (2D/3D PCA) and quadrant thematic map.',
        'Historiograph — the citation lineage between documents.',
        "Lotka's law — author productivity.",
      ],
    },
  },
  {
    id: 'report-builder',
    Icon: FileText,
    target: 'report-builder',
    tab: 'report',
    needsData: true,
    pt: {
      title: 'Relatório executivo',
      body: 'Monte o relatório marcando seções e gráficos — do resumo executivo às redes, mapas e análises visuais avançadas — e quantas linhas entram em cada tabela. Em Baixar, escolha PDF diagramado ou Word (DOCX) editável, gerados no navegador.',
    },
    en: {
      title: 'Executive report',
      body: 'Build the report by ticking sections and charts — from the executive summary to networks, maps and advanced visual analyses — and how many rows each table includes. Under Download, pick a laid-out PDF or an editable Word (DOCX) file, generated in the browser.',
    },
  },
  {
    id: 'report-preview',
    Icon: FileText,
    target: 'report-preview',
    tab: 'report',
    needsData: true,
    pt: {
      title: 'Pré-visualização',
      body: 'Uma prévia ao vivo do documento, ao lado da seleção, atualizada a cada item marcado ou desmarcado. O que você vê aqui é o que vai para o arquivo.',
    },
    en: {
      title: 'Preview',
      body: 'A live preview of the document, next to the selection, updated whenever you tick or untick an item. What you see here is what goes into the file.',
    },
  },
  {
    id: 'review-overview',
    Icon: ClipboardList,
    target: 'review-steps',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('protocol'),
    pt: {
      title: 'Revisão sistematizada',
      body: 'Revisão sistemática, de escopo, integrativa, rápida, guarda-chuva ou mapeamento sistemático, sobre a mesma base da análise bibliométrica. São seis etapas, da pergunta ao PRISMA, no fluxo de ferramentas como o Parsifal — subpáginas na barra lateral.',
      bullets: [
        'Este painel mostra o avanço: o total da revisão e a completude de cada etapa, que se atualiza a cada decisão. Os anéis na barra lateral acompanham.',
        'O exemplo traz uma revisão de escopo de amostra sobre memética, só para visualização.',
        '"Fazer cópia editável" transforma o exemplo num projeto seu, com a revisão junto.',
        'Tudo fica salvo no projeto, no seu navegador.',
      ],
    },
    en: {
      title: 'Systematized review',
      body: 'Systematic, scoping, integrative, rapid, umbrella review or systematic mapping, over the same dataset as the bibliometric analysis. Six stages, from the question to PRISMA, following tools like Parsifal — subpages in the sidebar.',
      bullets: [
        'This panel shows the progress: the review total and how complete each stage is, updated with every decision. The rings in the sidebar follow along.',
        'The demo ships a sample scoping review on memetics, view only.',
        '"Make an editable copy" turns the demo into your own project, review included.',
        'Everything is saved in the project, in your browser.',
      ],
    },
  },
  {
    id: 'review-protocol',
    Icon: ClipboardList,
    target: 'review-search',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('protocol'),
    pt: {
      title: 'Protocolo e string de busca',
      body: 'Defina o tipo de revisão, a pergunta (PICO, PICOC, PCC ou SPIDER), os critérios de elegibilidade, o checklist de qualidade e o formulário de extração. Os conceitos e sinônimos viram a string de busca já na sintaxe de cada base — Scopus, Web of Science, PubMed, Cochrane — pronta para copiar.',
    },
    en: {
      title: 'Protocol and search string',
      body: 'Set the review type, the question (PICO, PICOC, PCC or SPIDER), eligibility criteria, quality checklist and extraction form. Concepts and synonyms become the search string already in each database syntax — Scopus, Web of Science, PubMed, Cochrane — ready to copy.',
    },
  },
  {
    id: 'review-screening',
    Icon: Filter,
    target: 'review-screening',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('screening'),
    pt: {
      title: 'Triagem em duas etapas',
      body: 'Primeiro título e resumo, depois texto completo. Incluir acende em verde e excluir em vermelho; no texto completo, a exclusão pede o motivo, que vai para o diagrama PRISMA.',
      bullets: [
        'Atalhos: I incluir, E excluir, T talvez, N não recuperado, J/K para navegar, U para desfazer.',
        'Os termos da string de busca aparecem destacados no título e no resumo.',
      ],
    },
    en: {
      title: 'Two-stage screening',
      body: 'Title and abstract first, then full text. Including lights up green and excluding red; at full text, exclusion asks for the reason, which goes into the PRISMA diagram.',
      bullets: [
        'Shortcuts: I include, E exclude, T maybe, N not retrieved, J/K to navigate, U to undo.',
        'The search string terms are highlighted in the title and abstract.',
      ],
    },
  },
  {
    id: 'review-quality',
    Icon: ShieldCheck,
    target: 'review-step-quality',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('quality'),
    pt: {
      title: 'Qualidade e extração',
      body: 'Cada estudo incluído passa pelo checklist: as respostas têm peso, e a nota é comparada à nota de corte — verde passa, vermelho fica abaixo. Na etapa seguinte, o formulário de extração registra as características de cada estudo da seleção final.',
    },
    en: {
      title: 'Quality and extraction',
      body: 'Each included study goes through the checklist: answers carry weights, and the score is compared with the cutoff — green meets it, red falls below. In the next stage, the extraction form records the characteristics of each study in the final selection.',
    },
  },
  {
    id: 'review-synthesis',
    Icon: BarChart3,
    target: 'review-step-synthesis',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('synthesis'),
    pt: {
      title: 'Síntese',
      body: 'Indicadores da seleção final, estudos por ano, o resumo de cada campo extraído e as tabelas de características e de qualidade, exportáveis em CSV.',
    },
    en: {
      title: 'Synthesis',
      body: 'Final selection indicators, studies per year, a summary of each extracted field and the characteristics and quality tables, exportable as CSV.',
    },
  },
  {
    id: 'review-prisma',
    Icon: GitBranch,
    target: 'review-export',
    tab: 'review',
    needsData: true,
    prepare: openReviewStep('prisma'),
    pt: {
      title: 'PRISMA e relatório',
      body: 'O fluxo PRISMA 2020 sai das próprias decisões. Exporte o arquivo para o PRISMALab (prisma.scientata.com) desenhar o diagrama, as planilhas de decisões e o relatório Word da revisão.',
    },
    en: {
      title: 'PRISMA and report',
      body: 'The PRISMA 2020 flow comes straight from the decisions. Export the file for PRISMALab (prisma.scientata.com) to draw the diagram, the decision spreadsheets and the review Word report.',
    },
  },
  {
    id: 'chat',
    Icon: Bot,
    target: 'chat',
    tab: 'review',
    needsData: true,
    pt: {
      title: 'Simi, a assistente científica',
      body: 'Converse com a sua base: a Simi busca os documentos relevantes e responde a partir deles, listando as fontes. No Motor de Busca, use ✦ Perguntar à IA; nas análises, este botão — é a mesma conversa. Há perguntas gratuitas; depois, entre com o OpenRouter ou informe a chave do seu provedor, que fica só no seu navegador.',
    },
    en: {
      title: 'Simi, the scientific assistant',
      body: 'Chat with your dataset: Simi retrieves the relevant documents and answers from them, listing the sources. In the Search Engine use ✦ Ask AI; in the analyses, this button — it is the same conversation. A few questions are free; after that, sign in with OpenRouter or enter your provider key, which stays in your browser only.',
    },
  },
  {
    id: 'projects',
    Icon: FolderOpen,
    target: 'projects',
    pt: {
      title: 'Meus projetos',
      body: 'Cada base vira um projeto salvo automaticamente no seu navegador. Aqui você volta à lista para abrir, renomear, duplicar, exportar ou importar projetos.',
    },
    en: {
      title: 'My projects',
      body: 'Each dataset becomes a project saved automatically in your browser. From here you return to the list to open, rename, duplicate, export or import projects.',
    },
  },
  {
    id: 'settings',
    Icon: Settings,
    target: 'settings',
    pt: {
      title: 'Configurações',
      body: 'Tema claro ou escuro, idioma (português ou inglês), tamanho da fonte e alto contraste. As preferências valem para todos os projetos.',
    },
    en: {
      title: 'Settings',
      body: 'Light or dark theme, language (Portuguese or English), font size and high contrast. Preferences apply to every project.',
    },
  },
  {
    id: 'tutorial',
    Icon: HelpCircle,
    target: 'tutorial',
    pt: {
      title: 'Sempre à mão',
      body: 'O guia rápido e este tour ficam neste botão. Volte quando quiser rever algum bloco.',
    },
    en: {
      title: 'Always at hand',
      body: 'The quick guide and this tour live behind this button. Come back whenever you want to revisit a block.',
    },
  },
  {
    id: 'done',
    Icon: CircleCheckBig,
    pt: {
      title: 'Pronto para explorar',
      body: 'Esse foi o Simetrics completo. Importe a sua própria base em "Base de dados" ou continue explorando o exemplo. Boa pesquisa!',
    },
    en: {
      title: 'Ready to explore',
      body: 'That was all of Simetrics. Import your own dataset in "Dataset" or keep exploring the demo. Happy researching!',
    },
  },
];
