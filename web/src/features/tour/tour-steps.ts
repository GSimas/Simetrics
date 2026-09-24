import {
  BarChart3,
  Bot,
  CircleCheckBig,
  Compass,
  Copy,
  Database,
  FileText,
  FolderOpen,
  Gauge,
  Globe2,
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
import { useDataset } from '@/state/dataset.store';
import { resolveEntity, useNavigation } from '@/state/navigation.store';

export type TourTab = 'overview' | 'networks' | 'search' | 'report';

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
    useNavigation.setState({ activeTab: 'search' });
    return;
  }
  const candidate =
    tables?.authors[0]?.entity ?? (searchOptions ? optionsForType(searchOptions, 'Autor')[0] : undefined);
  const match = resolveEntity(candidate, ['Autor']);
  useNavigation.setState({
    activeTab: 'search',
    ...(match ? { searchType: match.type, searchTerm: match.term } : {}),
  });
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: 'welcome',
    Icon: Compass,
    pt: {
      title: 'Tour guiado pelo Simetrics',
      body: 'Vamos percorrer o Simetrics inteiro, bloco a bloco: importação, indicadores, gráficos, redes, Motor de Busca, relatório e a assistente de IA. Cada parada destaca uma parte da tela e explica como ler e usar.',
      bullets: [
        'Sem base aberta, o tour carrega a base de exemplo (973 documentos reais).',
        'Navegue com os botões, com as setas ← → do teclado, ou saia com Esc.',
        'A página continua clicável: experimente o que estiver destacado.',
      ],
    },
    en: {
      title: 'Guided tour of Simetrics',
      body: 'We will walk through all of Simetrics, block by block: import, indicators, charts, networks, Search Engine, report and the AI assistant. Each stop highlights part of the screen and explains how to read and use it.',
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
    tab: 'overview',
    pt: {
      title: 'As quatro áreas',
      body: 'O trabalho se divide em quatro abas, na ordem natural de uma análise bibliométrica.',
      bullets: [
        '01 Informações Principais — importação, indicadores, rankings e análises visuais.',
        '02 Redes — coocorrência, comunidades e colaboração internacional.',
        '03 Motor de Busca — o dossiê de qualquer autor, país, venue, termo ou documento.',
        '04 Relatório — um documento PDF ou Word com o que você escolher.',
      ],
    },
    en: {
      title: 'The four areas',
      body: 'Work is split into four tabs, in the natural order of a bibliometric analysis.',
      bullets: [
        '01 Main Information — import, indicators, rankings and visual analyses.',
        '02 Networks — co-occurrence, communities and international collaboration.',
        '03 Search Engine — the dossier of any author, country, venue, term or document.',
        '04 Report — a PDF or Word document with whatever you select.',
      ],
    },
  },
  {
    id: 'upload',
    Icon: Database,
    target: 'upload',
    tab: 'overview',
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
    tab: 'overview',
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
    id: 'ticker',
    Icon: Gauge,
    target: 'ticker',
    needsData: true,
    pt: {
      title: 'Faixa de indicadores',
      body: 'Os números-chave da base ficam sempre à vista no cabeçalho, em qualquer aba: bases de origem, documentos, período, autores, países, venues e crescimento. Clique na faixa para pausar a rolagem.',
    },
    en: {
      title: 'Indicator strip',
      body: 'The key numbers of the dataset stay visible in the header on every tab: source databases, documents, period, authors, countries, venues and growth. Click the strip to pause the scrolling.',
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
    id: 'launchers',
    Icon: Layers,
    target: 'launchers',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Aprofundar a análise',
      body: 'As análises mais pesadas ficam em blocos que abrem numa janela própria — e só são calculadas quando você abre. Vamos ver cada um.',
    },
    en: {
      title: 'Deeper analysis',
      body: 'The heavier analyses live in blocks that open in their own window — and are only computed when you open them. Let us look at each one.',
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
    id: 'launcher-visual',
    Icon: Radar,
    target: 'launcher-visual',
    tab: 'overview',
    needsData: true,
    pt: {
      title: 'Análises visuais avançadas',
      body: 'Sete visualizações clássicas da bibliometria, cada uma com seu "Como ler".',
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
      body: 'Seven classic bibliometric visualisations, each with its own "How to read".',
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
    id: 'launcher-tables',
    Icon: Table2,
    target: 'launcher-tables',
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
    id: 'search-picker',
    Icon: Search,
    target: 'search-picker',
    tab: 'search',
    needsData: true,
    prepare: openTopAuthor,
    pt: {
      title: 'Motor de Busca',
      body: 'Escolha o tipo de entidade — autor, país, venue, palavra-chave, tema ou documento — e digite para buscar. Todo nome clicável no Simetrics (barras, nós, países, linhas de tabela) também traz você para cá. Abrimos o autor de maior destaque da base como exemplo.',
    },
    en: {
      title: 'Search Engine',
      body: 'Pick an entity type — author, country, venue, keyword, theme or document — and type to search. Every clickable name in Simetrics (bars, nodes, countries, table rows) also brings you here. We opened the most prominent author in the dataset as an example.',
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
      body: 'O perfil reúne documentos, coautores, citações, índices h, g, i10 e m, período de atividade e a especialização temática. Os coautores e países também são clicáveis — dá para navegar de perfil em perfil.',
    },
    en: {
      title: 'Entity dossier',
      body: 'The profile gathers documents, co-authors, citations, h, g, i10 and m indices, active period and thematic specialisation. Co-authors and countries are clickable too — you can hop from profile to profile.',
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
    id: 'report-builder',
    Icon: FileText,
    target: 'report-builder',
    tab: 'report',
    needsData: true,
    pt: {
      title: 'Relatório executivo',
      body: 'Monte o relatório marcando as seções — resumo executivo, indicadores, rankings, gráficos, temas de IA, topologia da rede — e quantos itens entram em cada ranking. Gere em PDF diagramado ou Word (DOCX) editável, direto no navegador.',
    },
    en: {
      title: 'Executive report',
      body: 'Build the report by ticking sections — executive summary, indicators, rankings, charts, AI themes, network topology — and how many items each ranking includes. Generate a laid-out PDF or an editable Word (DOCX) file, right in the browser.',
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
      body: 'Uma prévia ao vivo do documento, atualizada a cada seção marcada ou desmarcada. O que você vê aqui é o que vai para o arquivo.',
    },
    en: {
      title: 'Preview',
      body: 'A live preview of the document, updated whenever you tick or untick a section. What you see here is what goes into the file.',
    },
  },
  {
    id: 'chat',
    Icon: Bot,
    target: 'chat',
    needsData: true,
    pt: {
      title: 'Simi, a assistente científica',
      body: 'Converse com a sua base: a Simi busca os documentos relevantes e responde citando-os. Está disponível em qualquer aba. Na primeira vez, informe a chave do seu provedor (Gemini, OpenAI, Claude, OpenRouter ou um modelo local) — ela fica só no seu navegador.',
    },
    en: {
      title: 'Simi, the scientific assistant',
      body: 'Chat with your dataset: Simi retrieves the relevant documents and answers citing them. It is available on every tab. The first time, enter your provider key (Gemini, OpenAI, Claude, OpenRouter or a local model) — it stays in your browser only.',
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
