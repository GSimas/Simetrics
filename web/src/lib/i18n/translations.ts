export type Locale = 'pt' | 'en';

export const TRANSLATIONS = {
  pt: {
    // Header & Brand
    app_title: 'Simetrics',
    app_subtitle: 'Plataforma de Inteligência Bibliométrica & Mapeamento Científico',
    app_version: 'v1.0',
    active_docs: 'documentos ativos',
    tutorial_btn: 'Como Usar / Tutorial',
    ai_settings_btn: 'Chave de IA (BYOK)',
    ai_configured: 'IA Configurada',
    ai_not_configured: 'Configurar IA',
    developed_by: 'Desenvolvido por',
    theme_dark: 'Alternar para Modo Claro',
    theme_light: 'Alternar para Modo Escuro',
    lang_toggle: 'Mudar para Inglês',
    buy_me_coffee: 'Pague-me um café',
    buy_me_coffee_tooltip: 'Apoie o Simetrics pagando um café! ☕',

    // Tabs
    tab_overview: 'Informações Principais',
    tab_networks: 'Redes',
    tab_search: 'Motor de Busca',
    tab_chat: 'Assistente Científico',
    tab_report: 'Relatório',
    tab_feedback: 'Feedback',

    // Upload Panel
    upload_title: 'Base de dados',
    upload_description:
      'Formatos aceitos: RIS (SciELO, WoS, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (WoS) e TXT/NBIB (PubMed). Limite de 10.000 documentos.',
    upload_select_files: 'Selecionar arquivos',
    upload_load_demo: 'Carregar exemplo',
    upload_loaded_count: 'documentos carregados',
    upload_clear: 'Limpar base',
    upload_confirm_sources: 'Confirme a base de origem de cada arquivo',
    upload_process_btn: 'Processar e integrar',
    upload_processing: 'Processando...',

    // Overview KPIs
    kpi_docs: 'Documentos',
    kpi_docs_sub: 'Período',
    kpi_authors: 'Autores',
    kpi_authors_sub: 'Produtores de conhecimento',
    kpi_countries: 'Países',
    kpi_countries_sub: 'Alcance geográfico',
    kpi_venues: 'Venues',
    kpi_venues_sub: 'Periódicos e eventos',
    kpi_growth: 'Crescimento anual',
    kpi_growth_sub: 'Taxa composta no período',
    kpi_citations_year: 'Citações por ano',
    kpi_citations_year_sub: 'Média por documento',
    kpi_collab: 'Colaboração internacional',
    kpi_collab_sub: 'de país único',
    kpi_authors_doc: 'Autores por documento',
    kpi_authors_doc_sub: 'com autor único',

    // Deduplication
    dedup_title: 'Deduplicação',
    dedup_description:
      'Bases diferentes indexam os mesmos artigos. O DOI é a evidência mais forte de identidade; a similaridade de título alcança os registros sem DOI, ao custo de algum risco de falso positivo.',
    dedup_strategy_label: 'Estratégia de deduplicação',
    dedup_none: 'Base completa',
    dedup_doi: 'Deduplicar por DOI',
    dedup_similarity: 'Deduplicar por similaridade',
    dedup_both: 'Deduplicação por ambas (DOI e similaridade)',
    dedup_execute_btn: 'Executar deduplicação',
    dedup_removed: 'documentos removidos',

    // Theme Panel (AI)
    theme_title: 'Mapeamento temático por IA',
    theme_description:
      'Os documentos são agrupados por similaridade textual no seu navegador (TF-IDF → LSA → K-Means, com o número de temas escolhido pelo Silhouette). Só as amostras de cada grupo vão ao modelo configurado em BYOK, que sintetiza o nome do tema.',
    theme_btn_identify: 'Identificar temas',
    theme_btn_recalc: 'Recategorizar temas',
    theme_clusters_found: 'temas identificados',
    theme_table_theme: 'Tema',
    theme_table_docs: 'Documentos',
    theme_table_top_author: 'Autor de maior QL',
    theme_table_top_country: 'País de maior QL',
    theme_table_top_venue: 'Venue de maior QL',
    theme_ql_explanation:
      'O Quociente Locacional acima de 1 indica especialização: a entidade publica naquele tema mais do que a média da base. O desempate entre QLs iguais é pelo volume, para que uma entidade com um único documento no tema não lidere.',
    theme_no_key_warning:
      'Chave de IA não configurada. Configure sua chave própria (Gemini, OpenAI, Claude, OpenRouter) para rotular os temas com IA ou use os rótulos automáticos.',

    // Visual Analyses
    visual_title: 'Análises visuais avançadas',
    visual_description:
      'Cada aba é calculada sob demanda, no seu navegador. Todos os gráficos exportam PNG pela barra de ferramentas.',
    visual_tab_boxplot: 'Distribuição',
    visual_tab_sankey: 'Evolução temática',
    visual_tab_genetics: 'Genética das ideias',
    visual_tab_concept: 'Mapa conceitual',
    visual_tab_thematic: 'Mapa temático',
    visual_tab_historiograph: 'Historiograph',

    // Production & Lotka
    prod_title: 'Produção ao longo do tempo',
    prod_description: 'Documentos publicados por ano.',
    prod_category_label: 'Categorizar por',
    prod_category_total: 'Total',
    prod_category_country: 'País',
    prod_category_database: 'Base de dados',
    prod_category_doctype: 'Tipo de trabalho',
    prod_category_theme: 'Temas (IA)',
    prod_category_theme_locked: 'mapeie os temas primeiro',
    prod_mode_label: 'Visualização',
    prod_mode_bars_grouped: 'Barras separadas',
    prod_mode_bars_stacked: 'Barras agrupadas',
    prod_mode_line: 'Linha',
    prod_empty_no_themes:
      'Nenhum tema mapeado ainda. Use o Mapeamento temático por IA acima para gerá-los.',
    prod_empty_generic: 'A base não traz esse dado para montar o gráfico.',
    lotka_title: 'Lei de Lotka',
    lotka_description:
      'Produtividade observada contra a distribuição teórica c/x². O afastamento da curva indica concentração atípica da autoria.',
    meta_quality_title: 'Qualidade e completude dos metadados',
    meta_quality_description:
      'Campos ausentes limitam o que a análise consegue enxergar — sem afiliação não há mapa de colaboração, sem referências não há rede de cocitação.',

    // Deep-dive Entity Tables
    tables_title: 'Tabelas analíticas',
    tables_description:
      'Índices h, g, i10 e m por entidade. Clique nos cabeçalhos para ordenar; exporte em CSV o recorte filtrado.',
    table_tab_all_docs: 'Todos os documentos',
    table_tab_authors: 'Autores',
    table_tab_countries: 'Países',
    table_tab_venues: 'Venues',
    table_tab_keywords: 'Palavras-chave',
    table_filter_placeholder: 'Filtrar...',
    table_export_csv: 'Exportar CSV',

    // Networks Tab
    network_deep_title: 'Ecologia profunda da rede',
    network_deep_desc: 'Grafo heterogêneo ligando documentos a autores, países e venues:',
    network_nodes: 'nós',
    network_edges: 'arestas',
    network_components: 'componentes',
    network_cooccurrence_title: 'Rede de coocorrência',
    network_cooccurrence_desc:
      'Entidades conectadas por aparecerem no mesmo documento. A espessura da aresta reflete a frequência da coocorrência; a cor, a comunidade detectada pelo Louvain.',
    network_kind_label: 'Tipo de rede',
    network_top_label: 'Entidades exibidas',
    network_size_label: 'Tamanho dos nós',
    network_collab_title: 'Colaboração internacional',
    network_collab_desc:
      'Dois países colaboram quando assinam o mesmo documento. A espessura da linha é o número de trabalhos em conjunto.',
    network_map_tab: 'Mapa-múndi',
    network_circular_tab: 'Grafo circular',
    network_nodes_metrics_title: 'Métricas por nó',
    network_nodes_metrics_desc: 'Todos os nós do grafo heterogêneo, ordenados por grau.',

    // Search Tab
    search_title: 'Motor de busca',
    search_desc:
      'Escolha uma entidade para montar seu dossiê: produção, impacto, documentos e perfis semelhantes.',
    search_type_label: 'Tipo',
    search_query_label: 'Buscar',
    search_placeholder: 'Digite para filtrar...',
    search_dossier_title: 'Dossiê Científico',
    search_similar_title: 'Entidades semelhantes',
    search_similar_desc:
      'Similaridade de Jaccard sobre o "DNA acadêmico": palavras-chave, coautores e veículos em comum.',
    search_lexico_title: 'Lexicometria e Produção',
    search_lexico_desc: 'Palavras-chave e evolução temporal das publicações desta entidade.',
    search_tab_cloud: 'Nuvem de palavras',
    search_tab_timeline: 'Produção histórica',
    search_timeline_title: 'Evolução anual de publicações',
    search_timeline_docs: 'Publicações',
    search_docs_title: 'Documentos',
    search_docs_desc: 'Ordenados por citações, do mais citado ao menos citado. Clique no título para ver o dossiê do documento.',
    search_country_authors: 'Autores vinculados ao país',
    search_country_authors_desc: 'Pesquisadores com publicações vinculadas a este país. Clique no autor para abrir seu dossiê.',
    search_doc_authors: 'Autores desta publicação',
    search_doc_authors_desc: 'Autores que assinam este documento e suas métricas na base. Clique no autor para abrir seu dossiê.',
    search_view_author: 'Ver perfil do autor',
    search_view_doc: 'Ver dossiê do documento',
    search_abstract: 'Resumo',
    search_keywords: 'Palavras-chave',
    search_doi_link: 'Acessar DOI / Link',

    // Chat Tab
    chat_title: 'Simi - Assistente Científico',
    chat_desc:
      'Conversando com seus documentos. A cada pergunta, os documentos mais relevantes são selecionados no seu navegador via BM25 e enviados com segurança ao modelo configurado.',
    chat_greeting:
      'Olá! Sou a Simi, sua assistente científica no Simetrics. Respondo com base nos documentos da sua base carregada. Posso recomendar leituras fundamentais, identificar especialistas ou sugerir periódicos para submissão. O que você gostaria de investigar?',
    chat_suggestions_label: 'Sugestões de perguntas:',
    chat_sugg_1: 'Quais são os documentos fundamentais desta base?',
    chat_sugg_2: 'Quem são os autores mais influentes e em que eles trabalham?',
    chat_sugg_3: 'Em quais periódicos eu deveria submeter um artigo sobre este tema?',
    chat_sugg_4: 'Que lacunas de pesquisa aparecem nesta literatura?',
    chat_placeholder: 'Ex.: quais são os documentos fundamentais sobre este tema?',
    chat_btn_send: 'Enviar',
    chat_btn_stop: 'Parar',
    chat_analyzing: 'Analisando a base bibliométrica...',
    chat_status_querying_table: 'Consultando tabela analítica local ({table})...',
    chat_status_filtering_docs: 'Filtrando e agregando estatísticas da base...',
    chat_status_general_metrics: 'Calculando indicadores bibliométricos globais...',
    chat_status_entity_profile: 'Analisando perfil detalhado de {entity}...',
    chat_tool_executed: 'Consulta analítica executada na base local',
    chat_tools_executed_count: 'consultas analíticas locais realizadas',
    chat_tools_badge: 'Cálculo determinístico no navegador',
    chat_no_key_warning:
      'Para conversar com a Simi, configure sua chave de API própria (BYOK) no botão acima.',

    // Feedback Tab
    feedback_title: 'Avaliação da Plataforma (SUS)',
    feedback_desc:
      'Este questionário avalia a sua experiência com o Simetrics. Não há respostas certas ou erradas — estamos avaliando o sistema. As respostas são anônimas.',
    feedback_part1: 'Parte 1 · Perfil do participante',
    feedback_titulacao: 'Titulação',
    feedback_area: 'Área de atuação',
    feedback_experiencia: 'Experiência prévia',
    feedback_part2: 'Parte 2 · Questionário de usabilidade',
    feedback_part2_desc: 'Para cada afirmação, marque de 1 (discordo totalmente) a 5 (concordo totalmente).',
    feedback_part3: 'Parte 3 · Interface e experiência',
    feedback_submit_btn: 'Enviar avaliação',
    feedback_success_title: 'Avaliação registrada',
    feedback_success_desc: 'Obrigado por dedicar seu tempo. Suas respostas orientam as próximas melhorias.',

    // Empty States
    empty_start_title: 'Comece por aqui',
    empty_start_desc:
      'O Simetrics transforma exports de bases bibliográficas em indicadores cientométricos, redes de conhecimento e mapeamento temático. Envie seus arquivos acima ou carregue a base de exemplo para explorar.',
    empty_client_note:
      'Todo o processamento acontece no seu navegador — os documentos não são enviados para nenhum servidor.',
    empty_generic_desc:
      'Carregue uma base de dados na aba Informações Principais para liberar esta análise.',

    // AI Settings Modal (BYOK)
    ai_modal_title: 'Configurações de IA (Bring Your Own Key)',
    ai_modal_subtitle: 'Utilize sua própria chave de API para habilitar os recursos generativos',
    ai_provider_label: 'Provedor de IA',
    ai_api_key_label: 'Chave de API (API Key)',
    ai_api_key_placeholder: 'Insira sua chave de API (ex: AIzaSy... / sk-...)',
    ai_model_label: 'Modelo',
    ai_base_url_label: 'URL Base do Endpoint (Opcional)',
    ai_test_btn: 'Testar Conexão',
    ai_save_btn: 'Salvar Configuração',
    ai_clear_btn: 'Remover Chave',
    ai_privacy_note:
      'Privacidade garantida: Sua chave de API é salva exclusivamente no armazenamento local do seu navegador (localStorage) e enviada diretamente para a API do provedor escolhido. Nossos servidores não têm acesso à sua chave.',
    ai_test_success: 'Conexão testada com sucesso!',
    ai_test_failed: 'Falha no teste de conexão:',

    // Navegação — Projetos
    nav_projects_btn: 'Meus Projetos',
    project_save_status_saving: 'Salvando…',
    project_save_status_saved: 'Salvo às {time}',
    project_save_status_error: 'Falha ao salvar',

    // Tela Inicial (Landing)
    landing_pitch:
      'O Simetrics transforma exports de bases bibliográficas (Scopus, Web of Science, SciELO e mais) em indicadores cientométricos, redes de colaboração e mapeamento temático por Inteligência Artificial — tudo processado no seu navegador, sem enviar seus dados a nenhum servidor.',
    landing_highlight_1_label: '100% no seu navegador',
    landing_highlight_1_text:
      'Seus dados nunca saem do seu computador. Todo o processamento roda via Web Workers, direto no navegador.',
    landing_highlight_2_label: 'Ecologia do Conhecimento',
    landing_highlight_2_text:
      'Mapeie a estrutura intelectual da sua base através de grafos, métricas de rede, PCA e similaridade.',
    landing_highlight_3_label: 'Inteligência Artificial',
    landing_highlight_3_text:
      'Categorização temática automática e assistente conversacional sobre a sua base de artigos.',
    landing_cta_continue: 'Continuar "{name}"',
    landing_cta_new_blank: 'Novo projeto em branco',
    landing_cta_start: 'Iniciar',
    landing_cta_tutorial: 'Ver tutorial',
    landing_projects_title: 'Meus Projetos',
    landing_projects_import: 'Importar projeto (.json)',
    landing_dismiss_error: 'Dispensar',
    landing_projects_empty_title: 'Nenhum projeto salvo ainda',
    landing_projects_empty_desc:
      'Carregue uma base para começar — o Simetrics salva seu progresso automaticamente neste navegador.',

    // Landing — Perguntas frequentes (FAQ)
    landing_faq_title: 'Perguntas frequentes',
    landing_faq_q1: 'O que é o Simetrics?',
    landing_faq_a1:
      'Uma plataforma gratuita de bibliometria e cientometria. Ela transforma exports de bases acadêmicas (Scopus, Web of Science, SciELO e outras) em indicadores cientométricos, redes de colaboração e mapeamento temático por IA.',
    landing_faq_q2: 'O Simetrics é gratuito?',
    landing_faq_a2:
      'Sim, totalmente gratuito e sem necessidade de cadastro. Os recursos de IA (categorização temática, assistente científico) são opcionais e usam a chave de API do próprio usuário (BYOK).',
    landing_faq_q3: 'Meus dados ficam salvos onde? É seguro enviar minha base?',
    landing_faq_a3:
      'Todo o processamento acontece no seu navegador — os documentos nunca são enviados a nenhum servidor. Os projetos que você salva ficam armazenados localmente, neste navegador e neste dispositivo, via IndexedDB.',
    landing_faq_q4: 'Quais formatos de arquivo o Simetrics aceita?',
    landing_faq_a4:
      'RIS (SciELO, Web of Science, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (Web of Science) e TXT/NBIB (PubMed).',
    landing_faq_q5: 'Preciso de conta ou chave de API para usar?',
    landing_faq_a5:
      'Não. A análise bibliométrica completa funciona sem conta e sem chave de API. Uma chave própria (Gemini, OpenAI, Claude ou OpenRouter) só é necessária para os recursos opcionais de IA.',

    // Cartão de Projeto
    project_card_open: 'Abrir',
    project_card_rename: 'Renomear',
    project_card_rename_save: 'Salvar nome',
    project_card_rename_cancel: 'Cancelar renomeação',
    project_card_duplicate: 'Duplicar',
    project_card_export: 'Exportar (.json)',
    project_card_delete: 'Excluir',
    project_card_docs: '{count} documentos',
    project_card_updated: 'Atualizado em {date}',
    project_delete_confirm_title: 'Excluir projeto?',
    project_delete_confirm_desc: 'Isso excluirá permanentemente "{name}" deste navegador. Esta ação não pode ser desfeita.',
    project_delete_cancel: 'Cancelar',

    // Projetos — mensagens do sistema
    project_untitled: 'Projeto sem título',
    project_copy_suffix: '(cópia)',
    project_import_suffix: '(importado)',
    project_save_quota_error: 'Armazenamento local cheio — exporte este projeto em JSON e apague um projeto antigo para liberar espaço.',
    project_save_error: 'Não foi possível salvar o projeto neste navegador.',
    project_busy_error: 'Aguarde a operação atual terminar antes de trocar de projeto.',
    project_not_found_error: 'Projeto não encontrado — pode ter sido excluído.',
    project_import_invalid_json: 'Arquivo inválido: selecione um arquivo .json exportado do Simetrics.',
  },
  en: {
    // Header & Brand
    app_title: 'Simetrics',
    app_subtitle: 'Bibliometric Intelligence & Scientific Mapping Platform',
    app_version: 'v1.0',
    active_docs: 'active documents',
    tutorial_btn: 'How to Use / Tutorial',
    ai_settings_btn: 'AI API Key (BYOK)',
    ai_configured: 'AI Configured',
    ai_not_configured: 'Setup AI Key',
    developed_by: 'Developed by',
    theme_dark: 'Switch to Light Mode',
    theme_light: 'Switch to Dark Mode',
    lang_toggle: 'Mudar para Português',
    buy_me_coffee: 'Buy me a coffee',
    buy_me_coffee_tooltip: 'Support Simetrics by buying me a coffee! ☕',

    // Tabs
    tab_overview: 'Overview & Metrics',
    tab_networks: 'Networks',
    tab_search: 'Search Engine',
    tab_chat: 'Scientific Assistant',
    tab_report: 'Report',
    tab_feedback: 'Feedback',

    // Upload Panel
    upload_title: 'Bibliographic Database',
    upload_description:
      'Accepted formats: RIS (SciELO, WoS, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (WoS), and TXT/NBIB (PubMed). Up to 10,000 documents limit.',
    upload_select_files: 'Select files',
    upload_load_demo: 'Load demo dataset',
    upload_loaded_count: 'documents loaded',
    upload_clear: 'Clear dataset',
    upload_confirm_sources: 'Confirm source database for each file',
    upload_process_btn: 'Process and integrate',
    upload_processing: 'Processing...',

    // Overview KPIs
    kpi_docs: 'Documents',
    kpi_docs_sub: 'Timespan',
    kpi_authors: 'Authors',
    kpi_authors_sub: 'Knowledge producers',
    kpi_countries: 'Countries',
    kpi_countries_sub: 'Geographic coverage',
    kpi_venues: 'Venues',
    kpi_venues_sub: 'Journals and events',
    kpi_growth: 'Annual growth',
    kpi_growth_sub: 'Compound rate in timespan',
    kpi_citations_year: 'Citations per year',
    kpi_citations_year_sub: 'Average per document',
    kpi_collab: 'International collab',
    kpi_collab_sub: 'single country',
    kpi_authors_doc: 'Authors per doc',
    kpi_authors_doc_sub: 'single authored',

    // Deduplication
    dedup_title: 'Deduplication',
    dedup_description:
      'Different databases index the same papers. DOI is the strongest identity evidence; title similarity catches records without DOI at some risk of false positives.',
    dedup_strategy_label: 'Deduplication strategy',
    dedup_none: 'Full dataset',
    dedup_doi: 'Deduplicate by DOI',
    dedup_similarity: 'Deduplicate by similarity',
    dedup_both: 'Deduplicate by both (DOI and similarity)',
    dedup_execute_btn: 'Run deduplication',
    dedup_removed: 'documents removed',

    // Theme Panel (AI)
    theme_title: 'AI Topic Mapping',
    theme_description:
      'Documents are clustered by textual similarity in your browser (TF-IDF → LSA → K-Means, with cluster count chosen by Silhouette). Only samples are sent to your configured BYOK model to synthesize the theme label.',
    theme_btn_identify: 'Identify themes',
    theme_btn_recalc: 'Re-categorize themes',
    theme_clusters_found: 'themes identified',
    theme_table_theme: 'Theme',
    theme_table_docs: 'Documents',
    theme_table_top_author: 'Top LQ Author',
    theme_table_top_country: 'Top LQ Country',
    theme_table_top_venue: 'Top LQ Venue',
    theme_ql_explanation:
      'A Locational Quotient (LQ) above 1 indicates specialization: the entity publishes in that theme more than the dataset average. Tiebreaks are resolved by volume.',
    theme_no_key_warning:
      'AI API Key not configured. Configure your own key (Gemini, OpenAI, Claude, OpenRouter) to label themes with AI or use automatic labels.',

    // Visual Analyses
    visual_title: 'Advanced Visual Analyses',
    visual_description:
      'Each panel is computed on demand in your browser. All charts support PNG export from the toolbar.',
    visual_tab_boxplot: 'Distribution',
    visual_tab_sankey: 'Thematic Evolution',
    visual_tab_genetics: 'Genetics of Ideas',
    visual_tab_concept: 'Concept Map',
    visual_tab_thematic: 'Thematic Map',
    visual_tab_historiograph: 'Historiograph',

    // Production & Lotka
    prod_title: 'Production Over Time',
    prod_description: 'Published documents per year.',
    prod_category_label: 'Break down by',
    prod_category_total: 'Total',
    prod_category_country: 'Country',
    prod_category_database: 'Database',
    prod_category_doctype: 'Document type',
    prod_category_theme: 'Themes (AI)',
    prod_category_theme_locked: 'map themes first',
    prod_mode_label: 'Chart type',
    prod_mode_bars_grouped: 'Grouped bars',
    prod_mode_bars_stacked: 'Stacked bars',
    prod_mode_line: 'Line',
    prod_empty_no_themes: 'No themes mapped yet. Use the AI thematic mapping above to generate them.',
    prod_empty_generic: "The dataset doesn't carry this data to build the chart.",
    lotka_title: "Lotka's Law",
    lotka_description:
      'Observed productivity against theoretical c/x² distribution. Curve divergence indicates atypical author concentration.',
    meta_quality_title: 'Metadata Quality & Completeness',
    meta_quality_description:
      'Missing fields limit analysis scope — without affiliation there is no collaboration map, without references there is no co-citation network.',

    // Deep-dive Entity Tables
    tables_title: 'Analytical Tables',
    tables_description:
      'Indices h, g, i10, and m per entity. Click headers to sort; export the filtered selection to CSV.',
    table_tab_all_docs: 'All Documents',
    table_tab_authors: 'Authors',
    table_tab_countries: 'Countries',
    table_tab_venues: 'Venues',
    table_tab_keywords: 'Keywords',
    table_filter_placeholder: 'Filter...',
    table_export_csv: 'Export CSV',

    // Networks Tab
    network_deep_title: 'Deep Network Ecology',
    network_deep_desc: 'Heterogeneous graph connecting documents to authors, countries, and venues:',
    network_nodes: 'nodes',
    network_edges: 'edges',
    network_components: 'components',
    network_cooccurrence_title: 'Co-occurrence Network',
    network_cooccurrence_desc:
      'Entities connected by appearing in the same document. Edge thickness reflects co-occurrence frequency; color indicates Louvain detected communities.',
    network_kind_label: 'Network type',
    network_top_label: 'Displayed entities',
    network_size_label: 'Node sizing',
    network_collab_title: 'International Collaboration',
    network_collab_desc:
      'Two countries collaborate when they co-author a paper. Line thickness indicates joint papers.',
    network_map_tab: 'World Map',
    network_circular_tab: 'Chordal Graph',
    network_nodes_metrics_title: 'Node Metrics',
    network_nodes_metrics_desc: 'All nodes from the heterogeneous graph sorted by degree.',

    // Search Tab
    search_title: 'Search Engine',
    search_desc:
      'Pick an entity to generate its dossier: scientific output, citation impact, documents, and similar profiles.',
    search_type_label: 'Type',
    search_query_label: 'Search',
    search_placeholder: 'Type to filter...',
    search_dossier_title: 'Scientific Dossier',
    search_similar_title: 'Similar Entities',
    search_similar_desc:
      'Jaccard similarity over "Academic DNA": shared keywords, co-authors, and publishing venues.',
    search_lexico_title: 'Lexicometrics & Production',
    search_lexico_desc: 'Keywords and historical publication output for this entity.',
    search_tab_cloud: 'Word Cloud',
    search_tab_timeline: 'Historical Production',
    search_timeline_title: 'Annual Publication Output',
    search_timeline_docs: 'Publications',
    search_docs_title: 'Documents',
    search_docs_desc: 'Sorted by citations from most to least cited. Click a title to view its dossier.',
    search_country_authors: 'Authors Affiliated with Country',
    search_country_authors_desc: 'Researchers with publications affiliated with this country. Click an author to open their dossier.',
    search_doc_authors: 'Authors of this Publication',
    search_doc_authors_desc: 'Authors of this document and their metrics in the dataset. Click an author to open their dossier.',
    search_view_author: 'View author profile',
    search_view_doc: 'View document dossier',
    search_abstract: 'Abstract',
    search_keywords: 'Keywords',
    search_doi_link: 'Access DOI / Link',

    // Chat Tab
    chat_title: 'Simi - Scientific Assistant',
    chat_desc:
      'Chatting with your dataset. For each query, the most relevant documents are selected in your browser via BM25 and safely sent to your configured AI model.',
    chat_greeting:
      "Hello! I'm Simi, your scientific assistant in Simetrics. I answer based on the documents in your loaded dataset. I can recommend foundational readings, identify research leaders, or suggest journals for submission. What would you like to explore?",
    chat_suggestions_label: 'Suggested questions:',
    chat_sugg_1: 'What are the foundational papers in this dataset?',
    chat_sugg_2: 'Who are the most influential authors and what are they working on?',
    chat_sugg_3: 'Which journals should I consider submitting an article on this topic to?',
    chat_sugg_4: 'What research gaps appear in this literature?',
    chat_placeholder: 'E.g., what are the foundational papers on this topic?',
    chat_btn_send: 'Send',
    chat_btn_stop: 'Stop',
    chat_analyzing: 'Analyzing bibliometric database...',
    chat_status_querying_table: 'Querying local analytical table ({table})...',
    chat_status_filtering_docs: 'Filtering and aggregating dataset statistics...',
    chat_status_general_metrics: 'Calculating global bibliometric indicators...',
    chat_status_entity_profile: 'Analyzing detailed profile for {entity}...',
    chat_tool_executed: 'Analytical query executed on local dataset',
    chat_tools_executed_count: 'local analytical queries executed',
    chat_tools_badge: 'Deterministic in-browser calculation',
    chat_no_key_warning:
      'To chat with Simi, please configure your own API key (BYOK) using the button above.',

    // Feedback Tab
    feedback_title: 'Platform Usability Scale (SUS)',
    feedback_desc:
      'This questionnaire assesses your experience with Simetrics. There are no right or wrong answers. Responses are anonymous.',
    feedback_part1: 'Part 1 · Participant Profile',
    feedback_titulacao: 'Academic Level',
    feedback_area: 'Research Field',
    feedback_experiencia: 'Prior Experience',
    feedback_part2: 'Part 2 · Usability Questionnaire',
    feedback_part2_desc: 'For each statement, rate from 1 (Strongly disagree) to 5 (Strongly agree).',
    feedback_part3: 'Part 3 · Interface & User Experience',
    feedback_submit_btn: 'Submit feedback',
    feedback_success_title: 'Feedback Registered',
    feedback_success_desc: 'Thank you for your time. Your feedback directly guides upcoming improvements.',

    // Empty States
    empty_start_title: 'Start Here',
    empty_start_desc:
      'Simetrics transforms exports from bibliographic databases into scientometric indicators, knowledge graphs, and AI-driven thematic mapping. Upload your files above or load the demo dataset to explore.',
    empty_client_note:
      'All processing runs entirely inside your browser — documents are never uploaded to any remote server.',
    empty_generic_desc:
      'Load a dataset in the Overview tab to unlock this analysis.',

    // AI Settings Modal (BYOK)
    ai_modal_title: 'AI Settings (Bring Your Own Key)',
    ai_modal_subtitle: 'Use your own API key to enable generative AI features',
    ai_provider_label: 'AI Provider',
    ai_api_key_label: 'API Key',
    ai_api_key_placeholder: 'Enter your API key (e.g. AIzaSy... / sk-...)',
    ai_model_label: 'Model Name',
    ai_base_url_label: 'Custom Base URL (Optional)',
    ai_test_btn: 'Test Connection',
    ai_save_btn: 'Save Settings',
    ai_clear_btn: 'Remove Key',
    ai_privacy_note:
      'Privacy guaranteed: Your API key is stored exclusively in your browser local storage (localStorage) and sent directly to the selected provider API endpoint. Our servers never see or touch your key.',
    ai_test_success: 'Connection tested successfully!',
    ai_test_failed: 'Connection test failed:',

    // Navigation — Projects
    nav_projects_btn: 'My Projects',
    project_save_status_saving: 'Saving…',
    project_save_status_saved: 'Saved at {time}',
    project_save_status_error: 'Failed to save',

    // Landing Screen
    landing_pitch:
      'Simetrics turns exports from bibliographic databases (Scopus, Web of Science, SciELO, and more) into scientometric indicators, collaboration networks, and AI-driven thematic mapping — all processed in your browser, without sending your data to any server.',
    landing_highlight_1_label: '100% in your browser',
    landing_highlight_1_text:
      'Your data never leaves your computer. All processing runs via Web Workers, right in the browser.',
    landing_highlight_2_label: 'Knowledge Ecology',
    landing_highlight_2_text:
      'Map the intellectual structure of your dataset through graphs, network metrics, PCA, and similarity.',
    landing_highlight_3_label: 'Artificial Intelligence',
    landing_highlight_3_text:
      'Automatic thematic categorization and a conversational assistant over your article dataset.',
    landing_cta_continue: 'Continue "{name}"',
    landing_cta_new_blank: 'New blank project',
    landing_cta_start: 'Get Started',
    landing_cta_tutorial: 'Watch the tour',
    landing_projects_title: 'My Projects',
    landing_projects_import: 'Import project (.json)',
    landing_dismiss_error: 'Dismiss',
    landing_projects_empty_title: 'No saved projects yet',
    landing_projects_empty_desc:
      'Load a dataset to get started — Simetrics saves your progress automatically in this browser.',

    // Landing — Frequently Asked Questions (FAQ)
    landing_faq_title: 'Frequently asked questions',
    landing_faq_q1: 'What is Simetrics?',
    landing_faq_a1:
      'A free bibliometric and scientometric analysis platform. It turns exports from academic databases (Scopus, Web of Science, SciELO, and others) into scientometric indicators, collaboration networks, and AI-driven thematic mapping.',
    landing_faq_q2: 'Is Simetrics free?',
    landing_faq_a2:
      'Yes, completely free and no account required. The AI features (thematic categorization, scientific assistant) are optional and use the user’s own API key (BYOK).',
    landing_faq_q3: 'Where is my data stored? Is it safe to upload my dataset?',
    landing_faq_a3:
      'All processing happens in your browser — documents are never sent to any server. Projects you save are stored locally, in this browser and on this device, via IndexedDB.',
    landing_faq_q4: 'Which file formats does Simetrics accept?',
    landing_faq_a4:
      'RIS (SciELO, Web of Science, Scopus, Mendeley, Cochrane), CSV (Scopus, Cochrane), Excel (Web of Science), and TXT/NBIB (PubMed).',
    landing_faq_q5: 'Do I need an account or an API key to use it?',
    landing_faq_a5:
      'No. Full bibliometric analysis works without an account or an API key. Your own key (Gemini, OpenAI, Claude, or OpenRouter) is only needed for the optional AI features.',

    // Project Card
    project_card_open: 'Open',
    project_card_rename: 'Rename',
    project_card_rename_save: 'Save name',
    project_card_rename_cancel: 'Cancel rename',
    project_card_duplicate: 'Duplicate',
    project_card_export: 'Export (.json)',
    project_card_delete: 'Delete',
    project_card_docs: '{count} documents',
    project_card_updated: 'Updated {date}',
    project_delete_confirm_title: 'Delete project?',
    project_delete_confirm_desc: 'This will permanently delete "{name}" from this browser. This action cannot be undone.',
    project_delete_cancel: 'Cancel',

    // Projects — system messages
    project_untitled: 'Untitled project',
    project_copy_suffix: '(copy)',
    project_import_suffix: '(imported)',
    project_save_quota_error: 'Local storage is full — export this project as JSON and delete an older project to free up space.',
    project_save_error: 'Could not save the project in this browser.',
    project_busy_error: 'Wait for the current operation to finish before switching projects.',
    project_not_found_error: 'Project not found — it may have been deleted.',
    project_import_invalid_json: 'Invalid file: please select a .json file exported from Simetrics.',
  },
} as const;

export type TranslationKey = keyof typeof TRANSLATIONS.pt;
