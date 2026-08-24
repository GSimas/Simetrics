import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { authorsTable, countriesTable, keywordsTable, venuesTable, type EntityRow } from './tables';
import { summarize, docsPerYear } from './summary';
import { mean, median, pyRound, sum } from './stats';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from './text';

/**
 * Definição abstrata de ferramenta agnóstica a provedor.
 */
export interface ToolProperty {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description: string;
  enum?: string[];
  items?: { type: string };
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, ToolProperty>;
    required?: string[];
  };
}

/**
 * Catálogo de ferramentas analíticas disponíveis para a IA.
 */
export const ANALYTICAL_TOOLS: ToolDefinition[] = [
  {
    name: 'query_analytical_table',
    description:
      'Consulta tabelas analíticas exatas da base de dados bibliométrica (autores, países, periódicos/venues, palavras-chave) com ordenação, filtros de texto e métricas cienciométricas (índice H, G, i10, citações, etc.). Use sempre que o usuário perguntar sobre quem mais publica, mais citado, métricas de autores, países ou periódicos.',
    parameters: {
      type: 'object',
      properties: {
        table: {
          type: 'string',
          description: 'Qual tabela analítica consultar.',
          enum: ['authors', 'countries', 'venues', 'keywords'],
        },
        sortBy: {
          type: 'string',
          description: 'Critério de ordenação.',
          enum: ['citations', 'documents', 'h_index', 'g_index', 'i10_index', 'mean_citations'],
        },
        sortOrder: {
          type: 'string',
          description: 'Direção da ordenação (desc = maior para menor, asc = menor para maior).',
          enum: ['desc', 'asc'],
        },
        limit: {
          type: 'number',
          description: 'Número máximo de registros a retornar (1 a 50). Padrão: 15.',
        },
        filterText: {
          type: 'string',
          description:
            'Termo opcional para filtrar pelo nome da entidade, coautores ou país (ex: "Brazil", "Silva", "Nature", "Machine Learning").',
        },
        minDocs: {
          type: 'number',
          description: 'Número mínimo de documentos necessários para incluir o registro.',
        },
        minCitations: {
          type: 'number',
          description: 'Número mínimo de citações totais necessárias para incluir o registro.',
        },
      },
      required: ['table'],
    },
  },
  {
    name: 'filter_and_aggregate_documents',
    description:
      'Filtra a base de dados por critérios múltiplos combinados (ano inicial/final, tema, autor, país, periódico, palavras-chave, citações mínimas) e calcula estatísticas matemáticas exatas (total de docs, soma e média de citações, lista dos artigos encontrados).',
    parameters: {
      type: 'object',
      properties: {
        yearMin: {
          type: 'number',
          description: 'Ano mínimo de publicação (ex: 2018).',
        },
        yearMax: {
          type: 'number',
          description: 'Ano máximo de publicação (ex: 2023).',
        },
        theme: {
          type: 'string',
          description: 'Nome do tema temático para filtrar (ex: cluster temático).',
        },
        author: {
          type: 'string',
          description: 'Nome (ou parte do nome) do autor para filtrar.',
        },
        country: {
          type: 'string',
          description: 'País de afiliação para filtrar.',
        },
        venue: {
          type: 'string',
          description: 'Nome ou sigla do periódico/conferência para filtrar.',
        },
        keyword: {
          type: 'string',
          description: 'Palavra-chave para filtrar.',
        },
        minCitations: {
          type: 'number',
          description: 'Filtrar apenas artigos com no mínimo este número de citações.',
        },
        limit: {
          type: 'number',
          description: 'Número máximo de artigos representativos a detalhar no retorno (1 a 25). Padrão: 10.',
        },
      },
    },
  },
  {
    name: 'get_dataset_general_metrics',
    description:
      'Obtém o panorama macrobibliométrico da base de dados: taxa de crescimento anual (CAGR), índice de colaboração internacional (MCP/SCP), índice de coautoria, idade média dos documentos, produção e citações por ano, e distribuição temática.',
    parameters: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'get_entity_profile',
    description:
      'Obtém o raio-X / perfil aprofundado de um autor, país, periódico ou tema específico: métricas cienciométricas completas, lista cronológica de todas as suas publicações na base, principais colaboradores/coautores e índice de especialização (Quociente Locacional).',
    parameters: {
      type: 'object',
      properties: {
        entityType: {
          type: 'string',
          description: 'Tipo da entidade a analisar.',
          enum: ['author', 'country', 'venue', 'theme'],
        },
        name: {
          type: 'string',
          description: 'Nome da entidade (ex: "Silva, A.", "Brazil", "Scientometrics", "Inteligência Artificial").',
        },
      },
      required: ['entityType', 'name'],
    },
  },
];

/* =========================================================================
 * Adaptadores de Formato de Ferramentas por Provedor
 * ========================================================================= */

/** Converte ferramentas para o formato do Google Gemini (REST API) */
export function toGeminiTools(tools: ToolDefinition[] = ANALYTICAL_TOOLS) {
  return [
    {
      function_declarations: tools.map((t) => {
        const properties: Record<string, unknown> = {};
        for (const [key, prop] of Object.entries(t.parameters.properties)) {
          properties[key] = {
            type: prop.type.toUpperCase(),
            description: prop.description,
            ...(prop.enum ? { enum: prop.enum } : {}),
            ...(prop.items ? { items: { type: prop.items.type.toUpperCase() } } : {}),
          };
        }

        return {
          name: t.name,
          description: t.description,
          parameters: {
            type: 'OBJECT',
            properties,
            ...(t.parameters.required ? { required: t.parameters.required } : {}),
          },
        };
      }),
    },
  ];
}

/** Converte ferramentas para o formato OpenAI / OpenRouter / Custom (JSON Schema) */
export function toOpenAiTools(tools: ToolDefinition[] = ANALYTICAL_TOOLS) {
  return tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: {
        type: 'object',
        properties: t.parameters.properties,
        ...(t.parameters.required ? { required: t.parameters.required } : {}),
      },
    },
  }));
}

/** Converte ferramentas para o formato Anthropic Claude */
export function toClaudeTools(tools: ToolDefinition[] = ANALYTICAL_TOOLS) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: {
      type: 'object',
      properties: t.parameters.properties,
      ...(t.parameters.required ? { required: t.parameters.required } : {}),
    },
  }));
}

/* =========================================================================
 * Motor de Execução Local Determinístico
 * ========================================================================= */

export interface ToolExecutionResponse {
  toolName: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

export function executeAnalyticalTool(
  rawToolName: string,
  args: Record<string, unknown>,
  dataset: Dataset,
): ToolExecutionResponse {
  const toolName = String(rawToolName ?? '').replace(/^.*:/, '').trim();
  try {
    switch (toolName) {
      case 'query_analytical_table':
        return {
          toolName,
          success: true,
          result: executeQueryAnalyticalTable(args, dataset),
        };

      case 'filter_and_aggregate_documents':
        return {
          toolName,
          success: true,
          result: executeFilterAndAggregate(args, dataset),
        };

      case 'get_dataset_general_metrics':
        return {
          toolName,
          success: true,
          result: executeGetGeneralMetrics(dataset),
        };

      case 'get_entity_profile':
        return {
          toolName,
          success: true,
          result: executeGetEntityProfile(args, dataset),
        };

      default:
        return {
          toolName,
          success: false,
          error: `Ferramenta desconhecida: ${toolName}`,
        };
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    return {
      toolName,
      success: false,
      error: `Erro ao executar ${toolName}: ${errorMsg}`,
    };
  }
}

/** Executa query na tabela analítica solicitada */
function executeQueryAnalyticalTable(
  args: Record<string, unknown>,
  dataset: Dataset,
): Record<string, unknown> {
  const table = String(args.table ?? 'authors').toLowerCase();
  const sortBy = String(args.sortBy ?? 'citations').toLowerCase();
  const sortOrder = String(args.sortOrder ?? 'desc').toLowerCase();
  const limit = Math.min(Math.max(Number(args.limit ?? 15) || 15, 1), 50);
  const filterText = args.filterText ? String(args.filterText).trim().toLowerCase() : '';
  const minDocs = Number(args.minDocs ?? 0) || 0;
  const minCitations = Number(args.minCitations ?? 0) || 0;

  let rawRows: EntityRow[] = [];
  if (table === 'authors') rawRows = authorsTable(dataset);
  else if (table === 'countries') rawRows = countriesTable(dataset);
  else if (table === 'venues') rawRows = venuesTable(dataset);
  else if (table === 'keywords') rawRows = keywordsTable(dataset);
  else throw new Error(`Tabela inválida: ${table}. Use authors, countries, venues ou keywords.`);

  // Filtros
  let filtered = rawRows;
  if (filterText) {
    filtered = filtered.filter((r) => {
      const matchEntity = r.entity.toLowerCase().includes(filterText);
      const matchCoauthors = r.coauthors?.some((c) => c.toLowerCase().includes(filterText));
      const matchCountries = r.countries?.some((c) => c.toLowerCase().includes(filterText));
      const matchTimeline = r.timeline?.toLowerCase().includes(filterText);
      return matchEntity || matchCoauthors || matchCountries || matchTimeline;
    });
  }

  if (minDocs > 0) {
    filtered = filtered.filter((r) => r.docCount >= minDocs);
  }
  if (minCitations > 0) {
    filtered = filtered.filter((r) => r.citations >= minCitations);
  }

  // Ordenação
  const sorted = [...filtered].sort((a, b) => {
    let valA = 0;
    let valB = 0;

    if (sortBy === 'documents') {
      valA = a.docCount;
      valB = b.docCount;
    } else if (sortBy === 'h_index') {
      valA = a.h;
      valB = b.h;
    } else if (sortBy === 'g_index') {
      valA = a.g;
      valB = b.g;
    } else if (sortBy === 'i10_index') {
      valA = a.i10;
      valB = b.i10;
    } else if (sortBy === 'mean_citations') {
      valA = a.meanCitations;
      valB = b.meanCitations;
    } else {
      // default: citations
      valA = a.citations;
      valB = b.citations;
    }

    if (valA === valB) {
      // desempate por citações ou docCount
      return b.citations - a.citations || b.docCount - a.docCount;
    }

    return sortOrder === 'asc' ? valA - valB : valB - valA;
  });

  const sliced = sorted.slice(0, limit);

  return {
    table,
    totalEntitiesInTable: rawRows.length,
    matchingEntitiesCount: filtered.length,
    returnedCount: sliced.length,
    sortBy,
    sortOrder,
    rows: sliced.map((r) => ({
      entity: r.entity,
      docCount: r.docCount,
      citations: r.citations,
      h_index: r.h,
      g_index: r.g,
      i10_index: r.i10,
      meanCitations: r.meanCitations,
      medianCitations: r.medianCitations,
      topSpecialization: r.topSpecialization,
      topDocument: r.topDocument,
      timeline: r.timeline,
      coauthors: r.coauthors?.slice(0, 5) ?? [],
      countries: r.countries?.slice(0, 5) ?? [],
    })),
  };
}

/** Executa filtro e agregação flexível sobre os documentos */
function executeFilterAndAggregate(
  args: Record<string, unknown>,
  dataset: Dataset,
): Record<string, unknown> {
  const yearMin = args.yearMin !== undefined ? Number(args.yearMin) : null;
  const yearMax = args.yearMax !== undefined ? Number(args.yearMax) : null;
  const theme = args.theme ? String(args.theme).trim().toLowerCase() : null;
  const author = args.author ? String(args.author).trim().toLowerCase() : null;
  const country = args.country ? String(args.country).trim().toLowerCase() : null;
  const venue = args.venue ? String(args.venue).trim().toLowerCase() : null;
  const keyword = args.keyword ? String(args.keyword).trim().toLowerCase() : null;
  const minCitations = args.minCitations !== undefined ? Number(args.minCitations) : null;
  const limit = Math.min(Math.max(Number(args.limit ?? 10) || 10, 1), 25);

  const columns = collectColumns(dataset);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);

  const matchingDocs = dataset.filter((doc) => {
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (yearMin !== null && (year === null || year < yearMin)) return false;
    if (yearMax !== null && (year === null || year > yearMax)) return false;

    const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
    if (minCitations !== null && citations < minCitations) return false;

    if (theme && columns.has(FIELD.THEME)) {
      const docTheme = String(doc[FIELD.THEME] ?? '').toLowerCase();
      if (!docTheme.includes(theme)) return false;
    }

    if (author && authorsColumn) {
      const docAuthors = String(doc[authorsColumn] ?? '').toLowerCase();
      if (!docAuthors.includes(author)) return false;
    }

    if (country && columns.has(FIELD.COUNTRY)) {
      const docCountry = String(doc[FIELD.COUNTRY] ?? '').toLowerCase();
      if (!docCountry.includes(country)) return false;
    }

    if (venue && venueColumn) {
      const docVenue = String(doc[venueColumn] ?? '').toLowerCase();
      if (!docVenue.includes(venue)) return false;
    }

    if (keyword && keywordsColumn) {
      const docKeywords = String(doc[keywordsColumn] ?? '').toLowerCase();
      if (!docKeywords.includes(keyword)) return false;
    }

    return true;
  });

  const citationList = matchingDocs.map((doc) => toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0);
  const totalCits = sum(citationList);
  const avgCits = citationList.length > 0 ? pyRound(mean(citationList), 2) : 0;
  const medCits = citationList.length > 0 ? pyRound(median(citationList), 2) : 0;

  const validYearsList = matchingDocs
    .map((doc) => toNumeric(doc[FIELD.YEAR_CLEAN]))
    .filter((y): y is number => y !== null);

  const timespan =
    validYearsList.length > 0
      ? `${Math.min(...validYearsList)} - ${Math.max(...validYearsList)}`
      : 'N/S';

  // Distribuição de temas nos matches
  const themeCounts = new Map<string, number>();
  if (columns.has(FIELD.THEME)) {
    for (const doc of matchingDocs) {
      const t = String(doc[FIELD.THEME] ?? '').trim();
      if (t && !isNullLike(t)) themeCounts.set(t, (themeCounts.get(t) ?? 0) + 1);
    }
  }

  // Ordenar matches por citações decrescente para amostragem
  const sortedMatches = [...matchingDocs].sort(
    (a, b) =>
      (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0),
  );

  const sampleDocuments = sortedMatches.slice(0, limit).map((doc) => {
    const rawAbstract = String(doc[FIELD.ABSTRACT] ?? '');
    return {
      title: titleColumn ? String(doc[titleColumn] ?? '') : 'Sem título',
      authors: authorsColumn ? String(doc[authorsColumn] ?? '') : '',
      year: toNumeric(doc[FIELD.YEAR_CLEAN]),
      venue: venueColumn ? String(doc[venueColumn] ?? '') : '',
      citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
      theme: columns.has(FIELD.THEME) ? String(doc[FIELD.THEME] ?? '') : undefined,
      abstractSnippet: rawAbstract ? rawAbstract.slice(0, 300) + '...' : undefined,
    };
  });

  return {
    totalMatchingDocuments: matchingDocs.length,
    percentageOfDataset:
      dataset.length > 0 ? pyRound((matchingDocs.length / dataset.length) * 100, 1) + '%' : '0%',
    totalCitations: totalCits,
    meanCitations: avgCits,
    medianCitations: medCits,
    timespan,
    topThemes: [...themeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, count]) => ({ theme: name, documents: count })),
    sampleDocuments,
  };
}

/** Executa obtenção de métricas gerais e bibliométricas */
function executeGetGeneralMetrics(dataset: Dataset): Record<string, unknown> {
  const summary = summarize(dataset);
  const yearly = docsPerYear(dataset);
  const columns = collectColumns(dataset);

  // Calcula citações anuais
  const citationsPerYear = new Map<number, number>();
  for (const doc of dataset) {
    const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
    if (year !== null) {
      const y = Math.trunc(year);
      const c = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
      citationsPerYear.set(y, (citationsPerYear.get(y) ?? 0) + c);
    }
  }

  const yearlyProduction = yearly.map(({ year, count }) => ({
    year,
    documents: count,
    citations: citationsPerYear.get(year) ?? 0,
  }));

  const themeCounts = new Map<string, number>();
  if (columns.has(FIELD.THEME)) {
    for (const doc of dataset) {
      const t = String(doc[FIELD.THEME] ?? '').trim();
      if (t && !isNullLike(t)) themeCounts.set(t, (themeCounts.get(t) ?? 0) + 1);
    }
  }

  return {
    totalDocuments: summary.totalDocs,
    timespan: summary.timespan,
    averageDocumentAge: summary.avgAge,
    uniqueAuthors: summary.authorsCount,
    uniqueCountries: summary.countriesCount,
    uniqueVenues: summary.venuesCount,
    uniqueKeywords: summary.keywordsCount,
    bibliometrix: {
      annualGrowthRateCAGR: `${summary.bibliometrix.growthRate}%`,
      internationalCollaborationDocs_MCP: summary.bibliometrix.mcp,
      singleCountryDocs_SCP: summary.bibliometrix.scp,
      singleAuthorDocuments: summary.bibliometrix.singleAuthorDocs,
      coauthorsPerDocumentIndex: summary.bibliometrix.coauthIndex,
      averageCitationsPerYear: summary.bibliometrix.avgCitPerYear,
    },
    yearlyProduction,
    themes: [...themeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([name, documents]) => ({ name, documents })),
  };
}

/** Executa raio-x / perfil aprofundado de entidade */
function executeGetEntityProfile(
  args: Record<string, unknown>,
  dataset: Dataset,
): Record<string, unknown> {
  const entityType = String(args.entityType ?? 'author').toLowerCase();
  const nameQuery = String(args.name ?? '').trim().toLowerCase();

  if (!nameQuery) throw new Error('Nome da entidade não fornecido.');

  const columns = collectColumns(dataset);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);

  if (entityType === 'theme') {
    if (!columns.has(FIELD.THEME)) {
      return { found: false, message: 'Base não possui classificação temática (TEMA_GEMINI).' };
    }

    const matchingDocs = dataset.filter((doc) =>
      String(doc[FIELD.THEME] ?? '').toLowerCase().includes(nameQuery),
    );

    if (matchingDocs.length === 0) {
      return { found: false, message: `Nenhum documento encontrado para o tema "${args.name}".` };
    }

    const cits = matchingDocs.map((d) => toNumeric(d[FIELD.TOTAL_CITATIONS]) ?? 0);
    const topPapers = [...matchingDocs]
      .sort(
        (a, b) =>
          (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0),
      )
      .slice(0, 10)
      .map((doc) => ({
        title: titleColumn ? String(doc[titleColumn] ?? '') : 'Sem título',
        authors: authorsColumn ? String(doc[authorsColumn] ?? '') : '',
        year: toNumeric(doc[FIELD.YEAR_CLEAN]),
        venue: venueColumn ? String(doc[venueColumn] ?? '') : '',
        citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
      }));

    return {
      entityType: 'theme',
      name: args.name,
      totalDocuments: matchingDocs.length,
      totalCitations: sum(cits),
      meanCitations: pyRound(mean(cits), 2),
      topCitedPapers: topPapers,
    };
  }

  // Para autores, países ou venues: busca na tabela analítica
  let tableRows: EntityRow[] = [];
  if (entityType === 'author') tableRows = authorsTable(dataset);
  else if (entityType === 'country') tableRows = countriesTable(dataset);
  else if (entityType === 'venue') tableRows = venuesTable(dataset);
  else throw new Error(`Tipo de entidade inválido: ${entityType}`);

  const match = tableRows.find((r) => r.entity.toLowerCase().includes(nameQuery));
  if (!match) {
    return {
      found: false,
      entityType,
      searchedName: args.name,
      message: `Entidade "${args.name}" não encontrada na tabela de ${entityType}s.`,
    };
  }

  // Coleta documentos desse match
  const matchingDocs = dataset.filter((doc) => {
    if (entityType === 'author' && authorsColumn) {
      return splitTokens(doc[authorsColumn], 'title').includes(match.entity);
    }
    if (entityType === 'country' && columns.has(FIELD.COUNTRY)) {
      return splitTokens(doc[FIELD.COUNTRY], 'title').includes(match.entity);
    }
    if (entityType === 'venue' && venueColumn) {
      return String(doc[venueColumn] ?? '').trim().toUpperCase() === match.entity;
    }
    return false;
  });

  const publications = matchingDocs
    .sort(
      (a, b) =>
        (toNumeric(b[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(a[FIELD.TOTAL_CITATIONS]) ?? 0),
    )
    .slice(0, 15)
    .map((doc) => ({
      title: titleColumn ? String(doc[titleColumn] ?? '') : 'Sem título',
      authors: authorsColumn ? String(doc[authorsColumn] ?? '') : '',
      year: toNumeric(doc[FIELD.YEAR_CLEAN]),
      venue: venueColumn ? String(doc[venueColumn] ?? '') : '',
      citations: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
      theme: columns.has(FIELD.THEME) ? String(doc[FIELD.THEME] ?? '') : undefined,
    }));

  return {
    found: true,
    entityType,
    entity: match.entity,
    totalDocuments: match.docCount,
    totalCitations: match.citations,
    scientometricIndices: {
      h_index: match.h,
      g_index: match.g,
      i10_index: match.i10,
      m_quotient: match.m,
    },
    citationStats: {
      mean: match.meanCitations,
      median: match.medianCitations,
      std: match.stdCitations,
    },
    topSpecialization_LocationalQuotient: match.topSpecialization,
    topCitedDocument: match.topDocument,
    timeline: match.timeline,
    topCollaborators: match.coauthors?.slice(0, 8) ?? [],
    countries: match.countries?.slice(0, 5) ?? [],
    publications,
  };
}
