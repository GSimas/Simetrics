import type { ReviewFlow } from './flow';
import type { ReviewType } from './types';

/**
 * Exportação para o PRISMALab (prisma.scientata.com), a ferramenta do ecossistema Scientata
 * que desenha o diagrama PRISMA 2020. O Simetrics não redesenha o diagrama: entrega as
 * contagens num projeto que o PRISMALab importa como está.
 *
 * O formato espelha `PrismaProject` (schemaVersion 2) de
 * github.com/GSimas/PRISMA-Diagram, web/src/domain/types.ts. Com `schemaVersion: 2` o
 * PRISMALab valida o objeto inteiro com zod — todo campo precisa existir, por isso nada
 * aqui é opcional. Se o PRISMALab mudar de versão, este arquivo muda junto.
 */

export const PRISMALAB_URL = 'https://prisma.scientata.com';

const PRISMALAB_SCHEMA_VERSION = 2;

export const PRISMALAB_COUNT_KEYS = [
  'previousStudies', 'previousReports', 'databases', 'registers', 'websites',
  'organisations', 'citationSearching', 'otherSources', 'duplicates',
  'automationExcluded', 'removedOther', 'screened', 'recordsExcluded',
  'reportsSought', 'reportsNotRetrieved', 'reportsAssessed', 'reportsExcluded',
  'otherReportsSought', 'otherReportsNotRetrieved', 'otherReportsAssessed', 'otherReportsExcluded',
  'newStudies', 'newReports', 'totalStudies', 'totalReports',
] as const;

export type PrismaLabCountKey = (typeof PRISMALAB_COUNT_KEYS)[number];

export interface PrismaLabProject {
  schemaVersion: typeof PRISMALAB_SCHEMA_VERSION;
  id: string;
  title: string;
  shortTitle: string;
  authors: string[];
  institution: string;
  protocolUrl: string;
  reviewType: 'systematic' | 'scoping' | 'living' | 'network-meta-analysis';
  reviewKind: 'new';
  model: 'new-databases';
  guideline: 'PRISMA 2020';
  extensions: string[];
  locale: 'pt-BR' | 'en';
  status: 'draft';
  updatedDate: string;
  observations: string;
  sources: { id: string; type: 'database'; name: string; count: number }[];
  counts: Record<PrismaLabCountKey, number | null>;
  overrides: Record<string, never>;
  exclusionReasons: { id: string; label: string; count: number }[];
  otherExclusionReasons: { id: string; label: string; count: number }[];
  provenance: Record<string, never>;
  checklist: {
    item: number;
    status: 'not-started';
    note: string;
    location: string;
    page: string;
    section: string;
    url: string;
    reviewedAt: string;
  }[];
  presentation: {
    mode: 'prisma';
    diagramStyle: 'classic';
    density: 'comfortable';
    orientation: 'portrait';
    accent: string;
    showTitle: boolean;
    showOptionalDetails: boolean;
  };
  history: { id: string; at: string; action: string }[];
  createdAt: string;
  updatedAt: string;
}

export interface PrismaLabExportInput {
  title: string;
  reviewType: ReviewType;
  flow: ReviewFlow;
  locale: 'pt' | 'en';
  now?: Date;
  makeId?: () => string;
}

export function toPrismaLabProject({
  title,
  reviewType,
  flow,
  locale,
  now = new Date(),
  makeId = () => crypto.randomUUID(),
}: PrismaLabExportInput): PrismaLabProject {
  const iso = now.toISOString();
  const isEn = locale === 'en';
  const counts = Object.fromEntries(PRISMALAB_COUNT_KEYS.map((key) => [key, null])) as Record<
    PrismaLabCountKey,
    number | null
  >;

  const assessed = flow.fullText.eligible - flow.fullText.notRetrieved;
  Object.assign(counts, {
    databases: flow.identified,
    // Tudo o que o Simetrics conhece veio dos arquivos importados: registros de ensaios e
    // remoções por automação ou outros motivos não passam por ele.
    registers: 0,
    duplicates: flow.duplicatesRemoved,
    automationExcluded: 0,
    removedOther: 0,
    screened: flow.screened,
    recordsExcluded: flow.titleAbstract.exclude,
    reportsSought: flow.fullText.eligible,
    reportsNotRetrieved: flow.fullText.notRetrieved,
    reportsAssessed: assessed,
    reportsExcluded: flow.fullText.exclude,
    // Cada registro triado conta como um relato; estudos com vários relatos se ajustam no PRISMALab.
    newStudies: flow.included,
    newReports: flow.included,
    totalStudies: flow.included,
    totalReports: flow.included,
  } satisfies Partial<Record<PrismaLabCountKey, number>>);

  return {
    schemaVersion: PRISMALAB_SCHEMA_VERSION,
    id: makeId(),
    title: title.trim() || (isEn ? 'Review from Simetrics' : 'Revisão do Simetrics'),
    shortTitle: '',
    authors: [],
    institution: '',
    protocolUrl: '',
    reviewType: reviewType === 'scoping' ? 'scoping' : 'systematic',
    reviewKind: 'new',
    model: 'new-databases',
    guideline: 'PRISMA 2020',
    extensions: reviewType === 'scoping' ? ['PRISMA-ScR'] : [],
    locale: isEn ? 'en' : 'pt-BR',
    status: 'draft',
    updatedDate: iso.slice(0, 10),
    observations: isEn
      ? 'Counts exported by Simetrics: identification from the imported files, duplicates from Simetrics deduplication, screening from the recorded decisions. Each screened record counts as one report.'
      : 'Contagens exportadas pelo Simetrics: identificação a partir dos arquivos importados, duplicatas da deduplicação do Simetrics e triagem a partir das decisões registradas. Cada registro triado conta como um relato.',
    sources: flow.identifiedBySource.map(({ name, count }) => ({ id: makeId(), type: 'database', name, count })),
    counts,
    overrides: {},
    exclusionReasons: flow.fullTextExclusions.map(({ label, count }) => ({ id: makeId(), label, count })),
    otherExclusionReasons: [],
    provenance: {},
    checklist: Array.from({ length: 27 }, (_, index) => ({
      item: index + 1,
      status: 'not-started',
      note: '',
      location: '',
      page: '',
      section: '',
      url: '',
      reviewedAt: '',
    })),
    presentation: {
      mode: 'prisma',
      diagramStyle: 'classic',
      density: 'comfortable',
      orientation: 'portrait',
      accent: '#c97a16',
      showTitle: true,
      showOptionalDetails: true,
    },
    history: [{ id: makeId(), at: iso, action: isEn ? 'Imported from Simetrics' : 'Importado do Simetrics' }],
    createdAt: iso,
    updatedAt: iso,
  };
}
