import { create } from 'zustand';

import { applyHybridThemes, otherCategoryName } from '@/core/hybrid/apply';
import { toHybridDocs, type HybridDoc } from '@/core/hybrid/documents';
import { buildJevRequest, buildThemeQuestion, parseJevDecision } from '@/core/hybrid/jev-request';
import {
  categoryCountsOf,
  compareLabels,
  coverageOf,
  leftoverIndices,
  weakCategories,
} from '@/core/hybrid/metrics';
import {
  buildClarifyPrompt,
  buildDiscoveryPrompt,
  buildExpandPrompt,
  parseClarify,
  parseDiscovery,
  parseExpansion,
  slugify,
  type ConfusionCase,
} from '@/core/hybrid/prompts';
import { seededRandom, stratifiedSample } from '@/core/hybrid/sampling';
import {
  OTHER_CATEGORY_ID,
  type DocumentDecision,
  type HybridCategory,
  type HybridRun,
  type TaxonomyOrigin,
  type TaxonomyVersion,
  type ValidationRound,
} from '@/core/hybrid/types';
import { buildSearchOptions } from '@/core/search';
import { chatJson } from '@/lib/deepseek-client';
import { jevEvaluate, runPool } from '@/lib/jev-client';
import type { Dataset } from '@/lib/types';
import { getAiWorker, proxyProgress } from '@/workers/client';
import { DERIVED_RESET, useDataset } from './dataset.store';
import { useHybridConfig, type HybridConfig } from './hybrid-config.store';
import { useLocale } from './locale.store';

/**
 * Orquestração da classificação híbrida.
 *
 *   1. amostra estratificada (agrupamentos do k-means × período + atípicos)
 *   2. o modelo gerativo propõe as categorias e rotula a amostra
 *   3. pausa para o usuário revisar/editar as categorias (opcional)
 *   4. o Jev classifica a amostra; a concordância com os rótulos do passo 2 é medida e,
 *      se alguma categoria ficar abaixo de 60%, o modelo gerativo reescreve as descrições
 *   5. o Jev classifica o restante da base
 *   6. se sobrar muito em "outro"/baixa confiança, o modelo gerativo propõe categorias
 *      novas a partir das sobras e a base inteira é reclassificada com a taxonomia final
 *   7. o resultado vai para a base (campo de tema + confiança + faixa) e o registro da
 *      execução vai para o projeto
 *
 * As chamadas são de rede, não de CPU, então rodam aqui na thread principal sem travar a
 * interface; só o k-means (quando ainda não existe) vai para o worker de IA.
 */

const SEED = 42;
/** Casos de discordância enviados para o esclarecimento das descrições. */
const MAX_CLARIFY_CASES = 30;
/** Documentos remanescentes enviados para a expansão. */
const MAX_EXPANSION_DOCS = 60;
const MAX_NEW_CATEGORIES = 4;

export type HybridStage = 'idle' | 'running' | 'review' | 'done' | 'error';

export interface HybridProgress {
  phase: string;
  ratio: number;
  detail?: string;
}

interface HybridState {
  stage: HybridStage;
  progress: HybridProgress | null;
  /** Categorias em revisão pelo usuário (etapa 3). */
  draft: HybridCategory[] | null;
  /** Rótulos do modelo gerativo por categoria na amostra — ajuda o usuário a revisar. */
  draftSupport: Record<string, number>;
  isManual: boolean;
  error: string | null;

  start: (focus?: string) => Promise<void>;
  startManual: () => void;
  updateDraft: (categories: HybridCategory[]) => void;
  confirm: () => Promise<void>;
  cancel: () => void;
  dismiss: () => void;
}

interface Session {
  dataset: Dataset;
  docs: HybridDoc[];
  config: HybridConfig;
  locale: 'pt' | 'en';
  otherName: string;
  startedAt: string;
  controller: AbortController;
  sample: HybridRun['sample'];
  sampleDocs: HybridDoc[];
  /** Rótulos do modelo gerativo: posição na base → id da categoria. */
  reference: Map<number, string>;
  versions: TaxonomyVersion[];
  discoveryModel: string | null;
  classifierModel: string;
  usage: HybridRun['usage'];
}

let session: Session | null = null;

function describeError(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function isAbort(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === 'AbortError';
}

function pt(): boolean {
  return useLocale.getState().locale === 'pt';
}

function pushVersion(target: Session, origin: TaxonomyOrigin, categories: HybridCategory[]): void {
  target.versions.push({
    version: target.versions.length + 1,
    origin,
    createdAt: new Date().toISOString(),
    categories: categories.map((category) => ({ ...category, examples: [...category.examples] })),
  });
}

function currentCategories(target: Session): HybridCategory[] {
  return target.versions[target.versions.length - 1]?.categories ?? [];
}

/** Limpa e valida as categorias vindas do editor. */
function sanitizeDraft(categories: readonly HybridCategory[], otherName: string): HybridCategory[] {
  const taken = new Set<string>();
  const ids = new Set<string>();
  const out: HybridCategory[] = [];
  for (const category of categories) {
    const name = category.name.replace(/\s+/g, ' ').trim().slice(0, 60);
    if (!name) continue;
    const key = name.toLowerCase();
    if (key === otherName.toLowerCase() || key === 'other' || key === 'outro' || key === 'outros') {
      throw new Error(
        pt()
          ? `"${name}" é reservado para a categoria "outros", que já é incluída automaticamente.`
          : `"${name}" is reserved for the "other" category, which is added automatically.`,
      );
    }
    if (taken.has(key)) {
      throw new Error(pt() ? `Há duas categorias chamadas "${name}".` : `Two categories are named "${name}".`);
    }
    taken.add(key);
    let id = category.id && category.id !== OTHER_CATEGORY_ID ? category.id : slugify(name);
    while (ids.has(id)) id = `${id}-2`;
    ids.add(id);
    out.push({
      id,
      name,
      what: category.what.trim(),
      notFor: category.notFor.trim(),
      examples: category.examples.map((example) => example.trim()).filter(Boolean),
    });
  }
  if (out.length < 2) {
    throw new Error(pt() ? 'Defina ao menos duas categorias.' : 'Define at least two categories.');
  }
  if (out.length > 254) {
    throw new Error(pt() ? 'O Jev aceita no máximo 254 categorias além de "outros".' : 'Jev accepts at most 254 categories besides "other".');
  }
  return out;
}

export const useHybrid = create<HybridState>((set, get) => {
  const setProgress = (phase: string, ratio: number, detail?: string) =>
    set({ progress: { phase, ratio, ...(detail ? { detail } : {}) } });

  /** Classifica documentos com o Jev, devolvendo posição na base → decisão. */
  async function classify(
    target: Session,
    docs: readonly HybridDoc[],
    categories: readonly HybridCategory[],
    phase: string,
  ): Promise<Map<number, DocumentDecision>> {
    const question = buildThemeQuestion(categories, target.otherName);
    const { jev, concurrency } = target.config;
    setProgress(phase, 0, `0 / ${docs.length}`);

    const results = await runPool(
      docs,
      concurrency,
      async (doc) => {
        const response = await jevEvaluate(buildJevRequest(doc, question, jev.model), jev.apiKey, target.controller.signal);
        target.usage.classifierInputTokens += response.usage?.input_tokens ?? 0;
        target.usage.classifierRequests += 1;
        if (response.model) target.classifierModel = response.model;
        return [doc.index, parseJevDecision(response, categories, target.otherName)] as const;
      },
      (done, total) => setProgress(phase, done / total, `${done.toLocaleString()} / ${total.toLocaleString()}`),
      target.controller.signal,
    );
    return new Map(results);
  }

  async function generate(target: Session, prompt: Parameters<typeof chatJson>[1]): Promise<string> {
    const result = await chatJson(target.config.generative, prompt, target.controller.signal);
    target.discoveryModel = result.model;
    target.usage.discoveryInputTokens += result.inputTokens;
    target.usage.discoveryOutputTokens += result.outputTokens;
    return result.text;
  }

  /** Etapas 4 a 7: validação, classificação da base, expansão e aplicação. */
  async function classifyAndApply(target: Session): Promise<void> {
    const { config, otherName } = target;
    const isEn = !pt();
    let categories = currentCategories(target);
    const validation: ValidationRound[] = [];

    // 4. Validação na amostra, com esclarecimento das descrições se alguma categoria falhar.
    let decisions = new Map<number, DocumentDecision>();
    if (target.reference.size > 0) {
      const referenceDocs = target.sampleDocs.filter((doc) => target.reference.has(doc.index));
      decisions = await classify(target, referenceDocs, categories, isEn ? 'Validating on the sample' : 'Validando na amostra');
      let round = compareLabels(target.reference, decisions, categories, otherName, target.versions.length);
      validation.push(round);

      for (let attempt = 0; attempt < config.maxClarifyRounds; attempt += 1) {
        const weak = new Set(weakCategories(round));
        if (weak.size === 0) break;
        const cases: ConfusionCase[] = referenceDocs
          .filter((doc) => {
            const reference = target.reference.get(doc.index) as string;
            const decided = decisions.get(doc.index)?.categoryId;
            return decided !== undefined && decided !== reference && (weak.has(reference) || weak.has(decided));
          })
          .slice(0, MAX_CLARIFY_CASES)
          .map((doc) => ({
            doc,
            referenceId: target.reference.get(doc.index) as string,
            classifierId: decisions.get(doc.index)?.categoryId as string,
          }));
        if (cases.length === 0) break;

        setProgress(isEn ? 'Clarifying ambiguous categories' : 'Esclarecendo categorias ambíguas', 0);
        const raw = await generate(target, buildClarifyPrompt(categories, cases, otherName));
        categories = parseClarify(raw, categories, cases.map((item) => item.doc));
        pushVersion(target, 'clarify', categories);

        decisions = await classify(target, referenceDocs, categories, isEn ? 'Re-validating on the sample' : 'Revalidando na amostra');
        round = compareLabels(target.reference, decisions, categories, otherName, target.versions.length);
        validation.push(round);
      }
    }

    // 5. Restante da base, com a mesma versão da taxonomia usada na última validação.
    const remaining = target.docs.filter((doc) => !decisions.has(doc.index));
    if (remaining.length > 0) {
      const rest = await classify(target, remaining, categories, isEn ? 'Classifying the corpus' : 'Classificando a base');
      for (const [index, decision] of rest) decisions.set(index, decision);
    }

    // 6. Expansão a partir das sobras.
    const allReference = new Map(target.reference);
    let expansionRounds = 0;
    const canGenerate = Boolean(config.generative.apiKey.trim());
    for (let attempt = 0; canGenerate && attempt < config.maxExpansionRounds; attempt += 1) {
      const leftovers = leftoverIndices(decisions, config.thresholds);
      if (leftovers.length / target.docs.length <= config.leftoverTrigger) break;

      const random = seededRandom(SEED + attempt + 1);
      const pool = leftovers
        .map((index) => target.docs[index] as HybridDoc)
        .map((doc) => ({ doc, key: random() }))
        .sort((a, b) => Number(Boolean(b.doc.abstract)) - Number(Boolean(a.doc.abstract)) || a.key - b.key)
        .slice(0, MAX_EXPANSION_DOCS)
        .map((item) => item.doc);

      setProgress(isEn ? 'Looking for missing categories' : 'Procurando categorias que faltaram', 0);
      const raw = await generate(
        target,
        buildExpandPrompt(categories, pool, { locale: target.locale, maxNew: MAX_NEW_CATEGORIES }),
      );
      const expansion = parseExpansion(raw, categories, pool, { maxNew: MAX_NEW_CATEGORIES, otherName });
      if (expansion.newCategories.length === 0) break;

      for (const [index, label] of expansion.labels) allReference.set(index, label);
      categories = [...categories, ...expansion.newCategories];
      pushVersion(target, 'expand', categories);
      expansionRounds += 1;
      // A base inteira é reclassificada: com uma categoria nova, um documento já
      // classificado pode caber melhor nela. Aplicar só às sobras deixaria a base com
      // duas taxonomias misturadas.
      decisions = await classify(target, target.docs, categories, isEn ? 'Reclassifying with the final taxonomy' : 'Reclassificando com a taxonomia final');
    }

    // 7. Aplicação.
    if (useDataset.getState().active !== target.dataset) {
      throw new Error(
        isEn
          ? 'The dataset changed during classification; the result was discarded.'
          : 'A base mudou durante a classificação; o resultado foi descartado.',
      );
    }

    const firstVersion = target.versions[0];
    const run: HybridRun = {
      id: crypto.randomUUID(),
      startedAt: target.startedAt,
      finishedAt: new Date().toISOString(),
      locale: target.locale,
      models: { discovery: target.discoveryModel, classifier: target.classifierModel },
      sample: target.sample,
      thresholds: config.thresholds,
      taxonomyVersions: target.versions,
      finalTaxonomy: categories,
      validation,
      finalAgreement:
        allReference.size > 0
          ? compareLabels(allReference, decisions, categories, otherName, target.versions.length)
          : null,
      expansionRounds,
      coverage: coverageOf(decisions, config.thresholds),
      categoryCounts: categoryCountsOf(decisions, categories, otherName),
      usage: target.usage,
      userEdited: target.versions.some((version) => version.origin === 'user-edit') && firstVersion?.origin !== 'manual',
    };

    const themed = applyHybridThemes(target.dataset, decisions, categories, config.thresholds, target.locale);
    session = null;
    useDataset.setState({
      active: themed,
      ...DERIVED_RESET,
      hybridRun: run,
      searchOptions: buildSearchOptions(themed),
      error: null,
    });
  }

  async function runGuarded(target: Session, work: () => Promise<void>): Promise<void> {
    set({ stage: 'running', error: null });
    useDataset.setState({ isCategorizingThemes: true });
    try {
      await work();
      if (get().stage === 'running') set({ stage: session ? 'review' : 'done', progress: null });
    } catch (cause) {
      if (isAbort(cause)) {
        session = null;
        set({ stage: 'idle', progress: null, draft: null, error: null });
      } else {
        // Numa falha, a sessão sobrevive se ainda houver categorias: o usuário pode
        // corrigir a chave e tentar de novo a partir da revisão, sem refazer a descoberta.
        const recoverable = session === target && target.versions.length > 0;
        set({
          stage: recoverable ? 'review' : 'error',
          progress: null,
          draft: recoverable ? currentCategories(target) : null,
          error: describeError(cause),
        });
        if (!recoverable) session = null;
        target.controller = new AbortController();
      }
    } finally {
      useDataset.setState({ isCategorizingThemes: false });
    }
  }

  function newSession(dataset: Dataset): Session {
    const config = useHybridConfig.getState().config;
    const locale = useLocale.getState().locale;
    return {
      dataset,
      docs: toHybridDocs(dataset),
      config,
      locale,
      otherName: otherCategoryName(locale),
      startedAt: new Date().toISOString(),
      controller: new AbortController(),
      sample: { size: 0, seed: SEED, strata: 0, outliers: 0, clusterCount: 0, silhouette: null },
      sampleDocs: [],
      reference: new Map(),
      versions: [],
      discoveryModel: null,
      classifierModel: config.jev.model,
      usage: { discoveryInputTokens: 0, discoveryOutputTokens: 0, classifierInputTokens: 0, classifierRequests: 0 },
    };
  }

  return {
    stage: 'idle',
    progress: null,
    draft: null,
    draftSupport: {},
    isManual: false,
    error: null,

    async start(focus) {
      const dataset = useDataset.getState().active;
      if (!dataset || dataset.length === 0) return;
      const target = newSession(dataset);
      session = target;
      set({ isManual: false, draft: null, draftSupport: {} });

      await runGuarded(target, async () => {
        const isEn = !pt();
        const { config } = target;
        if (!config.generative.apiKey.trim()) {
          throw new Error(isEn ? 'Set the generative model API key first.' : 'Informe primeiro a chave do modelo gerativo.');
        }

        // 1. Agrupamentos para estratificar: reaproveita os do k-means se já existirem.
        let clustering = useDataset.getState().clustering;
        if (!clustering || clustering.assignments.length !== dataset.length) {
          setProgress(isEn ? 'Grouping documents for sampling' : 'Agrupando documentos para a amostragem', 0);
          clustering = await getAiWorker().cluster(
            dataset,
            10,
            proxyProgress((update: { phase: string; ratio: number }) => setProgress(update.phase, update.ratio)),
          );
        }
        target.controller.signal.throwIfAborted();

        const terms = new Map((clustering?.clusters ?? []).map((cluster) => [cluster.clusterId, cluster.topTerms]));
        const sample = stratifiedSample(target.docs, clustering?.assignments ?? null, terms, {
          size: config.sampleSize,
          seed: SEED,
        });
        target.sample = {
          size: sample.indices.length,
          seed: SEED,
          strata: sample.strata,
          outliers: sample.outliers,
          clusterCount: clustering?.clusterCount ?? 0,
          silhouette: clustering?.silhouette ?? null,
        };
        target.sampleDocs = sample.indices.map((index) => target.docs[index] as HybridDoc);

        // 2. Descoberta.
        setProgress(
          isEn ? 'Discovering categories' : 'Descobrindo categorias',
          0.5,
          `${target.sampleDocs.length} ${isEn ? 'documents' : 'documentos'} · ${config.generative.model}`,
        );
        const raw = await generate(
          target,
          buildDiscoveryPrompt(target.sampleDocs, {
            locale: target.locale,
            maxCategories: config.maxCategories,
            ...(focus ? { focus } : {}),
          }),
        );
        const discovery = parseDiscovery(raw, target.sampleDocs, {
          maxCategories: config.maxCategories,
          otherName: target.otherName,
        });
        target.reference = discovery.labels;
        pushVersion(target, 'discovery', discovery.categories);

        const support: Record<string, number> = {};
        for (const label of discovery.labels.values()) support[label] = (support[label] ?? 0) + 1;

        // 3. Revisão pelo usuário, ou segue direto.
        if (config.reviewBeforeClassify) {
          set({ stage: 'review', progress: null, draft: discovery.categories, draftSupport: support });
          return;
        }
        set({ draftSupport: support });
        await classifyAndApply(target);
      });
    },

    startManual() {
      const dataset = useDataset.getState().active;
      if (!dataset || dataset.length === 0) return;
      const target = newSession(dataset);
      session = target;
      const blank = (): HybridCategory => ({ id: '', name: '', what: '', notFor: '', examples: [] });
      set({ stage: 'review', isManual: true, draft: [blank(), blank(), blank()], draftSupport: {}, error: null, progress: null });
    },

    updateDraft(categories) {
      set({ draft: categories });
    },

    async confirm() {
      const target = session;
      const draft = get().draft;
      if (!target || !draft) return;
      if (useDataset.getState().active !== target.dataset) {
        session = null;
        set({ stage: 'error', draft: null, error: pt() ? 'A base mudou; recomece a classificação.' : 'The dataset changed; start over.' });
        return;
      }

      let categories: HybridCategory[];
      try {
        categories = sanitizeDraft(draft, target.otherName);
      } catch (cause) {
        set({ error: describeError(cause) });
        return;
      }

      const previous = currentCategories(target);
      if (target.versions.length === 0) {
        pushVersion(target, 'manual', categories);
      } else if (JSON.stringify(previous) !== JSON.stringify(categories)) {
        pushVersion(target, 'user-edit', categories);
        // Rótulos de referência de categorias removidas não têm mais contra o que comparar.
        const kept = new Set(categories.map((category) => category.id));
        for (const [index, label] of target.reference) {
          if (label !== OTHER_CATEGORY_ID && !kept.has(label)) target.reference.delete(index);
        }
      }

      set({ draft: null });
      await runGuarded(target, () => classifyAndApply(target));
    },

    cancel() {
      if (session) session.controller.abort();
      if (get().stage === 'review') {
        session = null;
        set({ stage: 'idle', draft: null, error: null, progress: null });
      }
    },

    dismiss() {
      if (get().stage === 'done' || get().stage === 'error') set({ stage: 'idle', error: null, progress: null });
    },
  };
});

// Troca de base (upload, deduplicação, outro projeto) invalida uma sessão em andamento.
useDataset.subscribe(
  (state) => state.active,
  (active) => {
    if (session && active !== session.dataset) {
      session.controller.abort();
      session = null;
      useHybrid.setState({ stage: 'idle', draft: null, progress: null, error: null });
    }
  },
);
