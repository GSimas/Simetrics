import { useEffect, useRef, type CSSProperties } from 'react';
import { ArrowDown, ArrowRight, ArrowUpRight, Upload } from 'lucide-react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { GithubButton, SettingsButton } from '@/components/HeaderActions';
import { EmptyState } from '@/features/EmptyState';
import { ProjectCard } from '@/features/landing/ProjectCard';
import type { TranslationKey } from '@/lib/i18n/translations';
import type { AppView } from '@/lib/use-hash-route';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useProjectStore } from '@/state/project.store';

const FAQ_ITEMS = [
  { questionKey: 'landing_faq_q1', answerKey: 'landing_faq_a1' },
  { questionKey: 'landing_faq_q2', answerKey: 'landing_faq_a2' },
  { questionKey: 'landing_faq_q3', answerKey: 'landing_faq_a3' },
  { questionKey: 'landing_faq_q4', answerKey: 'landing_faq_a4' },
  { questionKey: 'landing_faq_q5', answerKey: 'landing_faq_a5' },
] as const satisfies readonly { questionKey: TranslationKey; answerKey: TranslationKey }[];

const HIGHLIGHTS = [
  { labelKey: 'landing_highlight_2_label', textKey: 'landing_highlight_2_text' },
  { labelKey: 'landing_highlight_3_label', textKey: 'landing_highlight_3_text' },
  { labelKey: 'landing_highlight_1_label', textKey: 'landing_highlight_1_text' },
] as const satisfies readonly { labelKey: TranslationKey; textKey: TranslationKey }[];

/**
 * Gerador pseudoaleatório com semente fixa: as partículas ficam espalhadas de forma
 * irregular, mas iguais a cada render (Math.random mudaria tudo a cada atualização).
 */
function seeded(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

interface Particle {
  x: number;
  y: number;
  r: number;
  duration: number;
  delay: number;
  dx: number;
  dy: number;
  tone: 'muted' | 'signal' | 'cyan';
}

const PARTICLES: Particle[] = (() => {
  const random = seeded(7);
  return Array.from({ length: 28 }, (_, index) => ({
    x: 20 + random() * 1160,
    y: 20 + random() * 600,
    r: 1.1 + random() * 1.8,
    duration: 5 + random() * 7,
    delay: random() * 8,
    dx: (random() - 0.5) * 36,
    dy: -10 - random() * 26,
    tone: index % 6 === 0 ? 'signal' : index % 7 === 0 ? 'cyan' : 'muted',
  }));
})();

const PARTICLE_FILL = { muted: 'currentColor', signal: 'var(--highlight)', cyan: 'var(--cyan)' };

/** Constelação: uma pequena rede — nós e arestas — no canto oposto às órbitas. */
const NODES = [
  [120, 430],
  [210, 372],
  [300, 452],
  [255, 540],
  [390, 395],
  [150, 560],
  [440, 505],
] as const;
const EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [1, 4],
  [2, 4],
  [0, 5],
  [3, 5],
  [4, 6],
  [2, 6],
] as const;

/**
 * Órbitas, mira, constelação e partículas do hero, no traço fino do Scientata: o
 * movimento de fundo da tela inicial. Animações em index.css, paradas com movimento
 * reduzido.
 */
function HeroOrbits() {
  return (
    <svg
      className="pointer-events-none absolute inset-0 hidden h-full w-full text-border md:block"
      viewBox="0 0 1200 640"
      preserveAspectRatio="xMidYMid slice"
      fill="none"
      aria-hidden
    >
      {/* Diagonal com um feixe de luz percorrendo-a. */}
      <line x1="0" y1="560" x2="1200" y2="80" stroke="currentColor" />
      <line
        className="beam"
        x1="0"
        y1="560"
        x2="1200"
        y2="80"
        stroke="var(--highlight)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Mira com pulso de radar. */}
      <path d="M930 190v120M870 250h120" stroke="currentColor" />
      <circle className="radar-ping" cx="930" cy="250" r="24" stroke="var(--highlight)" />
      <circle className="radar-ping radar-ping--late" cx="930" cy="250" r="24" stroke="var(--cyan)" />

      <g className="orbit-spin orbit-spin--outer">
        <circle cx="930" cy="250" r="260" stroke="currentColor" />
        <circle cx="930" cy="-10" r="4.5" fill="var(--highlight)" />
        <circle cx="930" cy="510" r="2.5" fill="currentColor" />
      </g>
      <g className="orbit-spin orbit-spin--middle">
        <circle cx="930" cy="250" r="205" stroke="currentColor" strokeDasharray="2 10" />
        <circle cx="725" cy="250" r="3" fill="var(--cyan)" />
      </g>
      <g className="orbit-spin orbit-spin--inner">
        <circle cx="930" cy="250" r="150" stroke="currentColor" />
        <circle cx="1080" cy="250" r="3.5" fill="var(--highlight)" />
      </g>

      {/* Constelação com fluxo nas arestas. */}
      {EDGES.map(([from, to], index) => (
        <line
          key={`${from}-${to}`}
          className="edge-flow"
          x1={NODES[from]![0]}
          y1={NODES[from]![1]}
          x2={NODES[to]![0]}
          y2={NODES[to]![1]}
          stroke="currentColor"
          style={{ animationDelay: `-${index * 0.7}s` }}
        />
      ))}
      {NODES.map(([cx, cy], index) => (
        <circle
          key={`${cx}-${cy}`}
          className="node-pulse"
          cx={cx}
          cy={cy}
          r={index % 3 === 0 ? 4 : 3}
          fill={index % 3 === 0 ? 'var(--highlight)' : 'currentColor'}
          style={{ animationDelay: `-${index * 0.9}s` }}
        />
      ))}

      {PARTICLES.map((particle) => (
        <circle
          key={`${particle.x.toFixed(1)}-${particle.y.toFixed(1)}`}
          className="particle-float"
          cx={particle.x}
          cy={particle.y}
          r={particle.r}
          fill={PARTICLE_FILL[particle.tone]}
          style={
            {
              animationDuration: `${particle.duration.toFixed(2)}s`,
              animationDelay: `-${particle.delay.toFixed(2)}s`,
              '--dx': `${particle.dx.toFixed(1)}px`,
              '--dy': `${particle.dy.toFixed(1)}px`,
            } as CSSProperties
          }
        />
      ))}
    </svg>
  );
}

/**
 * Fundo vivo da tela inicial: a grade milimetrada desliza e três brilhos difusos
 * derivam. Fica fixo atrás de todo o conteúdo e cobre a grade estática do body.
 */
function AmbientBackground() {
  return (
    <div className="landing-ambient" aria-hidden>
      <div className="ambient-glow ambient-glow--a" />
      <div className="ambient-glow ambient-glow--b" />
      <div className="ambient-glow ambient-glow--c" />
    </div>
  );
}

export interface LandingScreenProps {
  navigate: (view: AppView, projectId?: string) => void;
  onOpenTutorial: () => void;
}

export function LandingScreen({ navigate, onOpenTutorial }: LandingScreenProps) {
  const { t } = useLocale();
  const importInputRef = useRef<HTMLInputElement>(null);

  const projects = useProjectStore((state) => state.projects);
  const isLoadingList = useProjectStore((state) => state.isLoadingList);
  const error = useProjectStore((state) => state.error);
  const refreshList = useProjectStore((state) => state.refreshList);
  const openProject = useProjectStore((state) => state.open);
  const renameProject = useProjectStore((state) => state.rename);
  const duplicateProject = useProjectStore((state) => state.duplicate);
  const exportProject = useProjectStore((state) => state.exportToFile);
  const deleteProject = useProjectStore((state) => state.remove);
  const importFromFile = useProjectStore((state) => state.importFromFile);
  const clearError = useProjectStore((state) => state.clearError);
  const resetDataset = useDataset((state) => state.reset);

  useEffect(() => {
    void refreshList();
    // Só na montagem: a lista já se mantém atualizada sozinha após cada ação (rename,
    // duplicate, delete, import e o checkpoint automático já chamam refreshList).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const mostRecent = projects[0];

  const handleOpen = async (id: string): Promise<void> => {
    await openProject(id);
    if (!useProjectStore.getState().error) navigate('workspace', id);
  };

  const handleNewBlank = (): void => {
    resetDataset();
    navigate('workspace');
  };

  const handleImportChange = (fileList: FileList | null): void => {
    const file = fileList?.[0];
    if (!file) return;
    void importFromFile(file);
    if (importInputRef.current) importInputRef.current.value = '';
  };

  return (
    <div className="relative min-h-screen text-foreground">
      <AmbientBackground />
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="container flex items-center justify-between gap-4 py-4">
          <span className="brand-mark h-9 text-foreground" role="img" aria-label="Simetrics" />
          <div className="flex items-center gap-2">
            <a
              href="https://scientata.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="eyebrow hidden items-center gap-1.5 px-2 text-foreground transition-colors hover:text-highlight sm:inline-flex"
            >
              {t('landing_family')}
              <ArrowUpRight className="size-3.5 text-highlight" aria-hidden />
            </a>
            <SettingsButton />
            <GithubButton />
          </div>
        </div>
      </header>

      <section className="relative overflow-hidden border-b border-border">
        <HeroOrbits />
        <div className="container relative flex flex-col items-center gap-8 py-20 text-center sm:py-28">
          <div className="eyebrow absolute left-6 top-6 hidden sm:block">SMTR / 001</div>
          <div className="eyebrow absolute right-6 top-6 hidden sm:block">27°35′ S — 48°32′ W</div>

          <p className="eyebrow flex items-center gap-3">
            <span className="h-px w-10 bg-highlight" aria-hidden />
            {t('landing_eyebrow')}
          </p>

          <h1 className="max-w-5xl text-[clamp(2.75rem,7.5vw,6.25rem)] font-medium leading-[0.9] tracking-[-0.06em]">
            {t('landing_hero_a')} <em className="accent-serif text-foreground">{t('landing_hero_em')}</em>{' '}
            {t('landing_hero_b')}{' '}
            <span className="text-highlight dark:[text-shadow:0_0_40px_rgb(184_255_74/0.45)]">
              {t('landing_hero_signal')}
            </span>
          </h1>

          <p className="max-w-2xl text-base leading-relaxed text-muted-foreground sm:text-lg">
            {t('landing_pitch')}
          </p>

          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3 pt-2">
            {mostRecent ? (
              <>
                <Button size="lg" className="cursor-pointer" onClick={() => void handleOpen(mostRecent.id)}>
                  {t('landing_cta_continue').replace('{name}', mostRecent.name)}
                  <ArrowRight aria-hidden />
                </Button>
                <Button variant="outline" size="lg" className="cursor-pointer" onClick={handleNewBlank}>
                  {t('landing_cta_new_blank')}
                </Button>
              </>
            ) : (
              <Button size="lg" className="cursor-pointer" onClick={handleNewBlank}>
                {t('landing_cta_start')}
                <ArrowDown aria-hidden />
              </Button>
            )}
            <button
              type="button"
              onClick={onOpenTutorial}
              className="cursor-pointer border-b border-border pb-1 text-sm transition-colors hover:border-highlight hover:text-highlight"
            >
              {t('landing_cta_tutorial')} ↗
            </button>
          </div>
        </div>

        <div className="container relative">
          <div className="grid border-t border-border sm:grid-cols-3">
            {HIGHLIGHTS.map(({ labelKey, textKey }, index) => (
              <div
                key={labelKey}
                className="flex flex-col gap-2 border-border py-6 text-left sm:px-6 sm:[&:not(:first-child)]:border-l [&:not(:first-child)]:border-t sm:[&:not(:first-child)]:border-t-0"
              >
                <p className="flex items-baseline gap-3 text-sm font-semibold">
                  <span className="font-mono text-[11px] font-normal text-highlight">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  {t(labelKey)}
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">{t(textKey)}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <main className="container flex flex-col gap-20 py-16 sm:py-20">
        <section className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3 border-b border-border pb-4">
            <div className="space-y-3">
              <p className="eyebrow">— {t('landing_projects_eyebrow')}</p>
              <h2 className="text-3xl font-medium leading-none tracking-[-0.05em] sm:text-4xl">
                {t('landing_projects_title')}
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <input
                ref={importInputRef}
                type="file"
                accept="application/json"
                onChange={(event) => handleImportChange(event.target.files)}
                className="hidden"
                id="simetrics-import-project"
              />
              <Button asChild variant="outline" size="sm" className="header-chip cursor-pointer">
                <label htmlFor="simetrics-import-project">
                  <Upload className="size-4" aria-hidden />
                  {t('landing_projects_import')}
                </label>
              </Button>
            </div>
          </div>

          {error && (
            <div className="flex items-center justify-between gap-3 border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive animate-in fade-in-0">
              <span>{error}</span>
              <button
                type="button"
                onClick={clearError}
                className="shrink-0 font-medium underline underline-offset-2 cursor-pointer"
              >
                {t('landing_dismiss_error')}
              </button>
            </div>
          )}

          {projects.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {projects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onOpen={(id) => void handleOpen(id)}
                  onRename={(id, name) => void renameProject(id, name)}
                  onDuplicate={(id) => void duplicateProject(id)}
                  onExport={(id) => void exportProject(id)}
                  onDelete={(id) => void deleteProject(id)}
                />
              ))}
            </div>
          ) : (
            !isLoadingList && (
              <EmptyState
                title={t('landing_projects_empty_title')}
                description={t('landing_projects_empty_desc')}
              />
            )
          )}
        </section>

        <section className="grid w-full gap-8 border-t border-border pt-16 md:grid-cols-[1fr_2fr]">
          <div className="space-y-3">
            <p className="eyebrow">— {t('landing_faq_eyebrow')}</p>
            <h2 className="text-3xl font-medium leading-none tracking-[-0.05em] sm:text-4xl">
              {t('landing_faq_title')}
            </h2>
          </div>
          <Accordion type="single" collapsible className="border-t border-border">
            {FAQ_ITEMS.map(({ questionKey, answerKey }) => (
              <AccordionItem key={questionKey} value={questionKey} className="last:border-b">
                <AccordionTrigger className="cursor-pointer text-left">{t(questionKey)}</AccordionTrigger>
                <AccordionContent>{t(answerKey)}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>
      </main>

      <footer className="border-t border-border py-8">
        <div className="container flex flex-col items-center justify-between gap-3 text-center sm:flex-row sm:text-left">
          <span className="eyebrow">Simetrics · {t('app_subtitle')}</span>
          <a
            href="https://gustavosimas.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground"
          >
            {t('developed_by')} <span className="accent-serif text-base">Gustavo Simas</span>
          </a>
        </div>
      </footer>
    </div>
  );
}
