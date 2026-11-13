import { useEffect, useRef } from 'react';
import { ArrowRight, ArrowUpRight, FolderOpen, Upload } from 'lucide-react';

import { AmbientBackground, HeroOrbits } from '@/components/AmbientBackdrop';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { BrandMark } from '@/components/BrandMark';
import { SettingsButton } from '@/components/HeaderActions';
import { EmptyState } from '@/features/EmptyState';
import { ProjectCard } from '@/features/landing/ProjectCard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { TranslationKey } from '@/lib/i18n/translations';
import type { AppView } from '@/lib/use-hash-route';
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
  const startBlank = useProjectStore((state) => state.startBlank);

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
    startBlank();
    navigate('workspace');
  };

  const handleImportChange = (fileList: FileList | null): void => {
    const file = fileList?.[0];
    if (!file) return;
    void importFromFile(file);
    if (importInputRef.current) importInputRef.current.value = '';
  };

  const linkClass =
    'cursor-pointer border-b border-border pb-1 text-sm transition-colors hover:border-highlight hover:text-highlight';

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden text-foreground">
      <AmbientBackground />
      <header className="relative z-40 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="container flex items-center justify-between gap-4 py-4">
          <span role="img" aria-label="Simetrics" className="inline-flex text-foreground">
            <BrandMark className="h-9" />
          </span>
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
          </div>
        </div>
      </header>

      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <HeroOrbits />
        <div className="container relative flex flex-1 flex-col items-center justify-center gap-8 py-8 text-center">
          <div className="eyebrow absolute right-6 top-6 hidden sm:block">27°35′ S — 48°32′ W</div>

          <p className="eyebrow flex items-center gap-3">
            <span className="h-px w-10 bg-highlight" aria-hidden />
            {t('landing_eyebrow')}
          </p>

          <h1 className="max-w-5xl text-[clamp(2.5rem,min(7.5vw,10vh),6.25rem)] font-medium leading-[0.9] tracking-[-0.06em]">
            {t('landing_hero_a')} <em className="accent-serif text-foreground">{t('landing_hero_em')}</em>{' '}
            {t('landing_hero_and')} {t('landing_hero_a2')}{' '}
            <em className="accent-serif text-foreground">{t('landing_hero_em2')}</em> {t('landing_hero_b')}{' '}
            <span className="text-highlight dark:[text-shadow:0_0_40px_rgb(184_255_74/0.45)]">
              {t('landing_hero_signal')}
            </span>
          </h1>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
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
                <ArrowRight aria-hidden />
              </Button>
            )}

            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="lg" className="cursor-pointer">
                  <FolderOpen aria-hidden />
                  {t('landing_projects_title')}
                </Button>
              </DialogTrigger>
              <DialogContent className="max-h-[85dvh] max-w-5xl grid-rows-[auto_1fr] overflow-hidden">
                <DialogHeader className="flex-row flex-wrap items-end justify-between gap-3 space-y-0 border-b border-border pb-4 pr-10">
                  <div className="space-y-2">
                    <p className="eyebrow">— {t('landing_projects_eyebrow')}</p>
                    <DialogTitle className="text-3xl font-medium tracking-[-0.05em]">
                      {t('landing_projects_title')}
                    </DialogTitle>
                  </div>
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
                </DialogHeader>

                <div className="min-h-0 space-y-4 overflow-y-auto">
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
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3">
            <button type="button" onClick={onOpenTutorial} className={linkClass}>
              {t('landing_cta_tutorial')} ↗
            </button>

            <Dialog>
              <DialogTrigger asChild>
                <button type="button" className={linkClass}>
                  {t('landing_about_button')}
                </button>
              </DialogTrigger>
              <DialogContent className="max-h-[85dvh] max-w-4xl overflow-y-auto">
                <DialogTitle className="pr-10 text-3xl font-medium tracking-[-0.05em]">
                  {t('landing_about_button')}
                </DialogTitle>
                <div className="grid border-t border-border sm:grid-cols-3">
                  {HIGHLIGHTS.map(({ labelKey, textKey }, index) => (
                    <div
                      key={labelKey}
                      className="flex flex-col gap-2 border-border py-6 text-left sm:px-6 sm:first:pl-0 sm:last:pr-0 sm:[&:not(:first-child)]:border-l [&:not(:first-child)]:border-t sm:[&:not(:first-child)]:border-t-0"
                    >
                      <p className="flex items-baseline gap-3 text-sm font-semibold">
                        <span className="font-mono text-[0.6875rem] font-normal text-highlight">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        {t(labelKey)}
                      </p>
                      <p className="text-sm leading-relaxed text-muted-foreground">{t(textKey)}</p>
                    </div>
                  ))}
                </div>
              </DialogContent>
            </Dialog>

            <Dialog>
              <DialogTrigger asChild>
                <button type="button" className={linkClass}>
                  {t('landing_faq_title')}
                </button>
              </DialogTrigger>
              <DialogContent className="max-h-[90dvh] max-w-4xl overflow-y-auto">
                <div className="space-y-2 pr-10">
                  <p className="eyebrow">— {t('landing_faq_eyebrow')}</p>
                  <DialogTitle className="text-3xl font-medium tracking-[-0.05em]">
                    {t('landing_faq_title')}
                  </DialogTitle>
                </div>
                {/* Lista à esquerda, resposta à direita: altura fixa, a janela não cresce ao trocar de pergunta. */}
                <Tabs
                  orientation="vertical"
                  defaultValue={FAQ_ITEMS[0].questionKey}
                  className="grid border-t border-border sm:h-[min(22rem,55dvh)] sm:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]"
                >
                  <TabsList className="flex h-auto flex-col items-stretch justify-start overflow-y-auto border-b border-border sm:border-b-0 sm:border-r">
                    {FAQ_ITEMS.map(({ questionKey }) => (
                      <TabsTrigger
                        key={questionKey}
                        value={questionKey}
                        className="mb-0 cursor-pointer justify-start whitespace-normal border-b border-l-2 border-b-border border-l-transparent px-4 py-3 text-left leading-snug last:border-b-0 data-[state=active]:border-b-border data-[state=active]:border-l-highlight data-[state=active]:bg-accent/40"
                      >
                        {t(questionKey)}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {FAQ_ITEMS.map(({ questionKey, answerKey }) => (
                    <TabsContent
                      key={questionKey}
                      value={questionKey}
                      className="mt-0 h-56 overflow-y-auto py-4 text-sm leading-relaxed text-muted-foreground sm:h-auto sm:px-6"
                    >
                      <p className="mb-3 font-medium text-foreground">{t(questionKey)}</p>
                      {t(answerKey)}
                    </TabsContent>
                  ))}
                </Tabs>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </main>

      <footer className="relative border-t border-border py-4">
        <div className="container flex flex-col items-center justify-between gap-2 text-center sm:flex-row sm:text-left">
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
