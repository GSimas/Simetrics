import { useEffect, useRef } from 'react';
import { Network, PlayCircle, Sparkles, Upload, Zap } from 'lucide-react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
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
  {
    Icon: Zap,
    labelKey: 'landing_highlight_1_label' as TranslationKey,
    textKey: 'landing_highlight_1_text' as TranslationKey,
  },
  {
    Icon: Network,
    labelKey: 'landing_highlight_2_label' as TranslationKey,
    textKey: 'landing_highlight_2_text' as TranslationKey,
  },
  {
    Icon: Sparkles,
    labelKey: 'landing_highlight_3_label' as TranslationKey,
    textKey: 'landing_highlight_3_text' as TranslationKey,
  },
] as const;

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
    <div className="min-h-screen bg-background text-foreground">
      <div className="h-1.5 w-full bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600" />

      <main className="container flex flex-col gap-12 py-12 sm:py-16">
        <section className="flex flex-col items-center gap-6 text-center">
          <img src="/simetrics-logo.png" alt="Simetrics Logo" className="h-24 w-auto object-contain sm:h-28" />

          <div className="space-y-2">
            <h1 className="text-3xl font-extrabold tracking-tight text-foreground sm:text-4xl">
              {t('app_title')}
            </h1>
            <p className="text-base font-medium text-muted-foreground sm:text-lg">
              {t('app_subtitle')}
            </p>
          </div>

          <p className="max-w-2xl text-sm text-muted-foreground sm:text-base">
            {t('landing_pitch')}
          </p>

          <div className="grid w-full max-w-3xl gap-3 sm:grid-cols-3">
            {HIGHLIGHTS.map(({ Icon, labelKey, textKey }) => (
              <Card
                key={labelKey}
                className="group relative overflow-hidden border-border/80 text-left shadow-2xs transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:shadow-lg hover:shadow-primary/10"
              >
                <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100 bg-[radial-gradient(circle_at_50%_0%,theme(colors.primary/18%),transparent_70%)]" />
                <CardContent className="relative flex flex-col gap-2 p-4">
                  <Icon className="size-5 text-primary transition-transform duration-300 group-hover:scale-110" aria-hidden />
                  <p className="text-sm font-semibold text-foreground">{t(labelKey)}</p>
                  <p className="text-xs text-muted-foreground">{t(textKey)}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            {mostRecent ? (
              <>
                <Button
                  variant="gradient"
                  size="lg"
                  className="cursor-pointer font-semibold"
                  onClick={() => void handleOpen(mostRecent.id)}
                >
                  <PlayCircle className="size-5" aria-hidden />
                  {t('landing_cta_continue').replace('{name}', mostRecent.name)}
                </Button>
                <Button variant="outline" size="lg" className="cursor-pointer" onClick={handleNewBlank}>
                  {t('landing_cta_new_blank')}
                </Button>
              </>
            ) : (
              <Button variant="gradient" size="lg" className="cursor-pointer font-semibold" onClick={handleNewBlank}>
                <PlayCircle className="size-5" aria-hidden />
                {t('landing_cta_start')}
              </Button>
            )}
            <Button variant="ghost" size="lg" className="cursor-pointer" onClick={onOpenTutorial}>
              {t('landing_cta_tutorial')}
            </Button>
          </div>
        </section>

        <section className="mx-auto w-full max-w-3xl">
          <h2 className="mb-2 text-center text-lg font-bold text-foreground">
            {t('landing_faq_title')}
          </h2>
          <Accordion type="single" collapsible className="rounded-xl border border-border/80 bg-card px-4 shadow-2xs">
            {FAQ_ITEMS.map(({ questionKey, answerKey }) => (
              <AccordionItem key={questionKey} value={questionKey}>
                <AccordionTrigger className="cursor-pointer">{t(questionKey)}</AccordionTrigger>
                <AccordionContent>{t(answerKey)}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>

        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold text-foreground">{t('landing_projects_title')}</h2>

            <div className="flex items-center gap-2">
              <input
                ref={importInputRef}
                type="file"
                accept="application/json"
                onChange={(event) => handleImportChange(event.target.files)}
                className="hidden"
                id="simetrics-import-project"
              />
              <Button asChild variant="outline" size="sm" className="cursor-pointer font-medium">
                <label htmlFor="simetrics-import-project">
                  <Upload className="size-4" aria-hidden />
                  {t('landing_projects_import')}
                </label>
              </Button>
            </div>
          </div>

          {error && (
            <div className="flex items-center justify-between gap-3 rounded-md border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
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
      </main>
    </div>
  );
}
