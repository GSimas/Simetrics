import { useRef, useState } from 'react';
import { FileUp, Rocket, Trash2 } from 'lucide-react';

import { Collapse } from '@/components/Collapse';
import { SectionTitle } from '@/components/InfoTip';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { suggestDatabase, type UploadedFile } from '@/core/parsers';
import { DATABASES, MAX_DOCUMENTS, type DatabaseName } from '@/lib/schema';
import { useStickyValue } from '@/lib/use-sticky-value';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';

const ACCEPTED = '.ris,.csv,.xls,.xlsx,.txt,.nbib';

interface PendingFile {
  file: File;
  database: DatabaseName;
}

export function UploadPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState<PendingFile[]>([]);

  const loadFiles = useDataset((state) => state.loadFiles);
  const loadDemo = useDataset((state) => state.loadDemo);
  const reset = useDataset((state) => state.reset);
  const isIngesting = useDataset((state) => state.isIngesting);
  const isDeduplicating = useDataset((state) => state.isDeduplicating);
  const progress = useDataset((state) => state.progress);
  const error = useDataset((state) => state.error);
  const active = useDataset((state) => state.active);
  const t = useLocale((state) => state.t);

  const busy = isIngesting || isDeduplicating;
  const shownProgress = useStickyValue(progress).value;
  const shownError = useStickyValue(error).value;
  const shownPending = useStickyValue(pending.length > 0 ? pending : null).value ?? [];

  const handleSelect = (fileList: FileList | null): void => {
    if (!fileList) return;
    setPending(
      [...fileList].map((file) => ({ file, database: suggestDatabase(file.name) })),
    );
  };

  const handleProcess = async (): Promise<void> => {
    const uploads: UploadedFile[] = await Promise.all(
      pending.map(async ({ file, database }) => ({
        name: file.name,
        buffer: await file.arrayBuffer(),
        database,
      })),
    );

    await loadFiles(uploads);
    setPending([]);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <SectionTitle
          title={t('upload_title')}
          info={t('upload_description').replace('10.000', MAX_DOCUMENTS.toLocaleString('pt-BR'))}
        />
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2.5">
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED}
            onChange={(event) => handleSelect(event.target.files)}
            className="hidden"
            id="simetrics-upload"
          />

          <Button asChild disabled={isIngesting} className="cursor-pointer">
            <label htmlFor="simetrics-upload">
              <FileUp className="size-4" aria-hidden />
              {t('upload_select_files')}
            </label>
          </Button>

          <Button variant="outline" onClick={() => void loadDemo()} disabled={isIngesting} className="cursor-pointer">
            <Rocket className="size-4" aria-hidden />
            {t('upload_load_demo')}
          </Button>

          {active && (
            <>
              <span className="eyebrow inline-flex items-center gap-2 px-1 text-foreground">
                <span className="size-1.5 rounded-full bg-highlight shadow-[0_0_10px_var(--highlight)]" />
                {active.length.toLocaleString('pt-BR')} {t('upload_loaded_count')}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={reset}
                disabled={isIngesting}
                className="text-muted-foreground hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-950 dark:hover:text-red-300 cursor-pointer"
              >
                <Trash2 className="size-4" aria-hidden />
                {t('upload_clear')}
              </Button>
            </>
          )}
        </div>

        {/* Confirmação das bases: abre e fecha animada; ao fechar mostra a última lista. */}
        <Collapse open={pending.length > 0} delayOpen={false} className={pending.length > 0 ? '' : 'mb-0'}>
            <div className="space-y-3 border border-border bg-muted/40 p-4">
              <p className="text-sm font-semibold text-foreground">
                {t('upload_confirm_sources')} ({shownPending.length})
              </p>

              <div className="space-y-2">
                {shownPending.map((entry, index) => (
                  <div
                    key={entry.file.name}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/80 bg-card p-2.5 shadow-2xs"
                  >
                    <span className="min-w-0 flex-1 truncate text-xs sm:text-sm font-medium" title={entry.file.name}>
                      📄 {entry.file.name}
                    </span>

                    <Select
                      value={entry.database}
                      onValueChange={(value) =>
                        setPending((current) =>
                          current.map((item, position) =>
                            position === index ? { ...item, database: value as DatabaseName } : item,
                          ),
                        )
                      }
                    >
                      <SelectTrigger className="h-8 w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {DATABASES.map((database) => (
                          <SelectItem key={database} value={database}>
                            {database}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>

              <Button
                variant="gradient"
                onClick={() => void handleProcess()}
                disabled={busy}
                className="w-full font-semibold shadow-xs"
              >
                {t('upload_process_btn')}
              </Button>
            </div>
        </Collapse>

        {/* Progresso e erro entram e saem animados. O -mt-4 anula o espaçamento do
            space-y quando os dois estão fechados; cada um repõe o seu com pt-4. O último
            valor fica na tela durante o fechamento, em vez de sumir antes da animação. */}
        <div className="-mt-4">
          <Collapse open={progress !== null}>
            {shownProgress && (
              <div className="space-y-1.5 pt-4">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>
                    {shownProgress.detail
                      ? `${shownProgress.phase} — ${shownProgress.detail}`
                      : shownProgress.phase}
                  </span>
                  <span className="tabular-nums">{Math.round(shownProgress.ratio * 100)}%</span>
                </div>
                <Progress value={shownProgress.ratio * 100} />
              </div>
            )}
          </Collapse>

          <Collapse open={error !== null}>
            {shownError && (
              <p className="mt-4 border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
                {shownError}
              </p>
            )}
          </Collapse>
        </div>
      </CardContent>
    </Card>
  );
}
