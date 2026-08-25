import { useRef, useState } from 'react';
import { FileUp, Rocket, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
    <Card className="border-t-4 border-t-primary shadow-xs">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-bold text-foreground">{t('upload_title')}</CardTitle>
        <CardDescription>
          {t('upload_description').replace('10.000', MAX_DOCUMENTS.toLocaleString('pt-BR'))}
        </CardDescription>
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

          <Button asChild variant="gradient" disabled={isIngesting} className="cursor-pointer font-medium">
            <label htmlFor="simetrics-upload">
              <FileUp className="size-4" aria-hidden />
              {t('upload_select_files')}
            </label>
          </Button>

          <Button variant="success" onClick={() => void loadDemo()} disabled={isIngesting} className="font-medium cursor-pointer">
            <Rocket className="size-4" aria-hidden />
            {t('upload_load_demo')}
          </Button>

          {active && (
            <>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800 shadow-2xs dark:border-emerald-900 dark:bg-emerald-950/60 dark:text-emerald-300">
                <span className="size-1.5 rounded-full bg-emerald-500" />
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

        {pending.length > 0 && (
          <div className="space-y-3 rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50/60 to-indigo-50/30 p-4 dark:border-blue-900/60 dark:from-blue-950/40 dark:to-indigo-950/20">
            <p className="text-sm font-semibold text-foreground">
              {t('upload_confirm_sources')} ({pending.length})
            </p>

            <div className="space-y-2">
              {pending.map((entry, index) => (
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
        )}

        {progress && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{progress.detail ? `${progress.phase} — ${progress.detail}` : progress.phase}</span>
              <span className="tabular-nums">{Math.round(progress.ratio * 100)}%</span>
            </div>
            <Progress value={progress.ratio * 100} />
          </div>
        )}

        {error && (
          <p className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
            {error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
