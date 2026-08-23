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

/**
 * Envio de arquivos e modo de demonstração — ⇄ a barra lateral (Geral.py:209-355).
 *
 * A atribuição de base por arquivo é mantida porque continua importando: o mesmo `.ris`
 * sai da Scopus e da Cochrane com convenções incompatíveis, e a extensão sozinha não
 * distingue.
 */

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
  const progress = useDataset((state) => state.progress);
  const error = useDataset((state) => state.error);
  const active = useDataset((state) => state.active);

  const busy = progress !== null;

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
      <CardHeader>
        <CardTitle className="text-base">Base de dados</CardTitle>
        <CardDescription>
          Formatos aceitos: RIS (SciELO, WoS, Scopus, Mendeley, Cochrane), CSV (Scopus,
          Cochrane), Excel (WoS) e TXT/NBIB (PubMed). Limite de{' '}
          {MAX_DOCUMENTS.toLocaleString('pt-BR')} documentos.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED}
            onChange={(event) => handleSelect(event.target.files)}
            className="hidden"
            id="simetrics-upload"
          />

          <Button asChild variant="outline" disabled={busy}>
            <label htmlFor="simetrics-upload" className="cursor-pointer">
              <FileUp aria-hidden />
              Selecionar arquivos
            </label>
          </Button>

          <Button variant="secondary" onClick={() => void loadDemo()} disabled={busy}>
            <Rocket aria-hidden />
            Carregar exemplo
          </Button>

          {active && (
            <>
              <span className="text-sm text-muted-foreground">
                {active.length.toLocaleString('pt-BR')} documentos carregados
              </span>
              <Button variant="ghost" size="sm" onClick={reset} disabled={busy}>
                <Trash2 aria-hidden />
                Limpar
              </Button>
            </>
          )}
        </div>

        {pending.length > 0 && (
          <div className="space-y-2 rounded-md border p-3">
            <p className="text-sm font-medium">Confirme a base de origem de cada arquivo</p>

            {pending.map((entry, index) => (
              <div key={entry.file.name} className="flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate text-sm" title={entry.file.name}>
                  {entry.file.name}
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

            <Button onClick={() => void handleProcess()} disabled={busy} className="w-full">
              Processar e integrar
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
