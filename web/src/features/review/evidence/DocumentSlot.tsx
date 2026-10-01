import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, BookOpenText, FileUp, Loader2, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import type { EvidenceTargetKey } from '@/core/review/types';
import { ingestPdf } from '@/lib/pdf';
import { deletePdf, getPdf } from '@/lib/pdf-store';
import { cn } from '@/lib/utils';
import { useReview, useReviewReadOnly } from '@/state/review.store';
import { fill, useEvidenceCopy } from './copy';
import { useEvidenceReader, type ReaderScope } from './reader-store';

function sizeLabel(bytes: number): string {
  return bytes >= 1_048_576 ? `${(bytes / 1_048_576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Guarda o PDF de um estudo e devolve o erro, se houver, no idioma da interface. */
async function attachPdf(studyKey: string, file: File, readFailed: string, notPdf: string): Promise<string | null> {
  if (file.type !== 'application/pdf' && !/\.pdf$/i.test(file.name)) return notPdf;
  try {
    const { document } = await ingestPdf(file);
    useReview.getState().attachDocument(studyKey, document);
    return null;
  } catch (cause) {
    return fill(readFailed, { error: cause instanceof Error ? cause.message : String(cause) });
  }
}

/** Remove o PDF do estudo e, se nenhum outro estudo da revisão o usa, também do navegador. */
async function detachPdf(studyKey: string): Promise<void> {
  const state = useReview.getState();
  const hash = state.review?.documents[studyKey]?.hash;
  state.detachDocument(studyKey);
  const stillUsed = Object.values(useReview.getState().review?.documents ?? {}).some((doc) => doc.hash === hash);
  // ponytail: outro projeto deste navegador pode usar o mesmo arquivo; ali o leitor pede para anexá-lo de novo.
  if (hash && !stillUsed) await deletePdf(hash).catch(() => undefined);
}

/**
 * O texto completo de um estudo, dentro do estudo aberto: anexar, trocar, remover e abrir
 * o leitor com as evidências da etapa.
 */
export function DocumentSlot({ studyKey, scope }: { studyKey: string; scope: ReaderScope }) {
  const copy = useEvidenceCopy();
  const document = useReview((state) => state.review?.documents[studyKey]);
  const readOnly = useReviewReadOnly();
  const openReader = useEvidenceReader((state) => state.open);
  const input = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // O arquivo está neste navegador? Vale para o documento conferido (hash + data de anexo).
  const [check, setCheck] = useState<{ id: string; ok: boolean } | null>(null);
  const documentId = document ? `${document.hash}:${document.addedAt}` : null;
  const available = documentId && check?.id === documentId ? check.ok : null;
  const [confirmRemove, setConfirmRemove] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!documentId) return;
    void getPdf(documentId.split(':')[0]!).then((stored) => !cancelled && setCheck({ id: documentId, ok: !!stored }));
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const pick = async (file: File | undefined): Promise<void> => {
    if (!file) return;
    setError(null);
    setBusy(true);
    setError(await attachPdf(studyKey, file, copy.readFailed, copy.notPdf));
    setBusy(false);
  };

  return (
    <div className="space-y-2 rounded-lg border border-border/80 bg-muted/20 p-3" data-tour="review-fulltext">
      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">{copy.fullText}</span>
        {document ? (
          <span className="min-w-0 truncate text-xs text-muted-foreground" title={document.name}>
            {document.name} · {fill(copy.pages, { n: document.pages })} · {sizeLabel(document.size)}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">{copy.noPdfYet}</span>
        )}
        <span className="ml-auto flex flex-wrap gap-1.5">
          {document && available && (
            <Button type="button" size="sm" className="active:scale-[0.97]" onClick={() => openReader({ studyKey, scope })}>
              <BookOpenText aria-hidden />
              {copy.openReader}
            </Button>
          )}
          {!readOnly && (
            <>
              <input
                ref={input}
                type="file"
                accept="application/pdf,.pdf"
                className="hidden"
                onChange={(event) => {
                  void pick(event.target.files?.[0]);
                  event.target.value = '';
                }}
              />
              <Button
                type="button"
                size="sm"
                variant={document && available ? 'ghost' : 'outline'}
                disabled={busy}
                className="active:scale-[0.97]"
                onClick={() => input.current?.click()}
              >
                {busy ? <Loader2 className="animate-spin" aria-hidden /> : <FileUp aria-hidden />}
                {busy ? copy.reading : document ? copy.replace : copy.attach}
              </Button>
              {document && (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="text-muted-foreground hover:text-exclude active:scale-[0.97]"
                  onClick={() => setConfirmRemove(true)}
                  aria-label={copy.remove}
                  title={copy.remove}
                >
                  <Trash2 aria-hidden />
                </Button>
              )}
            </>
          )}
        </span>
      </div>
      {document && available === false && <p className="text-xs text-amber-700 dark:text-amber-300">{copy.missingFile}</p>}
      {document && available && document.textLayer !== 'ok' && (
        <p className={cn('flex items-start gap-1.5 text-xs', document.textLayer === 'none' ? 'text-amber-700 dark:text-amber-300' : 'text-muted-foreground')}>
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          {document.textLayer === 'none' ? copy.scanned : copy.partial}
        </p>
      )}
      {error && (
        <p className="text-xs text-exclude" role="alert">
          {error}
        </p>
      )}
      <ConfirmDialog
        open={confirmRemove}
        onOpenChange={setConfirmRemove}
        title={copy.removeConfirm}
        description={copy.removeHint}
        confirmLabel={copy.remove}
        cancelLabel={copy.cancel}
        onConfirm={() => void detachPdf(studyKey)}
      />
    </div>
  );
}

/**
 * Embaixo de cada pergunta no formulário: o status da conferência e as páginas dos trechos,
 * que abrem o leitor direto neles. A conferência em si acontece no leitor, ao lado do PDF.
 */
export function EvidenceInline({ studyKey, target, scope }: { studyKey: string; target: EvidenceTargetKey; scope: ReaderScope }) {
  const copy = useEvidenceCopy();
  const entry = useReview((state) => state.review?.evidence[studyKey]?.[target]);
  const hasPdf = useReview((state) => !!state.review?.documents[studyKey]);
  const openReader = useEvidenceReader((state) => state.open);
  if (!entry) return null;
  const pending = entry.status === 'suggested';
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] animate-in fade-in-0 duration-200">
      <button
        type="button"
        disabled={!hasPdf}
        onClick={() => openReader({ studyKey, scope, target })}
        className={cn(
          'rounded-full border px-2 py-0.5 transition-colors',
          pending ? 'border-amber-500/60 bg-amber-500/10 text-amber-800 hover:bg-amber-500/20 dark:text-amber-200' : 'border-border text-muted-foreground hover:text-foreground',
        )}
      >
        {copy.status[entry.status]}
      </button>
      {entry.evidence.map((item) => (
        <button
          key={item.id}
          type="button"
          disabled={!hasPdf}
          onClick={() => openReader({ studyKey, scope, target, evidenceId: item.id })}
          title={item.quote || copy.areaLabel}
          className={cn(
            'inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 font-mono transition-colors hover:border-highlight hover:text-highlight disabled:pointer-events-none',
            item.location === 'not-found' ? 'border-exclude/50 text-exclude' : 'border-border text-muted-foreground',
          )}
        >
          {item.location === 'not-found' && <AlertTriangle className="size-3" aria-hidden />}
          {copy.page.slice(0, 1).toLowerCase()}. {item.page}
        </button>
      ))}
    </div>
  );
}
