import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { BookOpen, ChevronDown, ExternalLink, FileText } from 'lucide-react';

import { Collapse } from '@/components/Collapse';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { isNullLike, splitTokens, toNumeric } from '@/core/text';
import { FIELD } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

import { cleanDoiUrl } from './doi';

/**
 * Documentos do dossiê, dos mais para os menos citados.
 *
 * Um país ou uma palavra-chave pode reunir milhares de documentos, cada um com título,
 * palavras-chave e resumo. Acima de VIRTUALIZE_FROM a lista monta só as linhas à vista
 * (mais uma folga) e troca as demais por espaçadores da mesma altura: a rolagem e a barra
 * continuam as de uma lista completa. Abaixo disso, monta todas, como antes.
 */

const VIRTUALIZE_FROM = 40;
/** Altura média de uma linha fechada (título, palavras-chave e o botão do resumo). */
const ESTIMATED_ROW_HEIGHT = 120;
/** Linhas montadas além da área visível, para a rolagem e o Tab não encontrarem vazio. */
const OVERSCAN = 6;
/** Cabeçalho da tabela (h-10), que rola junto, acima da primeira linha. */
const HEADER_HEIGHT = 40;
/** max-h-[36rem] do contêiner: a primeira renderização já monta as linhas certas. */
const VIEWPORT_HEIGHT = 576;

export interface DossierColumns {
  title: string | null;
  keywords: string | null;
  abstract: string | null;
  doi: string | null;
}

interface DossierDocumentsProps {
  documents: Dataset;
  columns: DossierColumns;
  onOpenDocument: (title: string) => void;
  onOpenKeyword: (keyword: string) => void;
}

export function DossierDocuments({ documents, columns, onOpenDocument, onOpenKeyword }: DossierDocumentsProps) {
  const locale = useLocale((state) => state.locale);

  const sorted = useMemo(
    () =>
      [...documents].sort(
        (left, right) =>
          (toNumeric(right[FIELD.TOTAL_CITATIONS]) ?? 0) - (toNumeric(left[FIELD.TOTAL_CITATIONS]) ?? 0),
      ),
    [documents],
  );

  // O estado dos resumos vive aqui: abrir um deles redesenha a lista, não o dossiê inteiro.
  // Trocar de entidade remonta o dossiê (chave por perfil), e com ele este estado.
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(() => new Set());
  const toggle = useCallback((index: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualize = sorted.length > VIRTUALIZE_FROM;
  // Mesma limitação do TanStack Table em DataTable: o React Compiler não memoiza o
  // retorno. Sem consequência aqui: o que atravessa o `memo` das linhas é só o
  // `measureElement`, um método fixo da instância.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: sorted.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ESTIMATED_ROW_HEIGHT,
    overscan: OVERSCAN,
    scrollMargin: HEADER_HEIGHT,
    initialRect: { width: 0, height: VIEWPORT_HEIGHT },
    enabled: virtualize,
  });

  const items = virtualize ? virtualizer.getVirtualItems() : null;
  const indexes = items ? items.map((item) => item.index) : sorted.map((_, index) => index);
  const first = items?.[0];
  const last = items?.[items.length - 1];
  const paddingTop = first ? first.start - HEADER_HEIGHT : 0;
  const paddingBottom = last ? virtualizer.getTotalSize() - (last.end - HEADER_HEIGHT) : 0;

  return (
    <div ref={scrollRef} className="max-h-[36rem] overflow-auto rounded-xl border">
      <Table aria-rowcount={sorted.length + 1}>
        <TableHeader>
          <TableRow aria-rowindex={1}>
            {/* No celular a tabela é mais larga que a tela e a coluna de título fica na largura
                mínima do conteúdo. Com só algumas linhas montadas, essa largura variaria durante
                a rolagem: aqui ela é fixa, e palavras longas quebram dentro da célula. */}
            <TableHead className="min-w-36">{locale === 'en' ? 'Title & Details' : 'Título e Detalhes'}</TableHead>
            <TableHead className="w-24 text-center">{locale === 'en' ? 'Year' : 'Ano'}</TableHead>
            <TableHead className="w-24 text-center">{locale === 'en' ? 'Citations' : 'Citações'}</TableHead>
            <TableHead className="w-48">Venue</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {paddingTop > 0 && <tr aria-hidden style={{ height: paddingTop }} />}
          {indexes.map((index) => {
            const doc = sorted[index];
            if (!doc) return null;
            const title = columns.title ? String(doc[columns.title] ?? '').trim() : '';
            return (
              <DocumentRow
                key={`${title}-${index}`}
                doc={doc}
                title={title}
                index={index}
                columns={columns}
                expanded={expanded.has(index)}
                onToggle={toggle}
                onOpen={onOpenDocument}
                onOpenKeyword={onOpenKeyword}
                measureRef={virtualize ? virtualizer.measureElement : undefined}
              />
            );
          })}
          {paddingBottom > 0 && <tr aria-hidden style={{ height: paddingBottom }} />}
        </TableBody>
      </Table>
    </div>
  );
}

interface DocumentRowProps {
  doc: SimetricsDoc;
  title: string;
  index: number;
  columns: DossierColumns;
  expanded: boolean;
  onToggle: (index: number) => void;
  onOpen: (title: string) => void;
  onOpenKeyword: (keyword: string) => void;
  measureRef: ((node: HTMLTableRowElement | null) => void) | undefined;
}

const DocumentRow = memo(function DocumentRow({
  doc,
  title,
  index,
  columns,
  expanded,
  onToggle,
  onOpen,
  onOpenKeyword,
  measureRef,
}: DocumentRowProps) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const keywords = columns.keywords ? String(doc[columns.keywords] ?? '').trim() : '';
  const abstract = columns.abstract ? String(doc[columns.abstract] ?? '').trim() : '';
  const rawDoi = columns.doi ? doc[columns.doi] : doc[FIELD.DOI];
  const doiUrl = cleanDoiUrl(rawDoi);

  return (
    <TableRow ref={measureRef} data-index={index} aria-rowindex={index + 2} className="align-top">
      <TableCell className="space-y-2 py-3 [overflow-wrap:anywhere]">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="text-left font-semibold text-primary hover:underline cursor-pointer break-words leading-snug"
            title={title}
            onClick={() => onOpen(title)}
          >
            {title || '—'}
          </button>

          {doiUrl && (
            <a
              href={doiUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-md border border-cyan-300 bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-cyan-800 hover:bg-cyan-100 dark:border-cyan-800 dark:bg-cyan-950/60 dark:text-cyan-300"
              title={doiUrl}
            >
              <ExternalLink className="size-3" />
              {t('search_doi_link')}
            </a>
          )}
        </div>

        {keywords && !isNullLike(keywords) && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold text-muted-foreground flex items-center gap-1">
              <BookOpen className="size-3" />
              {t('search_keywords')}:
            </span>
            {[...new Set(splitTokens(keywords))].slice(0, 6).map((kw) => (
              <button
                key={kw}
                type="button"
                onClick={() => onOpenKeyword(kw)}
                title={locale === 'en' ? `Search keyword: ${kw}` : `Buscar palavra-chave: ${kw}`}
                className="cursor-pointer"
              >
                <Badge
                  variant="secondary"
                  className="px-1.5 py-0 text-[10px] transition-colors hover:border-highlight hover:text-highlight"
                >
                  {kw}
                </Badge>
              </button>
            ))}
          </div>
        )}

        {abstract && !isNullLike(abstract) && (
          <div className="pt-1">
            <button
              type="button"
              onClick={() => onToggle(index)}
              aria-expanded={expanded}
              className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
            >
              <FileText className="size-3" />
              <span>{t('search_abstract')}</span>
              <ChevronDown
                className={cn(
                  'size-3 transition-transform duration-300',
                  expanded && 'rotate-180',
                )}
              />
            </button>

            {/* Abre animando a altura, sem saltar o conteúdo abaixo. */}
            <Collapse open={expanded} delayOpen={false}>
              <p className="mt-1.5 border border-border/60 bg-muted/40 p-2.5 text-xs text-muted-foreground leading-relaxed">
                {abstract}
              </p>
            </Collapse>
          </div>
        )}
      </TableCell>
      <TableCell className="text-center tabular-nums font-medium py-3">
        {toNumeric(doc[FIELD.YEAR_CLEAN]) ?? '—'}
      </TableCell>
      <TableCell className="text-center tabular-nums font-semibold text-foreground py-3">
        {toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0}
      </TableCell>
      <TableCell
        className="max-w-48 truncate text-xs text-muted-foreground py-3"
        title={String(doc[FIELD.SECONDARY_TITLE] ?? '')}
      >
        {String(doc[FIELD.SECONDARY_TITLE] ?? '—')}
      </TableCell>
    </TableRow>
  );
});
