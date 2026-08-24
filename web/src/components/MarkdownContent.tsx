import { Fragment, type ReactNode } from 'react';
import { ExternalLink } from 'lucide-react';

import { cn } from '@/lib/utils';

export interface MarkdownContentProps {
  content: string;
  className?: string;
}

/**
 * Renderizador de Markdown leve, seguro e reativo para mensagens de IA.
 * Suporta cabeçalhos, listas, tabelas estruturadas, negrito, itálico, blocos de código e links.
 */
export function MarkdownContent({ content, className }: MarkdownContentProps) {
  if (!content) return null;

  const normalized = normalizeMarkdownText(content);
  const blocks = parseMarkdownBlocks(normalized);

  return (
    <div className={cn('space-y-2 text-xs leading-relaxed break-words', className)}>
      {blocks.map((block, index) => (
        <Fragment key={index}>{renderBlock(block)}</Fragment>
      ))}
    </div>
  );
}

type Block =
  | { type: 'heading'; level: number; text: string }
  | { type: 'code'; language?: string | undefined; code: string }
  | { type: 'list'; ordered: boolean; items: string[] }
  | {
      type: 'table';
      headers: string[];
      alignments: ('left' | 'center' | 'right')[];
      rows: string[][];
    }
  | { type: 'quote'; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'divider' };

/** Normaliza quebras de linha e tabelas colapsadas (ex: '| |' -> '|\n|') */
function normalizeMarkdownText(raw: string): string {
  return raw
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\|\s*\|\s*(?=[^|])/g, '|\n|');
}

function isTableDelimiter(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes('|') || !trimmed.includes('-')) return false;
  // A linha de delimitação contém apenas |, -, :, espaços
  const withoutTableChars = trimmed.replace(/[|:\s-]/g, '');
  return withoutTableChars === '';
}

function splitTableCells(line: string): string[] {
  let trimmed = line.trim();
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
  if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1);
  return trimmed.split('|').map((cell) => cell.trim());
}

function parseTableAlignments(delimiterLine: string): ('left' | 'center' | 'right')[] {
  const cells = splitTableCells(delimiterLine);
  return cells.map((cell) => {
    const trimmed = cell.trim();
    const starts = trimmed.startsWith(':');
    const ends = trimmed.endsWith(':');
    if (starts && ends) return 'center';
    if (ends) return 'right';
    return 'left';
  });
}

function parseMarkdownBlocks(raw: string): Block[] {
  const lines = raw.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i] ?? '';

    // Bloco de Código Fenced (```)
    if (line.trim().startsWith('```')) {
      const language = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i]?.trim().startsWith('```')) {
        codeLines.push(lines[i] ?? '');
        i++;
      }
      i++; // consome o fechamento ```
      blocks.push({
        type: 'code',
        language: language || undefined,
        code: codeLines.join('\n'),
      });
      continue;
    }

    // Linha vazia
    if (!line.trim()) {
      i++;
      continue;
    }

    // Divisor Horizontal (--- ou ***)
    if (/^(\*{3,}|-{3,}|_{3,})$/.test(line.trim())) {
      blocks.push({ type: 'divider' });
      i++;
      continue;
    }

    // Cabeçalhos (#, ##, ###)
    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch && headingMatch[1] && headingMatch[2]) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      });
      i++;
      continue;
    }

    // Tabela Markdown (Linha com | seguida de linha delimitadora |:---|)
    if (line.includes('|') && i + 1 < lines.length && isTableDelimiter(lines[i + 1] ?? '')) {
      const headerLine = line;
      const delimiterLine = lines[i + 1] ?? '';
      const headers = splitTableCells(headerLine);
      const alignments = parseTableAlignments(delimiterLine);
      const rows: string[][] = [];

      i += 2; // pula cabeçalho e delimitador

      while (i < lines.length) {
        const rowLine = lines[i]?.trim() ?? '';
        if (!rowLine || !rowLine.includes('|') || isTableDelimiter(rowLine)) {
          break;
        }
        const cells = splitTableCells(rowLine);
        // Garante que o número de células não seja menor que os headers
        while (cells.length < headers.length) {
          cells.push('');
        }
        rows.push(cells);
        i++;
      }

      blocks.push({
        type: 'table',
        headers,
        alignments,
        rows,
      });
      continue;
    }

    // Blockquote (> ...)
    if (line.trim().startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i]?.trim().startsWith('>')) {
        quoteLines.push(lines[i]?.replace(/^>\s?/, '') ?? '');
        i++;
      }
      blocks.push({
        type: 'quote',
        text: quoteLines.join(' '),
      });
      continue;
    }

    // Lista Não-Ordenada (*, -, •)
    const unorderedMatch = line.match(/^(\s*)([-*•])\s+(.+)$/);
    if (unorderedMatch) {
      const items: string[] = [];
      while (i < lines.length) {
        const itemMatch = lines[i]?.match(/^(\s*)([-*•])\s+(.+)$/);
        if (itemMatch && itemMatch[3]) {
          items.push(itemMatch[3]);
          i++;
        } else if (lines[i]?.trim().startsWith(' ') && items.length > 0) {
          items[items.length - 1] += ' ' + lines[i]?.trim();
          i++;
        } else {
          break;
        }
      }
      blocks.push({
        type: 'list',
        ordered: false,
        items,
      });
      continue;
    }

    // Lista Ordenada (1., 2.)
    const orderedMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
    if (orderedMatch) {
      const items: string[] = [];
      while (i < lines.length) {
        const itemMatch = lines[i]?.match(/^(\s*)(\d+)\.\s+(.+)$/);
        if (itemMatch && itemMatch[3]) {
          items.push(itemMatch[3]);
          i++;
        } else if (lines[i]?.trim().startsWith(' ') && items.length > 0) {
          items[items.length - 1] += ' ' + lines[i]?.trim();
          i++;
        } else {
          break;
        }
      }
      blocks.push({
        type: 'list',
        ordered: true,
        items,
      });
      continue;
    }

    // Parágrafo Normal
    const paragraphLines: string[] = [];
    while (
      i < lines.length &&
      lines[i]?.trim() &&
      !lines[i]?.trim().startsWith('```') &&
      !lines[i]?.trim().startsWith('#') &&
      !lines[i]?.trim().startsWith('>') &&
      !lines[i]?.match(/^(\s*)([-*•]|\d+\.)\s+/) &&
      !(lines[i]?.includes('|') && i + 1 < lines.length && isTableDelimiter(lines[i + 1] ?? ''))
    ) {
      paragraphLines.push(lines[i] ?? '');
      i++;
    }

    if (paragraphLines.length > 0) {
      blocks.push({
        type: 'paragraph',
        text: paragraphLines.join(' '),
      });
    }
  }

  return blocks;
}

function renderBlock(block: Block): ReactNode {
  switch (block.type) {
    case 'heading': {
      if (block.level === 1) {
        return (
          <h3 className="mt-2.5 mb-1 text-sm font-extrabold text-foreground tracking-tight border-b border-border/60 pb-1">
            {renderInline(block.text)}
          </h3>
        );
      }
      if (block.level === 2) {
        return (
          <h4 className="mt-2 mb-1 text-xs font-bold text-foreground tracking-tight">
            {renderInline(block.text)}
          </h4>
        );
      }
      return (
        <h5 className="mt-1.5 mb-0.5 text-xs font-semibold text-foreground/90">
          {renderInline(block.text)}
        </h5>
      );
    }

    case 'table':
      return (
        <div className="my-2.5 overflow-x-auto rounded-xl border border-border/80 bg-card/90 shadow-2xs">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-100/90 text-muted-foreground dark:bg-slate-800/80 border-b border-border font-semibold">
              <tr>
                {block.headers.map((header, hIdx) => {
                  const align = block.alignments[hIdx] ?? 'left';
                  return (
                    <th
                      key={hIdx}
                      className={cn(
                        'px-3 py-2 text-[11px] font-bold text-foreground tracking-tight whitespace-nowrap',
                        align === 'center' && 'text-center',
                        align === 'right' && 'text-right',
                      )}
                    >
                      {renderInline(header)}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {block.rows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  className="transition-colors hover:bg-slate-50/80 dark:hover:bg-slate-800/40"
                >
                  {row.map((cell, cIdx) => {
                    const align = block.alignments[cIdx] ?? 'left';
                    return (
                      <td
                        key={cIdx}
                        className={cn(
                          'px-3 py-2 text-[11px] text-foreground/90 leading-snug',
                          align === 'center' && 'text-center',
                          align === 'right' && 'text-right font-mono',
                        )}
                      >
                        {renderInline(cell)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case 'code':
      return (
        <div className="my-2 overflow-hidden rounded-lg border border-border/80 bg-slate-950 text-slate-100 dark:bg-black">
          {block.language && (
            <div className="bg-slate-900 px-3 py-1 text-[10px] font-mono text-slate-400 border-b border-slate-800">
              {block.language}
            </div>
          )}
          <pre className="p-2.5 overflow-x-auto text-[11px] font-mono leading-relaxed">
            <code>{block.code}</code>
          </pre>
        </div>
      );

    case 'list': {
      if (block.ordered) {
        return (
          <ol className="my-1.5 pl-4 list-decimal space-y-1 text-foreground/90">
            {block.items.map((item, itemIdx) => (
              <li key={itemIdx} className="pl-0.5">
                {renderInline(item)}
              </li>
            ))}
          </ol>
        );
      }
      return (
        <ul className="my-1.5 pl-4 list-disc space-y-1 text-foreground/90 marker:text-emerald-500">
          {block.items.map((item, itemIdx) => (
            <li key={itemIdx} className="pl-0.5">
              {renderInline(item)}
            </li>
          ))}
        </ul>
      );
    }

    case 'quote':
      return (
        <blockquote className="my-2 border-l-2 border-emerald-500 pl-3 italic text-muted-foreground bg-emerald-50/30 dark:bg-emerald-950/20 py-1 rounded-r-md">
          {renderInline(block.text)}
        </blockquote>
      );

    case 'divider':
      return <hr className="my-2 border-border/60" />;

    case 'paragraph':
    default:
      return (
        <p className="my-1 text-foreground/90 leading-relaxed">
          {renderInline(block.text)}
        </p>
      );
  }
}

/**
 * Processador de formatação inline: **negrito**, *itálico*, `código`, [link](url).
 */
function renderInline(text: string): ReactNode {
  if (!text) return null;

  // Regex para capturar tags inline
  const pattern = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[[^\]]+\]\([^)]+\)|\*[^*]+\*|_[^_]+_)/g;
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    if (!part) return null;

    // Negrito: **texto** ou __texto__
    if ((part.startsWith('**') && part.endsWith('**')) || (part.startsWith('__') && part.endsWith('__'))) {
      const content = part.slice(2, -2);
      return (
        <strong key={index} className="font-bold text-foreground">
          {content}
        </strong>
      );
    }

    // Código inline: `código`
    if (part.startsWith('`') && part.endsWith('`')) {
      const content = part.slice(1, -1);
      return (
        <code
          key={index}
          className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] font-semibold text-primary"
        >
          {content}
        </code>
      );
    }

    // Link: [título](url)
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch && linkMatch[1] && linkMatch[2]) {
      return (
        <a
          key={index}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-0.5 font-semibold text-primary underline underline-offset-2 hover:text-primary/80"
        >
          <span>{linkMatch[1]}</span>
          <ExternalLink className="size-2.5 opacity-70" />
        </a>
      );
    }

    // Itálico: *texto* ou _texto_
    if ((part.startsWith('*') && part.endsWith('*')) || (part.startsWith('_') && part.endsWith('_'))) {
      const content = part.slice(1, -1);
      return (
        <em key={index} className="italic text-foreground/90">
          {content}
        </em>
      );
    }

    return <span key={index}>{part}</span>;
  });
}
