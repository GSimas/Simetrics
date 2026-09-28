import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { CalendarDays, ChevronLeft, ChevronRight } from 'lucide-react';

import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

/**
 * Seletor de data do Simetrics, no lugar do `<input type="date">` — cujo calendário é do
 * sistema operacional e muda em cada navegador e celular.
 *
 * O painel usa o popover nativo, como as dicas "i" (InfoTip): vai para a camada superior,
 * acima de modais e fora de qualquer `overflow`, e fecha com Esc e clique fora. O valor
 * é o mesmo texto ISO do input nativo (`AAAA-MM-DD`), então troca um pelo outro sem
 * mudar o dado salvo.
 */

export interface DatePickerProps {
  id?: string;
  /** Data em ISO (`AAAA-MM-DD`) ou vazia. */
  value: string;
  onValueChange: (value: string | null) => void;
  disabled?: boolean;
  className?: string;
  'aria-labelledby'?: string;
}

type View = 'days' | 'months' | 'years';

interface Ymd {
  y: number;
  m: number; // 0–11
  d: number;
}

const GAP = 6;

function parseIso(value: string): Ymd | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  return { y: Number(match[1]), m: Number(match[2]) - 1, d: Number(match[3]) };
}

function toIso({ y, m, d }: Ymd): string {
  return `${String(y).padStart(4, '0')}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

function today(): Ymd {
  const now = new Date();
  return { y: now.getFullYear(), m: now.getMonth(), d: now.getDate() };
}

function daysIn(y: number, m: number): number {
  return new Date(y, m + 1, 0).getDate();
}

/** Soma dias com a aritmética do `Date` (vira mês e ano sozinha). */
function addDays(date: Ymd, amount: number): Ymd {
  const next = new Date(date.y, date.m, date.d + amount);
  return { y: next.getFullYear(), m: next.getMonth(), d: next.getDate() };
}

function addMonths(date: Ymd, amount: number): Ymd {
  const first = new Date(date.y, date.m + amount, 1);
  const y = first.getFullYear();
  const m = first.getMonth();
  return { y, m, d: Math.min(date.d, daysIn(y, m)) };
}

const same = (a: Ymd | null, b: Ymd | null) => !!a && !!b && a.y === b.y && a.m === b.m && a.d === b.d;

const TEXT = {
  pt: {
    placeholder: 'Selecione uma data',
    today: 'Hoje',
    clear: 'Limpar',
    prevMonth: 'Mês anterior',
    nextMonth: 'Próximo mês',
    prevYear: 'Ano anterior',
    nextYear: 'Próximo ano',
    prevYears: 'Anos anteriores',
    nextYears: 'Próximos anos',
    chooseMonth: 'Escolher mês e ano',
    chooseYear: 'Escolher ano',
    dialog: 'Calendário',
  },
  en: {
    placeholder: 'Pick a date',
    today: 'Today',
    clear: 'Clear',
    prevMonth: 'Previous month',
    nextMonth: 'Next month',
    prevYear: 'Previous year',
    nextYear: 'Next year',
    prevYears: 'Earlier years',
    nextYears: 'Later years',
    chooseMonth: 'Choose month and year',
    chooseYear: 'Choose year',
    dialog: 'Calendar',
  },
};

export function DatePicker({ id, value, onValueChange, disabled, className, ...aria }: DatePickerProps) {
  const locale = useLocale((state) => state.locale);
  const text = TEXT[locale];
  const tag = locale === 'en' ? 'en' : 'pt-BR';
  const panelId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const selected = parseIso(value);
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<View>('days');
  const [cursor, setCursor] = useState<Ymd>(() => selected ?? today());
  const [focusDay, setFocusDay] = useState(false);

  const formats = useMemo(
    () => ({
      trigger: new Intl.DateTimeFormat(tag, { day: '2-digit', month: 'short', year: 'numeric' }),
      title: new Intl.DateTimeFormat(tag, { month: 'long', year: 'numeric' }),
      monthShort: new Intl.DateTimeFormat(tag, { month: 'short' }),
      weekday: new Intl.DateTimeFormat(tag, { weekday: 'narrow' }),
      weekdayLong: new Intl.DateTimeFormat(tag, { weekday: 'long' }),
      full: new Intl.DateTimeFormat(tag, { dateStyle: 'full' }),
    }),
    [tag],
  );

  const place = (): void => {
    const anchor = triggerRef.current?.getBoundingClientRect();
    const panel = panelRef.current;
    if (!anchor || !panel) return;
    const width = panel.offsetWidth || 288;
    const height = panel.offsetHeight;
    const left = Math.min(Math.max(GAP, anchor.left), window.innerWidth - width - GAP);
    const fitsBelow = anchor.bottom + GAP + height <= window.innerHeight - GAP;
    const top = fitsBelow ? anchor.bottom + GAP : Math.max(GAP, anchor.top - GAP - height);
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  };

  // Aberto, o painel acompanha o campo quando a página rola ou muda de tamanho.
  useEffect(() => {
    if (!open) return;
    const follow = (): void => place();
    window.addEventListener('scroll', follow, true);
    window.addEventListener('resize', follow);
    return () => {
      window.removeEventListener('scroll', follow, true);
      window.removeEventListener('resize', follow);
    };
  });

  // Navegando pelo teclado, o foco acompanha o dia do cursor.
  useEffect(() => {
    if (!open || view !== 'days' || !focusDay) return;
    gridRef.current?.querySelector<HTMLButtonElement>(`[data-date="${toIso(cursor)}"]`)?.focus();
  }, [open, view, cursor, focusDay]);

  const close = (): void => {
    panelRef.current?.hidePopover();
    triggerRef.current?.focus();
  };

  const choose = (date: Ymd | null): void => {
    onValueChange(date ? toIso(date) : null);
    close();
  };

  const onGridKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    const moves: Record<string, () => Ymd> = {
      ArrowLeft: () => addDays(cursor, -1),
      ArrowRight: () => addDays(cursor, 1),
      ArrowUp: () => addDays(cursor, -7),
      ArrowDown: () => addDays(cursor, 7),
      PageUp: () => addMonths(cursor, event.shiftKey ? -12 : -1),
      PageDown: () => addMonths(cursor, event.shiftKey ? 12 : 1),
      Home: () => addDays(cursor, -new Date(cursor.y, cursor.m, cursor.d).getDay()),
      End: () => addDays(cursor, 6 - new Date(cursor.y, cursor.m, cursor.d).getDay()),
    };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    setFocusDay(true);
    setCursor(move());
  };

  // Grade de 6 semanas começando no domingo, com os dias vizinhos esmaecidos.
  const cells = useMemo(() => {
    const offset = new Date(cursor.y, cursor.m, 1).getDay();
    const start = addDays({ y: cursor.y, m: cursor.m, d: 1 }, -offset);
    return Array.from({ length: 42 }, (_, index) => addDays(start, index));
  }, [cursor.y, cursor.m]);

  const weekdays = useMemo(
    () => Array.from({ length: 7 }, (_, index) => new Date(2023, 0, 1 + index)), // 1/1/2023 foi domingo
    [],
  );

  const now = today();
  // Só a primeira letra: "setembro de 2026" vira "Setembro de 2026", não "Setembro De 2026".
  const capitalize = (label: string) => label.charAt(0).toLocaleUpperCase(tag) + label.slice(1);
  const yearStart = cursor.y - (((cursor.y % 12) + 12) % 12);

  const navButton =
    'inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-[color,background-color,transform] duration-150 hover:bg-highlight/10 hover:text-highlight active:scale-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&_svg]:size-4';
  const cellBase =
    'inline-flex items-center justify-center rounded-md text-sm tabular-nums transition-[color,background-color,box-shadow,transform] duration-150 active:scale-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring';
  const cellActive = 'bg-highlight font-semibold text-primary-foreground shadow-[0_0_16px_-4px_var(--highlight)]';

  const shift = (amount: number): void =>
    setCursor((current) =>
      view === 'days' ? addMonths(current, amount) : { ...current, y: current.y + amount * (view === 'years' ? 12 : 1) },
    );

  return (
    <>
      <button
        ref={triggerRef}
        id={id}
        type="button"
        disabled={disabled}
        popoverTarget={panelId}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-labelledby={aria['aria-labelledby']}
        onClick={() => {
          setView('days');
          setFocusDay(false);
          setCursor(selected ?? today());
        }}
        className={cn(
          'group inline-flex h-9 w-full items-center gap-2 rounded-md border border-input bg-transparent px-3 text-left text-sm shadow-sm',
          'transition-[border-color,box-shadow] duration-200 hover:border-highlight/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          'disabled:cursor-not-allowed disabled:opacity-50',
          open && 'border-highlight shadow-[0_0_20px_-10px_var(--highlight)]',
          className,
        )}
      >
        <CalendarDays
          className={cn(
            'size-4 shrink-0 text-muted-foreground transition-colors duration-200 group-hover:text-highlight',
            open && 'text-highlight',
          )}
          aria-hidden
        />
        <span className={cn('flex-1 truncate', !selected && 'text-muted-foreground')}>
          {selected ? formats.trigger.format(new Date(selected.y, selected.m, selected.d)) : text.placeholder}
        </span>
      </button>

      <div
        ref={panelRef}
        id={panelId}
        popover="auto"
        role="dialog"
        aria-label={text.dialog}
        onToggle={(event) => {
          const isOpen = event.newState === 'open';
          setOpen(isOpen);
          if (isOpen) {
            place();
            // Abre com o foco no dia escolhido (ou hoje), como um calendário de verdade.
            setFocusDay(true);
          }
        }}
        className="fixed inset-auto m-0 w-72 max-w-[calc(100vw-12px)] rounded-xl border border-border bg-popover p-3 text-popover-foreground shadow-xl"
      >
        <div className="mb-2 flex items-center justify-between gap-1">
          <button
            type="button"
            onClick={() => shift(-1)}
            aria-label={view === 'days' ? text.prevMonth : view === 'months' ? text.prevYear : text.prevYears}
            className={navButton}
          >
            <ChevronLeft />
          </button>
          <button
            type="button"
            onClick={() => setView(view === 'days' ? 'months' : 'years')}
            disabled={view === 'years'}
            aria-label={view === 'days' ? text.chooseMonth : text.chooseYear}
            className="rounded-md px-2 py-1 text-sm font-semibold transition-colors duration-150 hover:bg-highlight/10 hover:text-highlight disabled:pointer-events-none"
          >
            {view === 'days'
              ? capitalize(formats.title.format(new Date(cursor.y, cursor.m, 1)))
              : view === 'months'
                ? cursor.y
                : `${yearStart} – ${yearStart + 11}`}
          </button>
          <button
            type="button"
            onClick={() => shift(1)}
            aria-label={view === 'days' ? text.nextMonth : view === 'months' ? text.nextYear : text.nextYears}
            className={navButton}
          >
            <ChevronRight />
          </button>
        </div>

        {view === 'days' && (
          <div key={`${cursor.y}-${cursor.m}`} className="animate-in fade-in-0 duration-200">
            <div className="mb-1 grid grid-cols-7 text-center">
              {weekdays.map((day) => (
                <abbr
                  key={day.getDay()}
                  title={formats.weekdayLong.format(day)}
                  className="eyebrow py-1 text-[10px] no-underline"
                >
                  {formats.weekday.format(day)}
                </abbr>
              ))}
            </div>
            <div ref={gridRef} role="grid" className="grid grid-cols-7 gap-0.5" onKeyDown={onGridKey}>
              {cells.map((cell) => {
                const outside = cell.m !== cursor.m;
                const isSelected = same(cell, selected);
                const isToday = same(cell, now);
                const isCursor = same(cell, cursor);
                return (
                  <button
                    key={toIso(cell)}
                    type="button"
                    role="gridcell"
                    data-date={toIso(cell)}
                    tabIndex={isCursor ? 0 : -1}
                    aria-selected={isSelected}
                    aria-label={formats.full.format(new Date(cell.y, cell.m, cell.d))}
                    onClick={() => choose(cell)}
                    className={cn(
                      cellBase,
                      'h-9',
                      isSelected
                        ? cellActive
                        : cn(
                            'hover:bg-highlight/10 hover:text-highlight',
                            outside ? 'text-muted-foreground/50' : 'text-foreground',
                            isToday && 'ring-1 ring-inset ring-highlight/60',
                          ),
                    )}
                  >
                    {cell.d}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {view === 'months' && (
          <div key={`m-${cursor.y}`} className="grid grid-cols-3 gap-1 animate-in fade-in-0 zoom-in-95 duration-200">
            {Array.from({ length: 12 }, (_, month) => {
              const active = selected?.y === cursor.y && selected.m === month;
              return (
                <button
                  key={month}
                  type="button"
                  onClick={() => {
                    setCursor({ y: cursor.y, m: month, d: Math.min(cursor.d, daysIn(cursor.y, month)) });
                    setView('days');
                  }}
                  className={cn(cellBase, 'h-10', active ? cellActive : 'hover:bg-highlight/10 hover:text-highlight')}
                >
                  {capitalize(formats.monthShort.format(new Date(cursor.y, month, 1)).replace('.', ''))}
                </button>
              );
            })}
          </div>
        )}

        {view === 'years' && (
          <div key={`y-${yearStart}`} className="grid grid-cols-3 gap-1 animate-in fade-in-0 zoom-in-95 duration-200">
            {Array.from({ length: 12 }, (_, index) => {
              const year = yearStart + index;
              const active = selected?.y === year;
              return (
                <button
                  key={year}
                  type="button"
                  onClick={() => {
                    setCursor({ y: year, m: cursor.m, d: Math.min(cursor.d, daysIn(year, cursor.m)) });
                    setView('months');
                  }}
                  className={cn(
                    cellBase,
                    'h-10',
                    active ? cellActive : cn('hover:bg-highlight/10 hover:text-highlight', year === now.y && 'ring-1 ring-inset ring-highlight/60'),
                  )}
                >
                  {year}
                </button>
              );
            })}
          </div>
        )}

        <div className="mt-2 flex items-center justify-between border-t border-border/80 pt-2">
          <button
            type="button"
            onClick={() => choose(null)}
            disabled={!selected}
            className="rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors duration-150 hover:bg-exclude/10 hover:text-exclude disabled:pointer-events-none disabled:opacity-40"
          >
            {text.clear}
          </button>
          <button
            type="button"
            onClick={() => choose(now)}
            className="rounded-md px-2 py-1 text-xs font-medium text-highlight transition-[background-color,transform] duration-150 hover:bg-highlight/10 active:scale-95"
          >
            {text.today}
          </button>
        </div>
      </div>
    </>
  );
}
