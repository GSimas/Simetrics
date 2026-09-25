import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Check, ChevronDown, Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

export interface SearchableSelectProps {
  options: readonly string[];
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyText?: string;
  className?: string;
  disabled?: boolean;
}

export function SearchableSelect({
  options,
  value,
  onChange,
  placeholder,
  searchPlaceholder,
  emptyText,
  className,
  disabled = false,
}: SearchableSelectProps) {
  const isEn = useLocale((s) => s.locale) === 'en';
  const placeholderText = placeholder ?? (isEn ? 'Select an option...' : 'Selecione uma opção...');
  const searchText = searchPlaceholder ?? (isEn ? 'Search...' : 'Buscar...');
  const emptyMessage = emptyText ?? (isEn ? 'No options found.' : 'Nenhuma opção encontrada.');
  const clearLabel = isEn ? 'Clear selection' : 'Limpar seleção';
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listId = useId();

  // Fecha ao clicar fora
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearch('');
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Foca no input de busca ao abrir
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => searchInputRef.current?.focus(), 50);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const filteredOptions = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return options.slice(0, 150);
    return options.filter((opt) => opt.toLowerCase().includes(q)).slice(0, 150);
  }, [options, search]);

  const handleSelect = (item: string) => {
    onChange(item);
    setIsOpen(false);
    setSearch('');
  };

  const handleToggle = () => {
    setIsOpen((prev) => {
      if (prev) setSearch('');
      return !prev;
    });
  };

  const handleClear = () => {
    onChange(null);
    triggerRef.current?.focus();
  };

  // Esc fecha a lista e devolve o foco ao seletor, como num select nativo.
  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== 'Escape' || !isOpen) return;
    event.preventDefault();
    setIsOpen(false);
    setSearch('');
    triggerRef.current?.focus();
  };

  return (
    <div ref={containerRef} className={cn('relative w-full', className)}>
      {/* Botão do Seletor Dropdown */}
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        onClick={handleToggle}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? listId : undefined}
        onKeyDown={handleKeyDown}
        className={cn(
          'flex h-10 w-full items-center justify-between rounded-xl border border-border/80 bg-card px-3 py-2 text-left text-xs sm:text-sm shadow-2xs transition-all hover:bg-muted/40 focus:outline-hidden focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50',
          isOpen && 'border-primary/60 ring-2 ring-primary/20',
        )}
      >
        <span className={cn('truncate', !value && 'text-muted-foreground')}>
          {value || placeholderText}
        </span>
        <div className="flex items-center gap-1.5 pl-2 text-muted-foreground">
          {/* Reserva o lugar do botão de limpar, que fica fora do seletor (abaixo): um
              botão dentro de outro é inacessível por teclado e leitor de tela. */}
          {value && <span className="block size-[18px]" aria-hidden />}
          <ChevronDown
            className={cn('size-4 transition-transform duration-200', isOpen && 'rotate-180 text-primary')}
            aria-hidden
          />
        </div>
      </button>

      {value && !disabled && (
        <button
          type="button"
          onClick={handleClear}
          aria-label={clearLabel}
          title={clearLabel}
          className="absolute right-[34px] top-1/2 -translate-y-1/2 rounded-full p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          <X className="size-3.5" aria-hidden />
        </button>
      )}

      {/* Lista Suspensa Flutuante */}
      {isOpen && (
        <div className="absolute left-0 top-full z-50 mt-1.5 w-full min-w-[280px] rounded-xl border border-border/90 bg-card p-2 shadow-xl animate-in fade-in-0 zoom-in-95">
          {/* Campo de Busca dentro da Lista Suspensa */}
          <div className="relative mb-2">
            <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
            <Input
              ref={searchInputRef}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={searchText}
              aria-label={searchText}
              aria-controls={listId}
              onKeyDown={handleKeyDown}
              className="h-8 pl-8 pr-3 text-xs bg-muted/30 rounded-lg"
            />
          </div>

          {/* Opções Roláveis */}
          <div id={listId} role="listbox" tabIndex={-1} onKeyDown={handleKeyDown} className="max-h-60 overflow-y-auto space-y-0.5 pr-1">
            {filteredOptions.length === 0 ? (
              <p className="p-3 text-center text-xs text-muted-foreground">{emptyMessage}</p>
            ) : (
              filteredOptions.map((option) => {
                const isSelected = value === option;
                return (
                  <button
                    key={option}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onClick={() => handleSelect(option)}
                    className={cn(
                      'flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-xs font-medium transition-colors',
                      isSelected
                        ? 'bg-primary text-primary-foreground font-semibold'
                        : 'text-foreground hover:bg-muted/80',
                    )}
                    title={option}
                  >
                    <span className="truncate pr-2">{option}</span>
                    {isSelected && <Check className="size-3.5 shrink-0" aria-hidden />}
                  </button>
                );
              })
            )}
          </div>

          {/* Rodapé informativo */}
          <div className="mt-2 border-t border-border/60 pt-1.5 px-1 flex items-center justify-between text-[11px] text-muted-foreground">
            <span>
              {options.length} {isEn
                ? options.length === 1 ? 'item available' : 'items available'
                : options.length === 1 ? 'item disponível' : 'itens disponíveis'}
            </span>
            {search && filteredOptions.length >= 150 && (
              <span className="text-[10px] text-amber-600 dark:text-amber-400">
                {isEn ? 'Refine the search' : 'Refine a busca'}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
