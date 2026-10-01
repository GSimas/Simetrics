import { useState, type ReactNode } from 'react';
import { BarChart3, Database, ListChecks, Lock, PanelLeftClose, PanelLeftOpen, Search, type LucideIcon } from 'lucide-react';

import { BIBLIOMETRIC_VIEWS } from '@/features/bibliometrics/views';
import { REVIEW_COPY } from '@/features/review/copy';
import { ProgressRing } from '@/features/review/progress-parts';
import { REVIEW_STEP_ITEMS, stepDetail } from '@/features/review/progress-steps';
import { STEPS_WITHOUT_DATA } from '@/core/review/progress';
import { numberLocale } from '@/lib/i18n/labels';
import type { TranslationKey } from '@/lib/i18n/translations';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation, type WorkspaceTab } from '@/state/navigation.store';
import { useReviewNav, useReviewProgress } from '@/state/review.store';

/** A ordem da jornada: primeiro os dados, depois a busca, depois as análises. */
const MODULES: readonly { value: WorkspaceTab; labelKey: TranslationKey; Icon: LucideIcon }[] = [
  { value: 'data', labelKey: 'nav_data', Icon: Database },
  { value: 'search', labelKey: 'tab_search', Icon: Search },
  { value: 'bibliometrics', labelKey: 'tab_bibliometrics', Icon: BarChart3 },
  { value: 'review', labelKey: 'tab_review', Icon: ListChecks },
];

interface SubItem {
  key: string;
  label: string;
  Icon: LucideIcon;
  active: boolean;
  onSelect: () => void;
  onHover?: () => void;
  /** Etapa que precisa da base: aparece com cadeado e não abre. */
  locked?: boolean;
  /** Dica ao passar o mouse (ex.: "5/7 itens"). */
  hint?: string;
  trailing?: ReactNode;
}

const STORAGE_KEY = 'simetrics-sidebar';
/** Abaixo disto a barra aberta se sobrepõe ao conteúdo em vez de empurrá-lo. */
const WIDE = '(min-width: 1024px)';

function initialCollapsed(): boolean {
  // Na tela estreita a barra aberta cobre o conteúdo: começa sempre recolhida.
  if (!window.matchMedia(WIDE).matches) return true;
  try {
    return localStorage.getItem(STORAGE_KEY) === 'collapsed';
  } catch {
    return false;
  }
}

/**
 * Barra lateral do workspace. Recolhida, vira um trilho de ícones (com dicas); aberta,
 * mostra os nomes e as subpáginas aninhadas: as vistas da Análise Bibliométrica e as
 * etapas da Revisão, estas com o anel de completude de cada uma. Sem base carregada,
 * só Dados abre — os módulos aparecem com cadeado.
 */
export function WorkspaceSidebar() {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const activeTab = useNavigation((state) => state.activeTab);
  const bibliometricView = useNavigation((state) => state.bibliometricView);
  const setActiveTab = useNavigation((state) => state.setActiveTab);
  const documentCount = useDataset((state) => state.active?.length ?? 0);
  const reviewStep = useReviewNav((state) => state.step);
  const setReviewStep = useReviewNav((state) => state.setStep);
  const reviewProgress = useReviewProgress();
  const reviewCopy = REVIEW_COPY[locale];
  const [collapsed, setCollapsed] = useState(initialCollapsed);

  const toggle = (): void => {
    const next = !collapsed;
    setCollapsed(next);
    try {
      localStorage.setItem(STORAGE_KEY, next ? 'collapsed' : 'expanded');
    } catch {
      // Vale só nesta sessão.
    }
  };

  const open = (tab: string): void => {
    setActiveTab(tab);
    window.scrollTo({ top: 0 });
    // Na tela estreita a barra aberta cobre o conteúdo: escolher um item a recolhe.
    if (!collapsed && !window.matchMedia(WIDE).matches) setCollapsed(true);
  };

  const locked = documentCount === 0;

  const subItems: Partial<Record<WorkspaceTab, SubItem[]>> = {
    bibliometrics: BIBLIOMETRIC_VIEWS.map(({ value, labelKey, Icon, Panel }) => ({
      key: value,
      label: t(labelKey),
      Icon,
      active: bibliometricView === value,
      onSelect: () => open(value),
      onHover: () => void Panel.preload(),
    })),
    review: REVIEW_STEP_ITEMS.map(({ value, Icon }) => {
      const progress = reviewProgress.steps[value];
      const stepLocked = locked && !STEPS_WITHOUT_DATA.includes(value);
      return {
        key: value,
        label: reviewCopy.steps[value],
        Icon,
        active: reviewStep === value && !stepLocked,
        locked: stepLocked,
        hint: stepLocked ? reviewCopy.progressNeedsData : stepDetail(value, progress, reviewCopy),
        onSelect: () => {
          setReviewStep(value);
          open('review');
        },
        trailing: stepLocked ? <Lock className="size-3.5 shrink-0" aria-hidden /> : <ProgressRing progress={progress} />,
      };
    }),
  };

  return (
    <div className={cn('relative z-45 shrink-0 transition-[width] duration-200', collapsed ? 'w-14' : 'w-14 lg:w-64')}>
      <nav
        aria-label={t('nav_modules')}
        data-tour="tabs"
        className={cn(
          'sticky top-0 flex h-dvh flex-col overflow-y-auto overflow-x-hidden border-r border-border bg-background/80 backdrop-blur-md transition-[width] duration-200',
          collapsed ? 'w-14' : 'w-64 max-lg:shadow-2xl',
        )}
      >
        <ul className="flex flex-1 flex-col gap-1 p-2 pt-4">
          {MODULES.map(({ value, labelKey, Icon }, index) => {
            // A Revisão abre sem base: o protocolo se escreve antes da busca.
            const isLocked = locked && value !== 'data' && value !== 'review';
            const isActive = activeTab === value;
            const label = t(labelKey);
            return (
              <li key={value}>
                <button
                  type="button"
                  onClick={() => !isLocked && open(value)}
                  aria-current={isActive ? 'page' : undefined}
                  // aria-disabled, e não disabled: o item continua focável e a dica explica o cadeado.
                  aria-disabled={isLocked || undefined}
                  title={isLocked ? `${label} — ${t('nav_locked')}` : collapsed ? label : undefined}
                  className={cn(
                    'group relative flex w-full items-center gap-3 rounded-md py-2.5 text-left text-sm font-medium transition-colors',
                    collapsed ? 'justify-center px-0' : 'px-2.5',
                    'focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring',
                    isActive
                      ? 'bg-highlight/10 text-foreground before:absolute before:inset-y-1.5 before:left-0 before:w-0.5 before:bg-highlight'
                      : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                    isLocked && 'cursor-not-allowed opacity-45 hover:bg-transparent hover:text-muted-foreground',
                  )}
                >
                  <Icon className={cn('size-4.5 shrink-0', isActive && 'text-highlight')} aria-hidden />
                  {!collapsed && (
                    <>
                      <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      <span className="min-w-0 flex-1 truncate">{label}</span>
                      {isLocked && <Lock className="size-3.5 shrink-0" aria-hidden />}
                      {value === 'review' && !isLocked && (
                        <span className="eyebrow tabular-nums text-foreground" title={reviewCopy.progressTitle}>
                          {Math.round(reviewProgress.overall * 100)}%
                        </span>
                      )}
                      {value === 'data' && documentCount > 0 && (
                        <span className="eyebrow tabular-nums text-foreground">
                          {documentCount.toLocaleString(numberLocale(locale))}
                        </span>
                      )}
                    </>
                  )}
                </button>

                {/* Subpáginas (vistas da Bibliometria, etapas da Revisão): sempre visíveis com
                    a barra aberta; no trilho recolhido, só as do módulo aberto. */}
                {!isLocked && (!collapsed || isActive) && subItems[value] && (
                  <ul
                    data-tour={value === 'bibliometrics' ? 'bibliometric-views' : `${value}-pages`}
                    aria-label={label}
                    className={cn('mt-1 flex flex-col gap-0.5', !collapsed && 'ml-[1.3rem] border-l border-border pl-2')}
                  >
                    {subItems[value].map((item) => {
                      const itemActive = isActive && item.active;
                      return (
                        <li key={item.key}>
                          <button
                            type="button"
                            onClick={item.locked ? undefined : item.onSelect}
                            onMouseEnter={item.onHover}
                            aria-current={itemActive ? 'page' : undefined}
                            aria-disabled={item.locked || undefined}
                            title={collapsed ? [item.label, item.hint].filter(Boolean).join(' — ') : item.hint}
                            className={cn(
                              'flex w-full items-center gap-2.5 rounded-md py-1.5 text-left text-[13px] transition-colors',
                              'focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring',
                              collapsed ? 'justify-center px-0' : 'px-2',
                              itemActive
                                ? 'bg-highlight/5 text-foreground'
                                : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
                              item.locked && 'cursor-not-allowed opacity-45 hover:bg-transparent hover:text-muted-foreground',
                            )}
                          >
                            <item.Icon
                              className={cn('shrink-0', collapsed ? 'size-3.5' : 'size-4', itemActive && 'text-highlight')}
                              aria-hidden
                            />
                            {!collapsed && <span className="min-w-0 flex-1 truncate">{item.label}</span>}
                            {!collapsed && item.trailing}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>

        <div className="border-t border-border p-2">
          <button
            type="button"
            onClick={toggle}
            aria-expanded={!collapsed}
            title={collapsed ? t('sidebar_expand') : undefined}
            className={cn(
              'flex w-full items-center gap-3 rounded-md py-2.5 text-sm text-muted-foreground transition-colors',
              'hover:bg-muted/60 hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring',
              collapsed ? 'justify-center px-0' : 'px-2.5',
            )}
          >
            {collapsed ? (
              <PanelLeftOpen className="size-4.5 shrink-0" aria-hidden />
            ) : (
              <PanelLeftClose className="size-4.5 shrink-0" aria-hidden />
            )}
            {!collapsed && <span>{t('sidebar_collapse')}</span>}
            {collapsed && <span className="sr-only">{t('sidebar_expand')}</span>}
          </button>
        </div>
      </nav>
    </div>
  );
}
