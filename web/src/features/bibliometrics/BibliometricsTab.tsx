import { Suspense } from 'react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { PANEL_ENTER, PRESS } from '@/features/review/motion';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { BIBLIOMETRIC_VIEWS } from './views';

/**
 * Aba Análise Bibliométrica: Informações Principais, Redes e Relatório, na ordem natural
 * de uma análise, sob uma navegação própria — o mesmo desenho das etapas da revisão.
 */
export default function BibliometricsTab() {
  const t = useLocale((state) => state.t);
  const view = useNavigation((state) => state.bibliometricView);
  const setActiveTab = useNavigation((state) => state.setActiveTab);
  const current = BIBLIOMETRIC_VIEWS.find((item) => item.value === view) ?? BIBLIOMETRIC_VIEWS[0];
  const { Panel } = current;

  return (
    <div className="space-y-5">
      <nav className="flex flex-wrap gap-2" aria-label={t('tab_bibliometrics')} data-tour="bibliometric-views">
        {BIBLIOMETRIC_VIEWS.map(({ value, labelKey, Icon }, index) => {
          const active = view === value;
          return (
            <button
              key={value}
              type="button"
              aria-current={active ? 'page' : undefined}
              onMouseEnter={() => void BIBLIOMETRIC_VIEWS[index]?.Panel.preload()}
              onClick={() => setActiveTab(value)}
              className={cn(
                'group flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium sm:text-sm',
                PRESS,
                active
                  ? 'border-highlight bg-highlight/5 text-foreground shadow-[0_0_24px_-10px_var(--highlight)]'
                  : 'border-border text-muted-foreground hover:border-highlight/50 hover:text-foreground',
              )}
            >
              <span className={cn('font-mono text-[10px] transition-colors duration-200', active && 'text-highlight')}>
                {index + 1}
              </span>
              <Icon
                className={cn(
                  'size-4 transition-[color,transform] duration-200 group-hover:scale-110 motion-reduce:transform-none',
                  active && 'text-highlight',
                )}
                aria-hidden
              />
              {t(labelKey)}
            </button>
          );
        })}
      </nav>

      {/* A chave nova a cada vista refaz a entrada: a troca desliza em vez de piscar. */}
      <section key={current.value} className={PANEL_ENTER}>
        <h3 className="sr-only">{t(current.labelKey)}</h3>
        {/* Uma vista que quebre (ou cujo chunk não baixe) não leva as outras junto. */}
        <ErrorBoundary variant="page" label={t(current.labelKey)}>
          <Suspense fallback={<ViewFallback />}>
            <Panel />
          </Suspense>
        </ErrorBoundary>
      </section>
    </div>
  );
}

/** Espaço da vista enquanto o chunk chega — ocupa a altura para o rodapé não pular. */
function ViewFallback() {
  const t = useLocale((state) => state.t);
  return (
    // data-tab-fallback: o rodapé fica invisível enquanto isto existe (ver index.css).
    <div className="min-h-[60vh]" aria-busy="true" data-tab-fallback>
      <span className="sr-only">{t('loading')}</span>
    </div>
  );
}
