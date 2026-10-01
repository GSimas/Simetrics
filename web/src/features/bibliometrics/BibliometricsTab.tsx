import { Suspense } from 'react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { PANEL_ENTER } from '@/features/review/motion';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { BIBLIOMETRIC_VIEWS } from './views';

/**
 * Aba Análise Bibliométrica: Informações Principais, Redes, Análises Avançadas e Relatório, na ordem natural
 * de uma análise. A troca de vista fica na barra lateral (`WorkspaceSidebar`).
 */
export default function BibliometricsTab() {
  const t = useLocale((state) => state.t);
  const view = useNavigation((state) => state.bibliometricView);
  const current = BIBLIOMETRIC_VIEWS.find((item) => item.value === view) ?? BIBLIOMETRIC_VIEWS[0];
  const { Panel } = current;

  return (
    <div className="space-y-5">
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
