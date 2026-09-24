import { useState, type ReactNode } from 'react';
import { Maximize2 } from 'lucide-react';

import { CHART_BUTTON_CLASS } from '@/components/charts/ExportImageButton';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog';
import { useLocale } from '@/state/locale.store';

/**
 * Botão só com ícone que abre o gráfico numa janela grande. O conteúdo é uma segunda
 * instância do gráfico, montada só enquanto a janela está aberta.
 */
export function ExpandChartButton({ children }: { children: ReactNode }) {
  const t = useLocale((state) => state.t);
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t('chart_expand')}
        aria-label={t('chart_expand')}
        className={CHART_BUTTON_CLASS}
      >
        <Maximize2 className="size-4" aria-hidden />
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[96dvh] w-[96vw] max-w-[1600px] overflow-y-auto">
          <div className="pr-10">
            <DialogTitle className="eyebrow text-xs">{t('chart_expanded_title')}</DialogTitle>
            <DialogDescription className="sr-only">{t('chart_expanded_desc')}</DialogDescription>
          </div>
          <div className="min-w-0">{children}</div>
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Altura do gráfico na janela ampliada: quase toda a tela, com espaço para a barra. */
export function expandedHeight(): number {
  return Math.max(420, Math.round(window.innerHeight * 0.74));
}
