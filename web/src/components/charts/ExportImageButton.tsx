import { useState } from 'react';
import { FileCode2, ImageDown, ImageIcon, Layers } from 'lucide-react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { exportChartImage, type ChartImage, type ImageFormat } from '@/lib/export-image';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';

export interface ExportImageButtonProps {
  /** Produz o SVG do gráfico no momento do clique (pode ser assíncrono). */
  getImage: () => ChartImage | null | Promise<ChartImage | null>;
  filename: string;
  className?: string;
}

const FORMATS: { format: ImageFormat; Icon: typeof ImageIcon; labelKey: 'export_svg' | 'export_jpg' | 'export_png'; hintKey: 'export_svg_hint' | 'export_jpg_hint' | 'export_png_hint' }[] = [
  { format: 'svg', Icon: FileCode2, labelKey: 'export_svg', hintKey: 'export_svg_hint' },
  { format: 'jpg', Icon: ImageIcon, labelKey: 'export_jpg', hintKey: 'export_jpg_hint' },
  { format: 'png', Icon: Layers, labelKey: 'export_png', hintKey: 'export_png_hint' },
];

/** Botão quadrado só com ícone da barra dos gráficos (exportar, ampliar, zoom). */
export const CHART_BUTTON_CLASS =
  'inline-flex size-8 shrink-0 cursor-pointer items-center justify-center border border-border bg-background/80 text-muted-foreground transition-colors hover:border-highlight hover:text-highlight focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-default disabled:opacity-40 disabled:hover:border-border disabled:hover:text-muted-foreground';

/**
 * Botão só com ícone que abre as opções de exportação do gráfico — o mesmo em todos os
 * gráficos do Simetrics.
 */
export function ExportImageButton({ getImage, filename, className }: ExportImageButtonProps) {
  const t = useLocale((state) => state.t);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<ImageFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (format: ImageFormat): Promise<void> => {
    setBusy(format);
    setError(null);
    try {
      const image = await getImage();
      if (!image) throw new Error(t('export_unavailable'));
      await exportChartImage(image, format, filename);
      setOpen(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t('export_image')}
        aria-label={t('export_image')}
        className={cn(CHART_BUTTON_CLASS, className)}
      >
        <ImageDown className="size-4" aria-hidden />
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>{t('export_image')}</DialogTitle>
            <DialogDescription>{t('export_desc')}</DialogDescription>
          </DialogHeader>

          <div className="grid gap-2">
            {FORMATS.map(({ format, Icon, labelKey, hintKey }) => (
              <button
                key={format}
                type="button"
                disabled={busy !== null}
                onClick={() => void run(format)}
                className="group flex cursor-pointer items-center gap-3 border border-border p-3 text-left transition-colors hover:border-highlight disabled:cursor-wait disabled:opacity-60"
              >
                <Icon className="size-5 shrink-0 text-highlight" aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{t(labelKey)}</span>
                  <span className="block text-xs text-muted-foreground">{t(hintKey)}</span>
                </span>
                <span className="eyebrow transition-colors group-hover:text-highlight">
                  {busy === format ? '…' : format.toUpperCase()}
                </span>
              </button>
            ))}
          </div>

          {error && <p className="text-sm text-destructive animate-in fade-in-0">{error}</p>}
        </DialogContent>
      </Dialog>
    </>
  );
}
