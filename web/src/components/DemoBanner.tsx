import { useState } from 'react';
import { Copy, Eye, Loader2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useProjectStore } from '@/state/project.store';

/** Transforma o exemplo aberto num projeto salvo e editável. */
export function DemoCopyButton({ size = 'sm' }: { size?: 'sm' | 'default' }) {
  const t = useLocale((state) => state.t);
  const saveDemoCopy = useProjectStore((state) => state.saveDemoCopy);
  const [saving, setSaving] = useState(false);

  return (
    <Button
      type="button"
      size={size}
      disabled={saving}
      onClick={() => {
        setSaving(true);
        void saveDemoCopy().finally(() => setSaving(false));
      }}
    >
      {saving ? <Loader2 className="animate-spin" aria-hidden /> : <Copy aria-hidden />}
      {t('demo_copy_btn')}
    </Button>
  );
}

/** Aviso do modo exemplo, no topo do workspace enquanto o exemplo estiver aberto. */
export function DemoBanner() {
  const t = useLocale((state) => state.t);
  const isDemo = useDataset((state) => state.isDemo);
  if (!isDemo) return null;

  return (
    <div role="status" className="border-b border-highlight/40 bg-highlight/10">
      <div className="container flex flex-wrap items-center gap-x-4 gap-y-2 py-2.5">
        <Eye className="size-4 shrink-0 text-highlight" aria-hidden />
        <p className="min-w-[14rem] flex-1 text-xs leading-relaxed sm:text-sm">
          <span className="font-semibold">{t('demo_banner_title')}.</span>{' '}
          <span className="text-muted-foreground">{t('demo_banner_desc')}</span>
        </p>
        <DemoCopyButton />
      </div>
    </div>
  );
}
