import { FolderOpen } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useLocale } from '@/state/locale.store';

/**
 * Saída do erro de raiz. O projeto aberto vive na URL (`#/workspace/<id>`) e é reaberto a
 * cada recarregamento — se foi ele que quebrou o app, recarregar repetiria o erro. Voltar
 * à lista de projetos limpa a rota antes.
 */
export function RootErrorActions() {
  const t = useLocale((state) => state.t);
  return (
    <Button
      size="sm"
      variant="ghost"
      className="gap-1.5"
      onClick={() => {
        window.location.hash = '#/';
        window.location.reload();
      }}
    >
      <FolderOpen className="size-3.5" aria-hidden />
      {t('error_back_projects')}
    </Button>
  );
}
