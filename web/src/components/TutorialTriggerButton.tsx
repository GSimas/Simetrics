import { HelpCircle } from 'lucide-react';

import { ICON_BUTTON } from '@/components/HeaderActions';
import { useLocale } from '@/state/locale.store';

/**
 * Botão do cabeçalho que abre o tutorial. Fica fora de `TutorialModal.tsx` para o modal
 * (e suas prévias) sair do chunk inicial: ele só é baixado quando alguém pede o tutorial.
 */
export function TutorialTriggerButton({ onClick }: { onClick: () => void }) {
  const t = useLocale((state) => state.t);

  return (
    <button
      type="button"
      data-tour="tutorial"
      onClick={onClick}
      // Só o ícone, no traço dos botões redondos do cabeçalho; o nome fica no rótulo.
      aria-label={t('tutorial_btn')}
      title={t('tutorial_btn')}
      className={ICON_BUTTON}
    >
      <HelpCircle className="size-4" aria-hidden />
    </button>
  );
}
