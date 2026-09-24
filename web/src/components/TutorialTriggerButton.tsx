import { HelpCircle } from 'lucide-react';

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
      // No celular o texto some; o nome acessível não pode sumir junto.
      aria-label={t('tutorial_btn')}
      className="header-chip cursor-pointer"
    >
      <HelpCircle className="size-3.5" aria-hidden />
      <span className="hidden sm:inline">{t('tutorial_btn')}</span>
    </button>
  );
}
