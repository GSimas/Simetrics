import type { NextAction, NextActionKind } from '@/core/review/progress';
import { useNavigation } from '@/state/navigation.store';
import { prefersReducedMotion } from '@/state/preferences.store';
import { useReviewNav } from '@/state/review.store';

/** Onde cada próxima ação acontece na tela. */
const TARGETS: Record<NextActionKind, string> = {
  title: '[data-review-target="title"]',
  objective: '[data-review-target="objective"]',
  framework: '[data-review-target="framework"]',
  question: '[data-review-target="question"]',
  concept: '[data-review-target="concept"]',
  inclusion: '[data-review-target="inclusion"]',
  exclusion: '[data-review-target="exclusion"]',
  import: '[data-tour="upload"]',
  'screen-ta': '[data-review-target="screen-ta"]',
  'screen-ft': '[data-review-target="screen-ft"]',
  quality: '[data-tour="review-step-quality"]',
  'extraction-form': '[data-review-target="extraction-form"]',
  extraction: '[data-tour="review-step-extraction"]',
  report: '[data-tour="review-export"]',
};

/** Campo a receber o foco: o primeiro vazio do alvo; sem campos, o botão de adicionar. */
function fieldIn(target: HTMLElement): HTMLElement | null {
  if (target.matches('input, textarea, [role="tab"]')) return target;
  const fields = [...target.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>('input:not([type="hidden"]), textarea')];
  return fields.find((field) => !field.value.trim()) ?? fields[0] ?? target.querySelector('button');
}

/**
 * Rola até o alvo assim que ele existir na tela e o faz piscar em destaque; com `focus`,
 * põe o cursor no campo a preencher. A etapa troca de painel e as telas chegam em chunks,
 * então procura o alvo por até 2 s.
 */
function highlightTarget(selector: string, focus: boolean): void {
  const reduced = prefersReducedMotion();
  const deadline = performance.now() + 2000;
  const seek = (): void => {
    const target = document.querySelector<HTMLElement>(selector);
    if (!target) {
      // setTimeout, e não requestAnimationFrame: este pausa com a aba em segundo plano.
      if (performance.now() < deadline) setTimeout(seek, 50);
      return;
    }
    // Aba da triagem (título/resumo × texto completo): abre a certa.
    if (target.getAttribute('role') === 'tab' && target.getAttribute('aria-selected') === 'false') target.click();
    // Alvo alto (um bloco inteiro) rola pelo topo, para o começo dele não ficar fora da tela.
    const tall = target.offsetHeight > window.innerHeight * 0.6;
    target.scrollIntoView({ block: tall ? 'start' : 'center', behavior: reduced ? 'auto' : 'smooth' });
    target.classList.remove('target-flash');
    void target.offsetWidth; // reinicia a animação num segundo clique
    target.classList.add('target-flash');
    setTimeout(() => target.classList.remove('target-flash'), 2800);
    if (focus) fieldIn(target)?.focus({ preventScroll: true });
  };
  setTimeout(seek, 0);
}

/** Abre o protocolo direto num bloco (`data-review-target`), piscando e com o foco nele. */
export function goToProtocolBlock(target: string): void {
  useReviewNav.getState().setStep('protocol');
  highlightTarget(`[data-review-target="${target}"]`, true);
}

/**
 * Leva à próxima ação: abre a tela e a etapa certas, rola até o alvo e o faz piscar em
 * destaque. No protocolo e na triagem, o foco já vai para o campo a preencher.
 */
export function goToNextAction(action: NextAction): void {
  const navigation = useNavigation.getState();
  if (action.kind === 'import') {
    navigation.setActiveTab('data');
  } else {
    useReviewNav.getState().setStep(action.step);
    navigation.setActiveTab('review');
  }
  highlightTarget(TARGETS[action.kind], action.step === 'protocol' || action.kind.startsWith('screen'));
}
