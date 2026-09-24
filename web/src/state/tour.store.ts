import { create } from 'zustand';

/**
 * Estado do tour guiado. Global porque quem inicia o tour (o modal do tutorial, que
 * também vive na landing) não é quem o desenha (o workspace, em `App`).
 */
interface TourState {
  active: boolean;
  index: number;
  /** Sentido do último passo — um passo sem alvo na tela é pulado nesse sentido. */
  direction: 1 | -1;
  start: () => void;
  stop: () => void;
  goTo: (index: number) => void;
}

export const useTour = create<TourState>()((set, get) => ({
  active: false,
  index: 0,
  direction: 1,
  start: () => set({ active: true, index: 0, direction: 1 }),
  stop: () => set({ active: false }),
  goTo: (index) => set({ index, direction: index >= get().index ? 1 : -1 }),
}));
