import type { ReactElement } from 'react';
import { createElement } from 'react';

/**
 * Constantes e helpers compartilhados pelos painéis de visualização.
 */

/**
 * Paleta qualitativa das visualizações.
 *
 * As cores codificam categorias — agrupamentos, períodos, entidades comparadas — e não
 * intensidade, então precisam ser distinguíveis entre si e não formar um gradiente.
 * Escolhidas para permanecerem separáveis também nas formas mais comuns de daltonismo.
 */
export const PALETTE = [
  '#1273B9',
  '#E8734A',
  '#3FA96C',
  '#A05FC4',
  '#D8A13A',
  '#4BAFC9',
  '#D45D79',
  '#7A8B99',
] as const;

/** Leitura dos quadrantes do mapa temático. */
export const QUADRANT_NOTE =
  'As linhas tracejadas marcam as médias e formam quatro quadrantes. Alta centralidade e ' +
  'alta densidade são temas motores; baixa centralidade e alta densidade são nichos ' +
  'isolados; alta centralidade e baixa densidade são temas básicos e transversais; ' +
  'baixa em ambas são temas emergentes ou em declínio.';

/** Mensagem centralizada, no lugar de um gráfico que não pôde ser desenhado. */
export function chartMessage(text: string): ReactElement {
  return createElement(
    'p',
    {
      className:
        'grid min-h-40 place-items-center rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground',
    },
    text,
  );
}
