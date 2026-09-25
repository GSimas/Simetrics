import type { ReactElement } from 'react';
import { createElement } from 'react';

import type { Locale } from '@/lib/i18n/translations';

/**
 * Constantes e helpers compartilhados pelos painéis de visualização.
 */

/**
 * Paleta qualitativa das visualizações.
 *
 * As cores codificam categorias — agrupamentos, períodos, entidades comparadas — e não
 * intensidade, então precisam ser distinguíveis entre si e não formar um gradiente.
 * Escolhidas para permanecerem separáveis também nas formas mais comuns de daltonismo,
 * em tons médios da família Scientata legíveis tanto sobre a tinta quanto sobre o papel.
 */
export const PALETTE = [
  '#3FAE8F',
  '#E56D45',
  '#9AD63A',
  '#6A7DFF',
  '#E0B040',
  '#4CC3D9',
  '#D25F86',
  '#8F9B95',
] as const;

/**
 * Paleta das comunidades detectadas pelo Louvain — compartilhada pelo grafo de forças
 * (Sigma) e pelo diagrama de cordas, para a mesma comunidade ter a mesma cor nos dois.
 * Qualitativa: a cor codifica pertencimento a um agrupamento, não intensidade.
 */
export const COMMUNITY_COLORS = [
  '#3FAE8F', '#E56D45', '#9AD63A', '#6A7DFF', '#E0B040',
  '#4CC3D9', '#D25F86', '#8F9B95', '#A37B4F', '#B07CE8',
] as const;

export function communityColor(community: number): string {
  return COMMUNITY_COLORS[community % COMMUNITY_COLORS.length] as string;
}

/** Leitura dos quadrantes do mapa temático, por idioma. */
export const QUADRANT_NOTE: Record<Locale, string> = {
  pt:
    'As linhas tracejadas marcam as médias e formam quatro quadrantes. Alta centralidade e ' +
    'alta densidade são temas motores; baixa centralidade e alta densidade são nichos ' +
    'isolados; alta centralidade e baixa densidade são temas básicos e transversais; ' +
    'baixa em ambas são temas emergentes ou em declínio.',
  en:
    'The dashed lines mark the means and form four quadrants. High centrality and high ' +
    'density are motor themes; low centrality and high density are isolated niche themes; ' +
    'high centrality and low density are basic, transversal themes; low in both are ' +
    'emerging or declining themes.',
};

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
