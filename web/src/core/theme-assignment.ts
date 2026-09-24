import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import type { ClusterSample } from './clustering';

/**
 * Nomes de tema e sua aplicação à base. Separado de `clustering.ts` porque o store usa
 * estas funções na thread principal, e `clustering.ts` importa o k-means (ml-kmeans + FFT):
 * juntos, o chunk inicial carregaria a biblioteca de clusterização, que só roda no worker.
 */
export function fallbackThemeName(cluster: ClusterSample): string {
  if (cluster.topTerms.length === 0) return `Tema ${cluster.clusterId + 1}`;
  // Sem o modelo, os próprios termos característicos já descrevem o grupo.
  return cluster.topTerms
    .slice(0, 3)
    .map((term) => term.charAt(0).toUpperCase() + term.slice(1))
    .join(', ');
}

export function applyThemes(
  rows: Dataset,
  assignments: readonly number[],
  names: ReadonlyMap<number, string>,
): Dataset {
  return rows.map((doc, index) => ({
    ...doc,
    [FIELD.THEME]: names.get(assignments[index] ?? -1) ?? 'Outros/Não Categorizado',
  }));
}
