import { create } from 'zustand';

import { optionsForType, type SearchOptions } from '@/core/search';
import type { SearchEntityType } from '@/lib/types';
import { useDataset } from './dataset.store';

/**
 * Navegação do workspace: aba ativa e a entidade aberta no Motor de Busca.
 *
 * Fica fora dos componentes porque qualquer gráfico pode mandar o usuário ao perfil de
 * um termo — clicar numa barra, num nó, numa palavra da nuvem. E, por ser global, a
 * busca sobrevive à troca de abas.
 */
interface NavigationState {
  activeTab: string;
  searchType: SearchEntityType;
  searchTerm: string | null;
  setActiveTab: (tab: string) => void;
  /**
   * Abre no Motor de Busca o perfil do termo, se ele existir no acervo sob algum dos
   * tipos dados (na ordem de preferência). Devolve `false` e não navega quando não há
   * correspondência — um rótulo de gráfico nem sempre é uma entidade da base.
   */
  openInSearch: (term: unknown, types: readonly SearchEntityType[]) => boolean;
}

export const useNavigation = create<NavigationState>()((set) => ({
  activeTab: 'overview',
  searchType: 'Autor',
  searchTerm: null,
  setActiveTab: (activeTab) => set({ activeTab }),
  openInSearch(term, types) {
    const match = resolveEntity(term, types);
    if (!match) return false;
    set({ activeTab: 'search', searchType: match.type, searchTerm: match.term });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return true;
  },
}));

// Índice minúsculo → grafia original, por tipo, construído uma vez por conjunto de opções.
const indexCache = new WeakMap<SearchOptions, Map<SearchEntityType, Map<string, string>>>();

function lookup(options: SearchOptions, type: SearchEntityType): Map<string, string> {
  let byType = indexCache.get(options);
  if (!byType) {
    byType = new Map();
    indexCache.set(options, byType);
  }
  let index = byType.get(type);
  if (!index) {
    index = new Map(optionsForType(options, type).map((value) => [value.toLowerCase(), value]));
    byType.set(type, index);
  }
  return index;
}

/** Encontra o termo no acervo carregado, ignorando maiúsculas e espaços nas pontas. */
export function resolveEntity(
  term: unknown,
  types: readonly SearchEntityType[],
): { type: SearchEntityType; term: string } | null {
  const options = useDataset.getState().searchOptions;
  if (!options || typeof term !== 'string') return null;
  const needle = term.trim().toLowerCase();
  if (!needle) return null;

  for (const type of types) {
    const original = lookup(options, type).get(needle);
    if (original) return { type, term: original };
  }
  return null;
}

// Outra base carregada: o termo aberto no Motor de Busca pertencia à anterior.
useDataset.subscribe(
  (state) => state.original,
  () => useNavigation.setState({ searchTerm: null }),
);

/** Atalho para handlers de clique em gráficos, fora do ciclo de render. */
export function openInSearch(term: unknown, types: readonly SearchEntityType[]): boolean {
  return useNavigation.getState().openInSearch(term, types);
}
