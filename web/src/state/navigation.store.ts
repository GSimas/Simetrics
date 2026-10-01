import { create } from 'zustand';

import { optionsForType, type SearchOptions } from '@/core/search';
import type { SearchEntityType } from '@/lib/types';
import { useDataset } from './dataset.store';

/**
 * Navegação do workspace: aba ativa, a vista aberta na Análise Bibliométrica e a entidade
 * aberta no Motor de Busca.
 *
 * Fica fora dos componentes porque qualquer gráfico pode mandar o usuário ao perfil de
 * um termo — clicar numa barra, num nó, numa palavra da nuvem. E, por ser global, a
 * busca sobrevive à troca de abas.
 */
/** As telas do workspace: Dados primeiro; os módulos só com uma base carregada. */
export type WorkspaceTab = 'data' | 'bibliometrics' | 'search' | 'review';
/** As vistas dentro da aba Análise Bibliométrica. */
export type BibliometricView = 'overview' | 'networks' | 'advanced' | 'report';

const BIBLIOMETRIC_VIEWS: readonly string[] = ['overview', 'networks', 'advanced', 'report'] satisfies BibliometricView[];

interface NavigationState {
  activeTab: WorkspaceTab;
  bibliometricView: BibliometricView;
  /** Tipo do termo aberto no dossiê. */
  searchType: SearchEntityType;
  searchTerm: string | null;
  /** Consulta enviada na caixa do Motor de Busca (resultados textuais). */
  searchQuery: string;
  /** Motor de Busca no modo conversa com a Simi, em vez de resultados e dossiê. */
  chatOpen: boolean;
  /**
   * Abre uma aba — ou, com o nome de uma vista bibliométrica (`overview`, `networks`,
   * `advanced`, `report`), a aba Análise Bibliométrica já nessa vista. Sem base
   * carregada, os módulos ficam bloqueados: só Dados e a Revisão (pelo protocolo) abrem.
   */
  setActiveTab: (tab: string) => void;
  /** Abre o dossiê de um termo. */
  selectEntity: (type: SearchEntityType, term: string) => void;
  /**
   * Abre no Motor de Busca o perfil do termo, se ele existir no acervo sob algum dos
   * tipos dados (na ordem de preferência). Devolve `false` e não navega quando não há
   * correspondência — um rótulo de gráfico nem sempre é uma entidade da base.
   */
  openInSearch: (term: unknown, types: readonly SearchEntityType[]) => boolean;
}

export const useNavigation = create<NavigationState>()((set, get) => ({
  activeTab: 'data',
  bibliometricView: 'overview',
  searchType: 'Autor',
  searchTerm: null,
  searchQuery: '',
  chatOpen: false,
  setActiveTab: (tab) => {
    // A revisão abre sem base: o protocolo se escreve antes da busca.
    if (tab !== 'data' && tab !== 'review' && !useDataset.getState().active) return;
    set(
      BIBLIOMETRIC_VIEWS.includes(tab)
        ? { activeTab: 'bibliometrics', bibliometricView: tab as BibliometricView }
        : { activeTab: tab as WorkspaceTab },
    );
  },
  selectEntity: (type, term) => set({ searchType: type, searchTerm: term, chatOpen: false }),
  openInSearch(term, types) {
    const match = resolveEntity(term, types);
    if (!match) return false;
    get().selectEntity(match.type, match.term);
    set({ activeTab: 'search' });
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
  () => useNavigation.setState({ searchTerm: null, searchQuery: '', chatOpen: false }),
);

// Base descarregada (limpar, trocar de projeto): os módulos voltam a bloquear.
useDataset.subscribe(
  (state) => state.active,
  (active) => {
    if (!active) useNavigation.setState({ activeTab: 'data' });
  },
);

/** Atalho para handlers de clique em gráficos, fora do ciclo de render. */
export function openInSearch(term: unknown, types: readonly SearchEntityType[]): boolean {
  return useNavigation.getState().openInSearch(term, types);
}

/** Abre o dossiê de uma entidade no Motor de Busca, de qualquer tela (sai do modo conversa). */
export function openDossier(type: SearchEntityType, term: string): void {
  const navigation = useNavigation.getState();
  navigation.selectEntity(type, term);
  navigation.setActiveTab('search');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
