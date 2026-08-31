import { useEffect, useState } from 'react';

/**
 * Roteamento mínimo por `location.hash` — landing ⇄ workspace(+projeto). Não é um
 * router de propósito geral: só existem essas duas telas, sem rotas aninhadas, então um
 * hook pequeno evita trazer uma dependência (nenhuma está instalada hoje) para um
 * problema deste tamanho.
 *
 * O motivo de usar o hash em vez de só `useState`: recarregar a página no meio de uma
 * sessão deve voltar para o MESMO projeto, não para a landing — datasets grandes tornam
 * reabrir sem reprocessar o ganho real do recurso de Projetos. Hash também dá
 * back/forward de graça, sem `pushState` manual, e não depende de nada no servidor (o
 * `netlify.toml` já redireciona tudo para `index.html`; fragmentos de hash nem chegam
 * a sair do navegador).
 */

export type AppView = 'landing' | 'workspace';

export interface AppRoute {
  view: AppView;
  projectId: string | null;
}

const LANDING_ROUTE: AppRoute = { view: 'landing', projectId: null };

function parseHash(hash: string): AppRoute {
  const segments = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (segments[0] !== 'workspace') return LANDING_ROUTE;
  return { view: 'workspace', projectId: segments[1] ?? null };
}

function readRoute(): AppRoute {
  if (typeof window === 'undefined') return LANDING_ROUTE;
  return parseHash(window.location.hash);
}

function routeToHash(route: AppRoute): string {
  if (route.view === 'landing') return '#/';
  return route.projectId ? `#/workspace/${route.projectId}` : '#/workspace';
}

export function useHashRoute(): [AppRoute, (view: AppView, projectId?: string) => void] {
  const [route, setRoute] = useState<AppRoute>(readRoute);

  useEffect(() => {
    const handleHashChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigate = (view: AppView, projectId?: string): void => {
    window.location.hash = routeToHash({ view, projectId: projectId ?? null });
  };

  return [route, navigate];
}
