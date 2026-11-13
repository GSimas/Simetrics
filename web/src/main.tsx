import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RootErrorActions } from './components/RootErrorActions';
import { TooltipLayer } from './components/TooltipLayer';
import { installOverlayScrollbars } from './lib/overlay-scrollbars';
import { completeOpenRouterLogin } from './lib/openrouter-oauth';
import './fonts.css';
import './index.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Elemento #root não encontrado em index.html');
}

void completeOpenRouterLogin();
installOverlayScrollbars();

createRoot(container).render(
  <StrictMode>
    {/* Última rede: sem ela, um erro fora das abas desmontaria a página inteira. */}
    <ErrorBoundary variant="page" className="m-4 sm:m-8" extraAction={<RootErrorActions />}>
      <App />
    </ErrorBoundary>
    {/* Dicas de texto no visual do app, para todo `title` da página. */}
    <TooltipLayer />
  </StrictMode>,
);
