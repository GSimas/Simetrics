import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RootErrorActions } from './components/RootErrorActions';
import './fonts.css';
import './index.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Elemento #root não encontrado em index.html');
}

createRoot(container).render(
  <StrictMode>
    {/* Última rede: sem ela, um erro fora das abas desmontaria a página inteira. */}
    <ErrorBoundary variant="page" className="m-4 sm:m-8" extraAction={<RootErrorActions />}>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
