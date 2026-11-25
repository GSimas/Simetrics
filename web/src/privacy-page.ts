/**
 * Página de privacidade (privacidade.html): só a tipografia e os tokens do app, sem React.
 * Mostra o idioma da interface (o salvo no app ou o do sistema) e troca pelos botões.
 */
import './fonts.css';
import './index.css';

type Lang = 'pt' | 'en';

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem('simetrics-locale');
    if (saved === 'pt' || saved === 'en') return saved;
  } catch {
    // Sem armazenamento: segue o sistema.
  }
  for (const lang of navigator.languages?.length ? navigator.languages : [navigator.language]) {
    const code = lang?.toLowerCase().slice(0, 2);
    if (code === 'pt' || code === 'en') return code;
  }
  return 'en';
}

function show(lang: Lang): void {
  const root = document.documentElement;
  root.dataset.lang = lang;
  root.lang = lang === 'en' ? 'en' : 'pt-BR';
  document.title = lang === 'en' ? 'Privacy · Simetrics' : 'Privacidade · Simetrics';
}

show(initialLang());
document.querySelectorAll<HTMLButtonElement>('[data-lang-switch]').forEach((button) => {
  button.addEventListener('click', () => show(button.dataset.langSwitch as Lang));
});
