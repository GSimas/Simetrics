import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useLocale } from '@/state/locale.store';

export function ThemeToggle() {
  const t = useLocale((state) => state.t);

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'light';
    const saved = localStorage.getItem('simetrics-theme');
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }

    // Escuta mudanças de tema do sistema operacional se o usuário não definiu manualmente
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = (e: MediaQueryListEvent) => {
      const manual = localStorage.getItem('simetrics-theme-manual');
      if (!manual) {
        setTheme(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleSystemChange);
    return () => mediaQuery.removeEventListener('change', handleSystemChange);
  }, [theme]);

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('simetrics-theme', next);
    localStorage.setItem('simetrics-theme-manual', 'true');
  };

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={toggleTheme}
      className="size-9 rounded-xl border-border/80 bg-card/80 text-foreground shadow-2xs transition-all hover:bg-muted"
      title={theme === 'dark' ? t('theme_dark') : t('theme_light')}
      aria-label={theme === 'dark' ? t('theme_dark') : t('theme_light')}
    >
      {theme === 'dark' ? (
        <Sun className="size-4.5 text-amber-400 transition-transform duration-300 hover:rotate-45" />
      ) : (
        <Moon className="size-4.5 text-slate-700 transition-transform duration-300 hover:-rotate-12" />
      )}
    </Button>
  );
}
