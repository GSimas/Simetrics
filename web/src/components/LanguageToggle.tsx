import { useLocale } from '@/state/locale.store';

export function LanguageToggle() {
  const { locale, setLocale } = useLocale();

  return (
    <div className="flex items-center rounded-xl border border-border/80 bg-card/80 p-0.5 shadow-2xs">
      <button
        type="button"
        onClick={() => setLocale('pt')}
        className={`flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-bold transition-all ${
          locale === 'pt'
            ? 'bg-primary text-primary-foreground shadow-2xs'
            : 'text-muted-foreground hover:text-foreground'
        }`}
        title="Português"
        aria-label="Português"
      >
        <span>PT</span>
      </button>
      <button
        type="button"
        onClick={() => setLocale('en')}
        className={`flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-bold transition-all ${
          locale === 'en'
            ? 'bg-primary text-primary-foreground shadow-2xs'
            : 'text-muted-foreground hover:text-foreground'
        }`}
        title="English"
        aria-label="English"
      >
        <span>EN</span>
      </button>
    </div>
  );
}
