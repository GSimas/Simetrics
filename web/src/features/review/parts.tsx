import type { ReactNode } from 'react';

/** Bloco com título e dica — a unidade visual dos painéis da revisão. */
export function Block({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="space-y-3 rounded-xl border border-border/80 p-4">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">{title}</h3>
        {hint && <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>}
      </div>
      {children}
    </section>
  );
}
