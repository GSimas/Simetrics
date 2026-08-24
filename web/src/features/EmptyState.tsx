import { Database } from 'lucide-react';

import { Card, CardContent } from '@/components/ui/card';
import { useLocale } from '@/state/locale.store';

export interface EmptyStateProps {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  const t = useLocale((state) => state.t);

  return (
    <Card className="border-border/80 shadow-xs">
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <div className="flex size-14 items-center justify-center rounded-2xl border border-blue-200/80 bg-gradient-to-br from-blue-100 to-indigo-100 shadow-2xs dark:border-blue-900/60 dark:from-blue-950 dark:to-indigo-950">
          <Database className="size-7 text-blue-600 dark:text-blue-400" aria-hidden />
        </div>
        <h3 className="mt-4 text-base font-bold text-foreground">{title}</h3>
        <p className="mt-1.5 max-w-md text-xs sm:text-sm text-muted-foreground">
          {description ?? t('empty_generic_desc')}
        </p>
      </CardContent>
    </Card>
  );
}
