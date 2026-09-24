import { Database } from 'lucide-react';

import { SectionTitle } from '@/components/InfoTip';
import { Card, CardContent } from '@/components/ui/card';
import { useLocale } from '@/state/locale.store';

export interface EmptyStateProps {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  const t = useLocale((state) => state.t);

  return (
    <Card className="border-dashed bg-transparent">
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <div className="flex size-14 items-center justify-center border border-border">
          <Database className="size-6 text-highlight" aria-hidden />
        </div>
        <SectionTitle
          className="mt-5 justify-center"
          title={title}
          info={description ?? t('empty_generic_desc')}
        />
      </CardContent>
    </Card>
  );
}
