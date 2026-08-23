import { Database } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

/**
 * Estado vazio compartilhado pelas abas de análise.
 *
 * Todas dependem de uma base carregada, e o envio de arquivos vive na aba de Informações
 * Principais — então o texto aponta para lá em vez de repetir o painel de upload em cada
 * aba, o que multiplicaria os lugares onde o usuário pode iniciar a mesma ação.
 */
export interface EmptyStateProps {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="size-5 text-muted-foreground" aria-hidden />
          {title}
        </CardTitle>
        <CardDescription>
          {description ??
            'Carregue uma base na aba Informações Principais para liberar esta análise.'}
        </CardDescription>
      </CardHeader>
      <CardContent />
    </Card>
  );
}
