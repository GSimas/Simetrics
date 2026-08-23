import { useMemo } from 'react';
import { Sparkles } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { topQuotientsByTheme } from '@/core/locational-quotient';
import { FIELD } from '@/lib/schema';
import { useDataset } from '@/state/dataset.store';

/**
 * Categorização temática por IA e Quociente Locacional.
 *
 * O QL só ganha sentido depois que os temas existem: ele mede o quanto uma entidade
 * publica num tema acima do que seria esperado pelo peso desse tema na base. Sem temas,
 * não há denominador — e é por isso que os dois vivem no mesmo painel.
 */
export function ThemePanel() {
  const active = useDataset((state) => state.active);
  const clustering = useDataset((state) => state.clustering);
  const categorize = useDataset((state) => state.categorizeThemes);
  const busy = useDataset((state) => state.progress !== null);

  const themes = useMemo(() => {
    if (!active || !clustering) return [];

    const counts = new Map<string, number>();
    for (const doc of active) {
      const theme = String(doc[FIELD.THEME] ?? '').trim();
      if (theme) counts.set(theme, (counts.get(theme) ?? 0) + 1);
    }

    return [...counts.entries()]
      .map(([name, documents]) => ({ name, documents }))
      .sort((left, right) => right.documents - left.documents);
  }, [active, clustering]);

  const quotients = useMemo(
    () => (active && clustering ? topQuotientsByTheme(active) : null),
    [active, clustering],
  );

  if (!active) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="size-4" aria-hidden />
          Mapeamento temático por IA
        </CardTitle>
        <CardDescription>
          Os documentos são agrupados por similaridade textual no seu navegador (TF-IDF →
          LSA → K-Means, com o número de temas escolhido pelo Silhouette). Só as amostras
          de cada grupo vão ao modelo, que devolve o nome do tema.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => void categorize()} disabled={busy}>
            <Sparkles aria-hidden />
            {clustering ? 'Recategorizar temas' : 'Identificar temas'}
          </Button>

          {clustering && (
            <span className="text-sm text-muted-foreground">
              {clustering.clusterCount} temas · Silhouette{' '}
              <span className="tabular-nums">{clustering.silhouette.toFixed(3)}</span>
            </span>
          )}
        </div>

        {themes.length > 0 && (
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Tema</TableHead>
                  <TableHead>Documentos</TableHead>
                  <TableHead>Autor de maior QL</TableHead>
                  <TableHead>País de maior QL</TableHead>
                  <TableHead>Venue de maior QL</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {themes.map((theme) => (
                  <TableRow key={theme.name}>
                    <TableCell className="font-medium">{theme.name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="tabular-nums">
                        {theme.documents.toLocaleString('pt-BR')}
                      </Badge>
                    </TableCell>
                    <TableCell
                      className="max-w-56 truncate text-xs text-muted-foreground"
                      title={quotients?.authors.get(theme.name)?.label}
                    >
                      {quotients?.authors.get(theme.name)?.label ?? '—'}
                    </TableCell>
                    <TableCell
                      className="max-w-48 truncate text-xs text-muted-foreground"
                      title={quotients?.countries.get(theme.name)?.label}
                    >
                      {quotients?.countries.get(theme.name)?.label ?? '—'}
                    </TableCell>
                    <TableCell
                      className="max-w-64 truncate text-xs text-muted-foreground"
                      title={quotients?.venues.get(theme.name)?.label}
                    >
                      {quotients?.venues.get(theme.name)?.label ?? '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {clustering && (
          <p className="text-xs text-muted-foreground">
            O Quociente Locacional acima de 1 indica especialização: a entidade publica
            naquele tema mais do que a média da base. O desempate entre QLs iguais é pelo
            volume, para que uma entidade com um único documento no tema não lidere.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
