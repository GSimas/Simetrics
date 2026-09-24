import { useEffect, useLayoutEffect, useRef, useState, type DependencyList } from 'react';

/**
 * Trabalho pesado e síncrono (desenhar um gráfico em canvas e codificar em PNG) fora do
 * caminho da interação.
 *
 * Antes, os gráficos do Relatório eram gerados em `useMemo` durante o render: abrir a aba
 * travava a página por mais de um segundo numa tarefa só. Aqui cada gráfico entra numa
 * fila e é gerado na sua própria tarefa — entre um e outro o navegador pinta e responde a
 * cliques.
 *
 * Retorna `undefined` enquanto pendente, `null` quando não há o que desenhar e o valor
 * quando pronto.
 */
const queue: { run: () => void; cancelled: boolean }[] = [];
let scheduled = false;

function pump(): void {
  scheduled = false;
  let job = queue.shift();
  while (job?.cancelled) job = queue.shift();
  job?.run();
  if (queue.length > 0) schedule();
}

function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  // setTimeout (e não requestIdleCallback): o trabalho precisa andar mesmo com a página
  // ocupada, só não pode ser feito todo de uma vez.
  setTimeout(pump, 0);
}

export function useIdleRender<T>(render: (() => T | null) | null, deps: DependencyList): T | null | undefined {
  // Identidade nova a cada mudança das dependências: o resultado antigo deixa de valer.
  // Padrão "ajustar estado quando a entrada muda" do React: comparar durante o render e
  // trocar o estado ali mesmo, sem efeito.
  const [current, setCurrent] = useState<{ deps: DependencyList; token: object }>(() => ({ deps, token: {} }));
  let token = current.token;
  if (!sameDeps(current.deps, deps)) {
    token = {};
    setCurrent({ deps, token });
  }
  const renderRef = useRef(render);
  // A fila roda depois do commit; o ref já aponta para a função do render mais recente.
  useLayoutEffect(() => {
    renderRef.current = render;
  });
  const [result, setResult] = useState<{ token: object; value: T | null } | null>(null);
  const enabled = render !== null;

  useEffect(() => {
    if (!enabled) return;
    const job = {
      cancelled: false,
      run: () => {
        let value: T | null;
        try {
          value = renderRef.current?.() ?? null;
        } catch {
          value = null;
        }
        if (!job.cancelled) setResult({ token, value });
      },
    };
    queue.push(job);
    schedule();
    return () => {
      job.cancelled = true;
    };
  }, [token, enabled]);

  if (!enabled) return null;
  return result?.token === token ? result.value : undefined;
}

function sameDeps(a: DependencyList, b: DependencyList): boolean {
  return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
}
