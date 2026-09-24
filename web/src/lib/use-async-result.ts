import { useEffect, useState } from 'react';

/**
 * Resultado assíncrono amarrado às entradas que o produziram.
 *
 * Resolve dois problemas de uma vez, ambos recorrentes nos painéis de visualização:
 *
 * 1. **Estado de carregamento sem `setState` síncrono no efeito.** Chamar
 *    `setLoading(true)` no corpo do efeito dispara uma renderização em cascata. Aqui o
 *    carregamento é DERIVADO: se o resultado guardado não pertence às entradas atuais,
 *    ainda estamos carregando.
 *
 * 2. **Resultados obsoletos.** Uma resposta que chega depois de a chave mudar é
 *    descartada; até a nova chegar, o último resultado segue exibido (`loading` avisa
 *    que ele está sendo substituído) — melhor que piscar o painel para vazio.
 *
 * @param key Identidade das entradas. Mude-a sempre que o cálculo deva refazer-se.
 * @param compute Função assíncrona, normalmente uma chamada ao worker.
 * @param options.enabled Com `false`, não calcula e mantém o último resultado — para
 *   esperar uma entrada que ainda está chegando de outro cálculo.
 */
export function useAsyncResult<T>(
  key: string,
  compute: () => Promise<T>,
  { enabled = true }: { enabled?: boolean } = {},
): { data: T | null; loading: boolean } {
  const [result, setResult] = useState<{ key: string; data: T | null; error?: unknown } | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    void (async () => {
      try {
        const data = await compute();
        if (!cancelled) setResult({ key, data });
      } catch (error) {
        if (!cancelled) setResult({ key, data: null, error });
      }
    })();

    return () => {
      cancelled = true;
    };
    // `compute` é recriada a cada render por ser um closure sobre as entradas; a chave é
    // que define quando recalcular, e incluí-la nas dependências causaria laço infinito.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  // Enquanto a nova chave calcula, o resultado anterior continua na tela: trocar um
  // filtro não pisca o gráfico para uma mensagem de carregamento e de volta.
  const fresh = result?.key === key;
  // Falha no worker: vai para o ErrorBoundary mais próximo (aviso + "Tentar novamente",
  // que remonta o painel e recalcula), em vez de deixar o painel carregando para sempre.
  if (fresh && result.error !== undefined) throw result.error;
  return { data: result?.data ?? null, loading: !fresh };
}

const identities = new WeakMap<object, number>();
let nextIdentity = 1;

/**
 * Número estável por objeto — para compor a chave de `useAsyncResult` com a base: sem
 * ele, um painel que continua montado depois da deduplicação mostraria a base antiga.
 */
export function identityKey(value: object): number {
  let id = identities.get(value);
  if (id === undefined) {
    id = nextIdentity++;
    identities.set(value, id);
  }
  return id;
}
