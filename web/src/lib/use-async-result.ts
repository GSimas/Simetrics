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
 * 2. **Resultados obsoletos.** Trocar de filtro antes de a resposta anterior chegar
 *    mostraria dados que não correspondem mais aos controles na tela. Comparar a chave
 *    descarta o que ficou para trás.
 *
 * @param key Identidade das entradas. Mude-a sempre que o cálculo deva refazer-se.
 * @param compute Função assíncrona, normalmente uma chamada ao worker.
 */
export function useAsyncResult<T>(
  key: string,
  compute: () => Promise<T>,
): { data: T | null; loading: boolean } {
  const [result, setResult] = useState<{ key: string; data: T } | null>(null);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      const data = await compute();
      if (!cancelled) setResult({ key, data });
    })();

    return () => {
      cancelled = true;
    };
    // `compute` é recriada a cada render por ser um closure sobre as entradas; a chave é
    // que define quando recalcular, e incluí-la nas dependências causaria laço infinito.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const fresh = result?.key === key;
  return { data: fresh ? result.data : null, loading: !fresh };
}
