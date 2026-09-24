/**
 * Layout do diagrama de cordas (grafo radial): todos os nós num único círculo,
 * agrupados por comunidade e, dentro dela, do maior para o menor peso. Nós do mesmo grupo
 * ocupam um arco contínuo, e as cordas entre grupos cruzam o centro — é o que torna
 * visíveis as pontes entre comunidades.
 *
 * O primeiro nó fica no topo (−90°) e a ordem segue no sentido horário. Coordenadas no
 * círculo unitário; `angle` em radianos, para girar rótulos.
 */
export interface ChordItem<K> {
  key: K;
  /** Importância: documentos, grau, tamanho do nó. */
  weight: number;
  group?: number;
}

export interface ChordPosition<K> {
  key: K;
  angle: number;
  x: number;
  y: number;
}

export function chordLayout<K>(items: readonly ChordItem<K>[]): ChordPosition<K>[] {
  const ordered = [...items].sort(
    (left, right) => (left.group ?? 0) - (right.group ?? 0) || right.weight - left.weight,
  );

  return ordered.map((item, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(ordered.length, 1);
    return { key: item.key, angle, x: Math.cos(angle), y: Math.sin(angle) };
  });
}
