/**
 * Imagem de gráfico da prévia do relatório.
 *
 * `image`: `undefined` enquanto o gráfico é gerado (fila de `useIdleRender`), `null`
 * quando não há dados, ou o PNG. Pendente, o bloco já ocupa a altura final (mesma
 * proporção do canvas), para a prévia não pular quando a imagem chega.
 */
export function ReportChartImage({
  image,
  width,
  height,
  alt,
}: {
  image: string | null | undefined;
  width: number;
  height: number;
  alt: string;
}) {
  if (image === null) return null;
  return (
    <div className="rounded-xl border border-border/80 overflow-hidden shadow-xs">
      {image === undefined ? (
        <div style={{ aspectRatio: `${width} / ${height}` }} aria-busy="true" role="img" aria-label={alt} />
      ) : (
        <img src={image} alt={alt} width={width} height={height} className="w-full h-auto object-contain" />
      )}
    </div>
  );
}
