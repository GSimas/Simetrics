/** Byte Order Mark (U+FEFF), que o Scopus e a Cochrane prefixam aos CSVs exportados. */
export const BOM = '\uFEFF';

/**
 * Remove o BOM inicial de um texto.
 *
 * Sem isso, o cabeçalho da primeira coluna do CSV vira `\uFEFF + "Title"` em vez de
 * `"Title"`, e o mapeamento de colunas falha em silêncio justamente no campo do título.
 *
 * O caractere é escrito como escape, e nao literal, porque literal ele é invisível no
 * editor e no diff — e o ESLint o rejeita como espaço em branco irregular.
 */
export function stripBom(text: string): string {
  return text.startsWith(BOM) ? text.slice(BOM.length) : text;
}
