import type { KeyboardEvent } from 'react';

/**
 * Enter num item preenchido de uma lista cria o próximo — o campo novo já nasce com o
 * foco, então dá para escrever a lista inteira sem tirar as mãos do teclado. Num item
 * vazio não faz nada, para não empilhar linhas em branco.
 */
export function addOnEnter(add: () => void) {
  return (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    if (event.currentTarget.value.trim()) add();
  };
}
