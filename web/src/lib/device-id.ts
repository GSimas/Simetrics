/**
 * Identificador aleatório deste navegador, usado só para contar a cota gratuita (perguntas
 * do Simi, classificações híbridas). Não carrega informação nenhuma sobre o usuário, e o
 * servidor só o grava como hash.
 */

const STORAGE_KEY = 'simetrics_device_id';
let memoryId: string | null = null;

export function getDeviceId(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && /^[A-Za-z0-9-]{8,64}$/.test(saved)) return saved;
    const created = crypto.randomUUID();
    localStorage.setItem(STORAGE_KEY, created);
    return created;
  } catch {
    // Armazenamento bloqueado: vale só para esta aba.
    memoryId ??= crypto.randomUUID();
    return memoryId;
  }
}

/** Cabeçalhos que identificam o dispositivo, a unidade de cota e o idioma nas rotas /api. */
export function freeTierHeaders(unit: string | null, locale: 'pt' | 'en'): Record<string, string> {
  return {
    'X-Simetrics-Device': getDeviceId(),
    'X-Simetrics-Locale': locale,
    ...(unit ? { 'X-Simetrics-Unit': unit } : {}),
  };
}
