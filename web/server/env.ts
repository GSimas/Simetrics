/**
 * Configuração do lado servidor, lida de variáveis de ambiente.
 *
 * Em desenvolvimento, o Vite lê do `web/.env` (vite.config.ts); em produção, do painel da
 * Netlify. Nada daqui chega ao bundle do cliente: nenhuma variável tem o prefixo `VITE_`.
 */

export interface ServerEnv {
  deepseekKey: string;
  deepseekModel: string;
  deepseekBaseUrl: string;
  typesafeKey: string;
  /** Perguntas grátis do Simi por dispositivo. */
  simiFreeQuestions: number;
  /** Classificações híbridas grátis (descoberta pelo DeepSeek) por dispositivo. */
  hybridFreeRuns: number;
  /** Tetos diários por IP — contêm quem apaga o armazenamento local para zerar a cota. */
  simiIpDaily: number;
  hybridIpDaily: number;
  labelIpDaily: number;
  /** Sal dos hashes de dispositivo e IP: nenhum identificador é gravado em claro. */
  quotaSalt: string;
}

type EnvSource = Record<string, string | undefined>;

function integer(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

export function readServerEnv(source: EnvSource): ServerEnv {
  return {
    deepseekKey: source['DEEPSEEK_API_KEY']?.trim() ?? '',
    deepseekModel: source['DEEPSEEK_MODEL']?.trim() || 'deepseek-flash',
    deepseekBaseUrl: (source['DEEPSEEK_BASE_URL']?.trim() || 'https://api.deepseek.com').replace(/\/+$/, ''),
    typesafeKey: source['TYPESAFE_API_KEY']?.trim() ?? '',
    simiFreeQuestions: integer(source['SIMI_FREE_QUESTIONS'], 10),
    hybridFreeRuns: integer(source['HYBRID_FREE_RUNS'], 3),
    simiIpDaily: integer(source['SIMI_IP_DAILY_LIMIT'], 100),
    hybridIpDaily: integer(source['HYBRID_IP_DAILY_LIMIT'], 20),
    labelIpDaily: integer(source['LABEL_IP_DAILY_LIMIT'], 400),
    quotaSalt: source['QUOTA_SALT']?.trim() || 'simetrics',
  };
}
