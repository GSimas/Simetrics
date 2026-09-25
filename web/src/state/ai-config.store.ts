import { create } from 'zustand';

import type { Locale } from '@/lib/i18n/translations';

/**
 * Provedores e modelos do chat BYOK.
 *
 * Todos são chamados direto do navegador, então só entram provedores cuja API responde ao
 * preflight de CORS. Gemini e Anthropic têm protocolo próprio; os demais falam o formato de
 * chat completions da OpenAI e diferem só na URL base.
 *
 * Lista revisada em 2026-09-24 contra as fontes oficiais (páginas de modelos de cada
 * provedor, openrouter.ai/api/v1/models e ollama.com/library), mantendo só modelos com
 * chamada de ferramentas — o Simi depende delas.
 */

export type AiProvider =
  | 'gemini'
  | 'openai'
  | 'claude'
  | 'deepseek'
  | 'mistral'
  | 'xai'
  | 'groq'
  | 'cerebras'
  | 'together'
  | 'fireworks'
  | 'moonshot'
  | 'qwen'
  | 'zhipu'
  | 'cohere'
  | 'huggingface'
  | 'openrouter'
  | 'custom';

export interface AiConfig {
  provider: AiProvider;
  apiKey: string;
  model: string;
  baseUrl?: string;
}

export interface ModelOption {
  id: string;
  name: string;
  badge?: string;
}

/** Termos dos selos de modelo em inglês; os selos combinam termos separados por " · ". */
const BADGE_TERMS_EN: Record<string, string> = {
  Recomendado: 'Recommended',
  '1M contexto': '1M context',
  Estável: 'Stable',
  Raciocínio: 'Reasoning',
  Econômico: 'Budget',
  Ultrarrápido: 'Ultra-fast',
  'Pesos abertos': 'Open weights',
  'Topo de linha': 'Flagship',
  Agentes: 'Agents',
  'Alto volume': 'High volume',
  'Geração anterior': 'Previous generation',
  'Mais recente': 'Latest',
  Equilíbrio: 'Balanced',
  'Mais capaz': 'Most capable',
  Rápido: 'Fast',
  Legado: 'Legacy',
  Leve: 'Lightweight',
  Gratuito: 'Free',
  'Custo baixo': 'Low cost',
  'MoE rápido': 'Fast MoE',
};

export function modelBadgeLabel(badge: string, locale: Locale): string {
  if (locale !== 'en') return badge;
  return badge
    .split(' · ')
    .map((term) => BADGE_TERMS_EN[term] ?? term)
    .join(' · ');
}

export const DEFAULT_MODELS: Record<AiProvider, string> = {
  gemini: 'gemini-3.8-flash',
  openai: 'gpt-5.6-terra',
  claude: 'claude-opus-5',
  deepseek: 'deepseek-flash',
  mistral: 'mistral-medium-2604',
  xai: 'grok-4.7',
  groq: 'openai/gpt-oss-120b',
  cerebras: 'gpt-oss-120b',
  together: 'moonshotai/Kimi-K3',
  fireworks: 'accounts/fireworks/models/deepseek-v4p1-flash',
  moonshot: 'kimi-k3',
  qwen: 'qwen3.8-flash',
  zhipu: 'glm-5.3',
  cohere: 'command-a-plus-05-2026',
  huggingface: 'deepseek-ai/DeepSeek-V4.1-Flash',
  openrouter: 'google/gemini-3.8-flash',
  custom: 'gpt-oss:20b',
};

export const PROVIDER_MODELS: Record<AiProvider, ModelOption[]> = {
  gemini: [
    { id: 'gemini-3.8-flash', name: 'Gemini 3.8 Flash', badge: 'Recomendado · 1M contexto' },
    { id: 'gemini-3.7-flash', name: 'Gemini 3.7 Flash', badge: 'Estável' },
    { id: 'gemini-3.6-flash', name: 'Gemini 3.6 Flash', badge: 'Estável' },
    { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', badge: 'Raciocínio · Preview' },
    { id: 'gemini-3.5-flash-lite', name: 'Gemini 3.5 Flash-Lite', badge: 'Econômico' },
    { id: 'gemini-3.1-flash-lite', name: 'Gemini 3.1 Flash-Lite', badge: 'Ultrarrápido' },
    { id: 'gemma-4-31b-it', name: 'Gemma 4 31B', badge: 'Pesos abertos' },
    { id: 'gemma-4-26b-a4b-it', name: 'Gemma 4 26B A4B', badge: 'Pesos abertos · MoE' },
  ],
  openai: [
    { id: 'gpt-5.6-terra', name: 'GPT-5.6 Terra', badge: 'Recomendado' },
    { id: 'gpt-6-sol', name: 'GPT-6 Sol', badge: 'Topo de linha · Agentes' },
    { id: 'gpt-6-luna', name: 'GPT-6 Luna', badge: 'Econômico' },
    { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol', badge: 'Raciocínio' },
    { id: 'gpt-5.6-luna', name: 'GPT-5.6 Luna', badge: 'Alto volume' },
    { id: 'gpt-5.5', name: 'GPT-5.5', badge: 'Geração anterior' },
    { id: 'gpt-4.1', name: 'GPT-4.1', badge: '1M contexto' },
  ],
  claude: [
    { id: 'claude-opus-5', name: 'Claude Opus 5', badge: 'Recomendado · 1M' },
    { id: 'claude-opus-5-5', name: 'Claude Opus 5.5', badge: 'Mais recente · 1M' },
    { id: 'claude-sonnet-5', name: 'Claude Sonnet 5', badge: 'Equilíbrio · 1M' },
    { id: 'claude-fable-5-1', name: 'Claude Fable 5.1', badge: 'Mais capaz' },
    { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', badge: 'Rápido · Econômico' },
    { id: 'claude-opus-4-8', name: 'Claude Opus 4.8', badge: 'Legado' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', badge: 'Legado' },
  ],
  deepseek: [
    { id: 'deepseek-flash', name: 'DeepSeek V4.1 Flash', badge: 'Recomendado · 1M · Econômico' },
    { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro', badge: 'Raciocínio' },
  ],
  mistral: [
    { id: 'mistral-medium-2604', name: 'Mistral Medium', badge: 'Recomendado' },
    { id: 'mistral-small-2603', name: 'Mistral Small', badge: 'Econômico' },
    { id: 'mistral-large-2512', name: 'Mistral Large', badge: 'Pesos abertos' },
    { id: 'ministral-3-8b-2512', name: 'Ministral 3 8B', badge: 'Leve' },
  ],
  xai: [
    { id: 'grok-4.7', name: 'Grok 4.7', badge: 'Recomendado' },
    { id: 'grok-4.6', name: 'Grok 4.6' },
    { id: 'grok-4.3', name: 'Grok 4.3', badge: '1M · Econômico' },
  ],
  groq: [
    { id: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', badge: 'Ultrarrápido' },
    { id: 'openai/gpt-oss-20b', name: 'gpt-oss-20b', badge: 'Leve' },
    { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B' },
    { id: 'qwen/qwen3.8-27b', name: 'Qwen 3.8 27B', badge: 'Preview' },
  ],
  cerebras: [
    { id: 'gpt-oss-120b', name: 'gpt-oss-120b', badge: 'Ultrarrápido' },
    { id: 'qwen-3.8-27b', name: 'Qwen 3.8 27B' },
  ],
  together: [
    { id: 'moonshotai/Kimi-K3', name: 'Kimi K3', badge: 'Recomendado' },
    { id: 'zai-org/GLM-5.3', name: 'GLM 5.3' },
    { id: 'deepseek-ai/DeepSeek-V4.1-Flash', name: 'DeepSeek V4.1 Flash', badge: 'Econômico' },
    { id: 'deepseek-ai/DeepSeek-V4-Pro-0813', name: 'DeepSeek V4 Pro', badge: 'Raciocínio' },
    { id: 'openai/gpt-oss-120b', name: 'gpt-oss-120b' },
    { id: 'meta-llama/Llama-3.3-70B-Instruct-Turbo', name: 'Llama 3.3 70B Turbo' },
  ],
  fireworks: [
    { id: 'accounts/fireworks/models/deepseek-v4p1-flash', name: 'DeepSeek V4.1 Flash', badge: 'Recomendado' },
    { id: 'accounts/fireworks/models/glm-5p3', name: 'GLM 5.3' },
    { id: 'accounts/fireworks/models/kimi-k3', name: 'Kimi K3' },
    { id: 'accounts/fireworks/models/deepseek-v4-pro-0813', name: 'DeepSeek V4 Pro', badge: 'Raciocínio' },
    { id: 'accounts/fireworks/models/gpt-oss-120b', name: 'gpt-oss-120b' },
  ],
  moonshot: [
    { id: 'kimi-k3', name: 'Kimi K3', badge: 'Recomendado · 1M' },
    { id: 'kimi-k2.6', name: 'Kimi K2.6' },
  ],
  qwen: [
    { id: 'qwen3.8-flash', name: 'Qwen 3.8 Flash', badge: 'Recomendado' },
    { id: 'qwen3.8-max', name: 'Qwen 3.8 Max', badge: 'Mais capaz' },
    { id: 'qwen3.7-plus', name: 'Qwen 3.7 Plus' },
  ],
  zhipu: [
    { id: 'glm-5.3', name: 'GLM 5.3', badge: 'Recomendado' },
    { id: 'glm-5.3-flash', name: 'GLM 5.3 Flash', badge: 'Rápido' },
    { id: 'glm-4.7-flash', name: 'GLM 4.7 Flash', badge: 'Gratuito' },
  ],
  cohere: [{ id: 'command-a-plus-05-2026', name: 'Command A+', badge: 'Recomendado' }],
  huggingface: [
    { id: 'deepseek-ai/DeepSeek-V4.1-Flash', name: 'DeepSeek V4.1 Flash', badge: 'Recomendado' },
    { id: 'zai-org/GLM-5.3', name: 'GLM 5.3' },
    { id: 'moonshotai/Kimi-K3', name: 'Kimi K3' },
    { id: 'Qwen/Qwen3.8-27B', name: 'Qwen 3.8 27B' },
    { id: 'google/gemma-4-31B-it', name: 'Gemma 4 31B', badge: 'Pesos abertos' },
  ],
  openrouter: [
    { id: 'google/gemini-3.8-flash', name: 'Gemini 3.8 Flash (Google)', badge: 'Recomendado' },
    { id: 'anthropic/claude-sonnet-5', name: 'Claude Sonnet 5 (Anthropic)', badge: 'Equilíbrio' },
    { id: 'anthropic/claude-opus-5.5', name: 'Claude Opus 5.5 (Anthropic)', badge: 'Mais capaz' },
    { id: 'openai/gpt-6-sol', name: 'GPT-6 Sol (OpenAI)', badge: 'Topo de linha' },
    { id: 'openai/gpt-6-luna', name: 'GPT-6 Luna (OpenAI)', badge: 'Econômico' },
    { id: 'deepseek/deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash', badge: 'Custo baixo' },
    { id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek V4 Pro', badge: 'Raciocínio' },
    { id: 'moonshotai/kimi-k3', name: 'Kimi K3 (Moonshot)' },
    { id: 'z-ai/glm-5.3', name: 'GLM 5.3 (Z.ai)' },
    { id: 'x-ai/grok-4.7', name: 'Grok 4.7 (xAI)' },
  ],
  custom: [
    { id: 'gpt-oss:20b', name: 'gpt-oss 20B', badge: 'Recomendado' },
    { id: 'qwen3.8:27b', name: 'Qwen 3.8 27B', badge: 'Mais recente' },
    { id: 'qwen3.6:35b-a3b', name: 'Qwen 3.6 35B A3B', badge: 'MoE rápido' },
    { id: 'gemma4:12b', name: 'Gemma 4 12B', badge: 'Leve' },
    { id: 'llama3.3:70b', name: 'Llama 3.3 70B' },
    { id: 'ministral-3:8b', name: 'Ministral 3 8B', badge: 'Leve' },
    { id: 'devstral-small-2:24b', name: 'Devstral Small 2 24B', badge: 'Agentes' },
    { id: 'gpt-oss:120b', name: 'gpt-oss 120B', badge: 'Pesos abertos' },
  ],
};

export interface ProviderOption {
  id: AiProvider;
  label: string;
  placeholder: string;
  /** Variantes em inglês, só onde o texto em português difere. */
  labelEn?: string;
  placeholderEn?: string;
  helpUrl: string;
  /** Endpoint compatível com a OpenAI; ausente em Gemini e Anthropic (protocolos próprios). */
  baseUrl?: string;
}

export const PROVIDER_OPTIONS: ProviderOption[] = [
  { id: 'gemini', label: 'Google Gemini', placeholder: 'AIzaSy...', helpUrl: 'https://aistudio.google.com/app/apikey' },
  {
    id: 'openai',
    label: 'OpenAI',
    placeholder: 'sk-proj-...',
    helpUrl: 'https://platform.openai.com/api-keys',
    baseUrl: 'https://api.openai.com/v1',
  },
  { id: 'claude', label: 'Anthropic (Claude)', placeholder: 'sk-ant-...', helpUrl: 'https://console.anthropic.com/settings/keys' },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    placeholder: 'sk-...',
    helpUrl: 'https://platform.deepseek.com/api_keys',
    baseUrl: 'https://api.deepseek.com',
  },
  {
    id: 'mistral',
    label: 'Mistral AI',
    placeholder: 'Chave da Mistral',
    placeholderEn: 'Mistral key',
    helpUrl: 'https://console.mistral.ai/api-keys',
    baseUrl: 'https://api.mistral.ai/v1',
  },
  { id: 'xai', label: 'xAI (Grok)', placeholder: 'xai-...', helpUrl: 'https://console.x.ai', baseUrl: 'https://api.x.ai/v1' },
  {
    id: 'groq',
    label: 'Groq',
    placeholder: 'gsk_...',
    helpUrl: 'https://console.groq.com/keys',
    baseUrl: 'https://api.groq.com/openai/v1',
  },
  {
    id: 'cerebras',
    label: 'Cerebras',
    placeholder: 'csk-...',
    helpUrl: 'https://cloud.cerebras.ai',
    baseUrl: 'https://api.cerebras.ai/v1',
  },
  {
    id: 'together',
    label: 'Together AI',
    placeholder: 'Chave da Together',
    placeholderEn: 'Together key',
    helpUrl: 'https://api.together.ai/settings/api-keys',
    baseUrl: 'https://api.together.xyz/v1',
  },
  {
    id: 'fireworks',
    label: 'Fireworks AI',
    placeholder: 'fw_...',
    helpUrl: 'https://app.fireworks.ai/account/api-keys',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
  },
  {
    id: 'moonshot',
    label: 'Moonshot (Kimi)',
    placeholder: 'sk-...',
    helpUrl: 'https://platform.kimi.ai/console/api-keys',
    baseUrl: 'https://api.moonshot.ai/v1',
  },
  {
    id: 'qwen',
    label: 'Alibaba Qwen (Model Studio)',
    placeholder: 'sk-...',
    helpUrl: 'https://modelstudio.console.alibabacloud.com',
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
  },
  {
    id: 'zhipu',
    label: 'Zhipu GLM (BigModel)',
    placeholder: 'Chave da BigModel',
    placeholderEn: 'BigModel key',
    helpUrl: 'https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys',
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
  },
  {
    id: 'cohere',
    label: 'Cohere',
    placeholder: 'Chave da Cohere',
    placeholderEn: 'Cohere key',
    helpUrl: 'https://dashboard.cohere.com/api-keys',
    baseUrl: 'https://api.cohere.ai/compatibility/v1',
  },
  {
    id: 'huggingface',
    label: 'Hugging Face (Inference Providers)',
    placeholder: 'hf_...',
    helpUrl: 'https://huggingface.co/settings/tokens',
    baseUrl: 'https://router.huggingface.co/v1',
  },
  {
    id: 'openrouter',
    label: 'OpenRouter (multimodelos)',
    labelEn: 'OpenRouter (multi-model)',
    placeholder: 'sk-or-...',
    helpUrl: 'https://openrouter.ai/keys',
    baseUrl: 'https://openrouter.ai/api/v1',
  },
  {
    id: 'custom',
    label: 'Local / compatível com OpenAI (Ollama, LM Studio, vLLM)',
    labelEn: 'Local / OpenAI-compatible (Ollama, LM Studio, vLLM)',
    placeholder: 'sk-... (opcional para local)',
    placeholderEn: 'sk-... (optional for local)',
    helpUrl: 'https://ollama.com/',
  },
];

const PROVIDER_IDS = new Set<string>(PROVIDER_OPTIONS.map((option) => option.id));

/** URL base de chat completions para os provedores compatíveis com a OpenAI. */
export function openAiCompatibleBaseUrl(config: AiConfig): string | null {
  if (config.provider === 'custom') return config.baseUrl?.trim() || 'http://localhost:11434/v1';
  return PROVIDER_OPTIONS.find((option) => option.id === config.provider)?.baseUrl ?? null;
}

/**
 * Identificadores que saíram de circulação ou estavam errados na lista anterior, por
 * provedor (o mesmo id pode ser válido em outro — `gpt-oss-120b` existe na Cerebras).
 * Uma configuração salva com eles é migrada ao carregar, em vez de falhar no primeiro uso.
 */
const RETIRED_MODELS: Record<string, string> = {
  'gemini:gemini-3-flash': 'gemini-3.8-flash',
  'gemini:gemma-4-31b': 'gemma-4-31b-it',
  'gemini:gemma-3-27b': 'gemma-4-31b-it',
  'gemini:gemini-2.5-pro': 'gemini-3.1-pro-preview',
  'gemini:gemini-2.5-flash': 'gemini-3.8-flash',
  'gemini:gemini-2.5-flash-lite': 'gemini-3.5-flash-lite',
  'claude:claude-sonnet-3-7': 'claude-sonnet-5',
  'claude:claude-haiku-3-5': 'claude-haiku-4-5',
  'claude:claude-opus-4-5': 'claude-opus-5',
  'claude:claude-fable-5': 'claude-fable-5-1',
  'openrouter:anthropic/claude-opus-4-8': 'anthropic/claude-opus-4.8',
  'openrouter:anthropic/claude-haiku-4-5': 'anthropic/claude-haiku-4.5',
  'openrouter:deepseek/deepseek-v4': 'deepseek/deepseek-v4-pro',
  'openrouter:deepseek/deepseek-r1': 'deepseek/deepseek-v4-pro',
  'openrouter:qwen/qwen-3.8-72b': 'google/gemini-3.8-flash',
  'openrouter:mistralai/mistral-medium-3.5': 'mistralai/mistral-medium-3-5',
  'custom:gpt-oss-120b': 'gpt-oss:120b',
  'custom:gpt-oss-20b': 'gpt-oss:20b',
  'custom:llama-4-scout': 'llama4:scout',
  'custom:gemma-4:31b': 'gemma4:31b',
  'custom:mistral-small:3.5': 'mistral-small3.2:24b',
  'custom:deepseek-v4': 'gpt-oss:20b',
};

export function migrateModel(provider: AiProvider, model: string): string {
  return RETIRED_MODELS[`${provider}:${model}`] ?? model;
}

interface AiConfigState {
  config: AiConfig;
  setConfig: (config: Partial<AiConfig>) => void;
  clearConfig: () => void;
  isConfigured: () => boolean;
}

const STORAGE_KEY = 'simetrics_ai_byok_config';
const DEFAULT_CONFIG: AiConfig = { provider: 'gemini', apiKey: '', model: DEFAULT_MODELS.gemini };

function loadSavedConfig(): AiConfig {
  if (typeof window === 'undefined') return DEFAULT_CONFIG;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AiConfig>;
      const provider: AiProvider = parsed.provider && PROVIDER_IDS.has(parsed.provider) ? parsed.provider : 'gemini';
      const config: AiConfig = {
        provider,
        apiKey: parsed.apiKey ?? '',
        model: migrateModel(provider, parsed.model || DEFAULT_MODELS[provider]),
      };
      if (parsed.baseUrl) config.baseUrl = parsed.baseUrl;
      return config;
    }
  } catch {
    // Ignora erro de JSON e cai no padrão
  }
  return DEFAULT_CONFIG;
}

export const useAiConfig = create<AiConfigState>((set, get) => ({
  config: loadSavedConfig(),
  setConfig: (partial) => {
    const current = get().config;
    const provider = partial.provider ?? current.provider;
    const next: AiConfig = {
      ...current,
      ...partial,
      provider,
      model: partial.model || (partial.provider && partial.provider !== current.provider ? DEFAULT_MODELS[provider] : current.model),
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Ignora erro de quota
    }
    set({ config: next });
  },
  clearConfig: () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignora
    }
    set({ config: DEFAULT_CONFIG });
  },
  isConfigured: () => {
    const { apiKey, provider, baseUrl } = get().config;
    if (provider === 'custom') {
      return Boolean(baseUrl && baseUrl.trim());
    }
    return Boolean(apiKey && apiKey.trim().length > 3);
  },
}));
