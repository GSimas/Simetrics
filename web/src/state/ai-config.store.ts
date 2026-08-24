import { create } from 'zustand';

export type AiProvider = 'gemini' | 'openai' | 'claude' | 'openrouter' | 'custom';

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

export const DEFAULT_MODELS: Record<AiProvider, string> = {
  gemini: 'gemini-3.7-flash',
  openai: 'gpt-5.6-terra',
  claude: 'claude-sonnet-5',
  openrouter: 'google/gemini-3.7-flash',
  custom: 'gpt-oss-20b',
};

export const PROVIDER_MODELS: Record<AiProvider, ModelOption[]> = {
  gemini: [
    { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', badge: 'Flagship · 1M Contexto' },
    { id: 'gemini-3.7-flash', name: 'Gemini 3.7 Flash', badge: 'Mais Recente · Coding' },
    { id: 'gemini-3.6-flash', name: 'Gemini 3.6 Flash', badge: 'GA · Alta Eficiência' },
    { id: 'gemini-3.5-flash', name: 'Gemini 3.5 Flash', badge: 'GA · Baixo Custo' },
    { id: 'gemini-3.5-flash-lite', name: 'Gemini 3.5 Flash-Lite', badge: 'Ultrarrápido' },
    { id: 'gemini-3-flash', name: 'Gemini 3 Flash', badge: 'Preview' },
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', badge: 'Estável' },
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', badge: 'Estável' },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash-Lite', badge: 'Econômico' },
    { id: 'gemma-4-31b', name: 'Gemma 4 31B (Aberto)', badge: 'Open Weights' },
    { id: 'gemma-3-27b', name: 'Gemma 3 27B (Aberto)', badge: 'Open Weights' },
  ],
  openai: [
    { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol', badge: 'Topo de Linha · Raciocínio' },
    { id: 'gpt-5.6-terra', name: 'GPT-5.6 Terra', badge: 'Recomendado · Padrão' },
    { id: 'gpt-5.6-luna', name: 'GPT-5.6 Luna', badge: 'Econômico · Alto Volume' },
    { id: 'gpt-5.6-luna-pro', name: 'GPT-5.6 Luna Pro', badge: 'Alto Desempenho' },
    { id: 'gpt-5.3-codex', name: 'GPT-5.3 Codex', badge: 'Coding & Agentes' },
    { id: 'gpt-5.5', name: 'GPT-5.5', badge: 'Geração Anterior' },
    { id: 'gpt-5.4', name: 'GPT-5.4', badge: 'Geração Anterior' },
    { id: 'gpt-4.1', name: 'GPT-4.1', badge: '1M Tokens' },
    { id: 'gpt-4o', name: 'GPT-4o', badge: 'Clássico' },
    { id: 'gpt-4o-mini', name: 'GPT-4o Mini', badge: 'Clássico' },
  ],
  claude: [
    { id: 'claude-opus-4-8', name: 'Claude Opus 4.8', badge: 'Mais Capaz · Topo' },
    { id: 'claude-sonnet-5', name: 'Claude Sonnet 5', badge: 'Recomendado · Alto Desempenho' },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', badge: 'Rápido & Econômico' },
    { id: 'claude-fable-5', name: 'Claude Fable 5', badge: 'Mythos com Salvaguardas' },
    { id: 'claude-opus-4-7', name: 'Claude Opus 4.7' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
    { id: 'claude-opus-4-5', name: 'Claude Opus 4.5' },
    { id: 'claude-sonnet-3-7', name: 'Claude Sonnet 3.7' },
    { id: 'claude-haiku-3-5', name: 'Claude Haiku 3.5' },
  ],
  openrouter: [
    { id: 'google/gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro (Google)', badge: 'Flagship 1M' },
    { id: 'google/gemini-3.7-flash', name: 'Gemini 3.7 Flash (Google)', badge: 'Recomendado' },
    { id: 'openai/gpt-5.6-sol', name: 'GPT-5.6 Sol (OpenAI)', badge: 'Topo de Linha' },
    { id: 'openai/gpt-5.6-terra', name: 'GPT-5.6 Terra (OpenAI)', badge: 'Padrão' },
    { id: 'openai/gpt-5.6-luna', name: 'GPT-5.6 Luna (OpenAI)', badge: 'Econômico' },
    { id: 'anthropic/claude-opus-4-8', name: 'Claude Opus 4.8 (Anthropic)', badge: 'Mais Capaz' },
    { id: 'anthropic/claude-sonnet-5', name: 'Claude Sonnet 5 (Anthropic)', badge: 'Alto Desempenho' },
    { id: 'anthropic/claude-haiku-4-5', name: 'Claude Haiku 4.5 (Anthropic)', badge: 'Rápido' },
    { id: 'deepseek/deepseek-v4', name: 'DeepSeek V4 (DeepSeek)', badge: 'Raciocínio' },
    { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1 (DeepSeek)', badge: 'Raciocínio Aberto' },
    { id: 'meta-llama/llama-4-scout', name: 'Llama 4 Scout (Meta)', badge: 'Contexto Longo' },
    { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B (Meta)' },
    { id: 'qwen/qwen-3.8-72b', name: 'Qwen 3.8 72B (Alibaba)', badge: 'Mais Recente' },
    { id: 'qwen/qwen3-coder', name: 'Qwen 3 Coder (256K)', badge: 'Coding' },
    { id: 'mistralai/mistral-medium-3.5', name: 'Mistral Medium 3.5 (Mistral AI)' },
  ],
  custom: [
    { id: 'gpt-oss-120b', name: 'gpt-oss-120b (OpenAI / Apache 2.0)', badge: '120B Pesos Abertos' },
    { id: 'gpt-oss-20b', name: 'gpt-oss-20b (OpenAI / Apache 2.0)', badge: 'Recomendado 20B' },
    { id: 'qwen3.8', name: 'Qwen 3.8 (Alibaba)', badge: 'Mais Recente' },
    { id: 'qwen3.6', name: 'Qwen 3.6 (Alibaba)', badge: 'Multilíngue' },
    { id: 'qwen3-coder', name: 'Qwen 3 Coder (Alibaba)', badge: 'Coding 256K' },
    { id: 'llama-4-scout', name: 'Llama 4 Scout (Meta)', badge: 'Contexto Ultra Longo' },
    { id: 'llama3.3:70b', name: 'Llama 3.3 70B (Meta)' },
    { id: 'gemma-4:31b', name: 'Gemma 4 31B (Google)', badge: 'Gemma 4' },
    { id: 'deepseek-v4', name: 'DeepSeek V4 (Raciocínio)', badge: 'Fronteira' },
    { id: 'deepseek-r1:8b', name: 'DeepSeek R1 8B (Ollama)', badge: 'Leve & Rápido' },
    { id: 'deepseek-r1:14b', name: 'DeepSeek R1 14B (Ollama)' },
    { id: 'mistral-small:3.5', name: 'Mistral Small 3.5 (Mistral)' },
    { id: 'devstral:24b', name: 'Devstral 24B (Coding Agêntico)', badge: 'Agente Local' },
    { id: 'glm-5.2', name: 'GLM 5.2 (Zhipu AI)', badge: 'Fronteira' },
    { id: 'kimi-k3', name: 'Kimi K3 (Moonshot AI)', badge: 'Fronteira' },
  ],
};

export const PROVIDER_OPTIONS: { id: AiProvider; label: string; placeholder: string; helpUrl: string }[] = [
  {
    id: 'gemini',
    label: 'Google Gemini',
    placeholder: 'AIzaSy...',
    helpUrl: 'https://aistudio.google.com/app/apikey',
  },
  {
    id: 'openai',
    label: 'OpenAI (ChatGPT)',
    placeholder: 'sk-proj-...',
    helpUrl: 'https://platform.openai.com/api-keys',
  },
  {
    id: 'claude',
    label: 'Anthropic (Claude)',
    placeholder: 'sk-ant-...',
    helpUrl: 'https://console.anthropic.com/settings/keys',
  },
  {
    id: 'openrouter',
    label: 'OpenRouter (Multi-Modelos 500+)',
    placeholder: 'sk-or-...',
    helpUrl: 'https://openrouter.ai/keys',
  },
  {
    id: 'custom',
    label: 'Local / OpenAI-Compatible (Ollama, LM Studio, vLLM)',
    placeholder: 'sk-... (opcional para local)',
    helpUrl: 'https://ollama.com/',
  },
];

interface AiConfigState {
  config: AiConfig;
  setConfig: (config: Partial<AiConfig>) => void;
  clearConfig: () => void;
  isConfigured: () => boolean;
}

const STORAGE_KEY = 'simetrics_ai_byok_config';

function loadSavedConfig(): AiConfig {
  if (typeof window === 'undefined') {
    return { provider: 'gemini', apiKey: '', model: DEFAULT_MODELS.gemini };
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AiConfig>;
      const provider = parsed.provider ?? 'gemini';
      const config: AiConfig = {
        provider,
        apiKey: parsed.apiKey ?? '',
        model: parsed.model || DEFAULT_MODELS[provider],
      };
      if (parsed.baseUrl) config.baseUrl = parsed.baseUrl;
      return config;
    }
  } catch {
    // Ignora erro de JSON e cai no padrão
  }
  return { provider: 'gemini', apiKey: '', model: DEFAULT_MODELS.gemini };
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
    const reset: AiConfig = { provider: 'gemini', apiKey: '', model: DEFAULT_MODELS.gemini };
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignora
    }
    set({ config: reset });
  },
  isConfigured: () => {
    const { apiKey, provider, baseUrl } = get().config;
    if (provider === 'custom') {
      return Boolean(baseUrl && baseUrl.trim());
    }
    return Boolean(apiKey && apiKey.trim().length > 3);
  },
}));
