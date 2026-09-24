import { afterEach, describe, expect, it, vi } from 'vitest';

import { readServerEnv } from '../server/env';
import { handleApi, type ApiContext } from '../server/handlers';
import { consume, MemoryQuotaStore, refund, type QuotaPolicy } from '../server/quota';

const DEVICE = '11111111-2222-3333-4444-555555555555';
const policy: QuotaPolicy = { scope: 'simi', deviceLimit: 2, ipDailyLimit: 3, maxRequestsPerUnit: 2 };

function context(overrides: Record<string, string> = {}): ApiContext {
  return {
    env: readServerEnv({ DEEPSEEK_API_KEY: 'sk-test', TYPESAFE_API_KEY: 'ts-test', ...overrides }),
    store: new MemoryQuotaStore(),
    ip: '10.0.0.1',
  };
}

function chat(unit: string, device = DEVICE): Request {
  return new Request('http://localhost/api/simi/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Simetrics-Device': device, 'X-Simetrics-Unit': unit },
    body: JSON.stringify({ model: 'anything', messages: [{ role: 'user', content: 'oi' }], max_tokens: 999999, stream: true }),
  });
}

afterEach(() => vi.unstubAllGlobals());

describe('quota', () => {
  it('counts units, not requests, and caps requests per unit', async () => {
    const store = new MemoryQuotaStore();
    const id = { device: DEVICE, ip: '1.1.1.1' };
    expect(await consume(store, policy, id, 'unit-aaaa', 's')).toMatchObject({ ok: true, used: 1, newUnit: true });
    expect(await consume(store, policy, id, 'unit-aaaa', 's')).toMatchObject({ ok: true, used: 1, newUnit: false });
    expect(await consume(store, policy, id, 'unit-aaaa', 's')).toMatchObject({ ok: false, reason: 'unit' });
    expect(await consume(store, policy, id, 'unit-bbbb', 's')).toMatchObject({ ok: true, used: 2 });
    expect(await consume(store, policy, id, 'unit-cccc', 's')).toMatchObject({ ok: false, reason: 'device' });
  });

  it('applies a daily cap per IP across devices', async () => {
    const store = new MemoryQuotaStore();
    for (let i = 0; i < 3; i += 1) {
      expect((await consume(store, policy, { device: `device-${i}xxxx`, ip: '2.2.2.2' }, `unit-${i}xxxx`, 's')).ok).toBe(true);
    }
    expect(await consume(store, policy, { device: 'device-9xxxx', ip: '2.2.2.2' }, 'unit-9xxxx', 's')).toMatchObject({
      ok: false,
      reason: 'ip',
    });
  });

  it('refunds a unit whose first call failed', async () => {
    const store = new MemoryQuotaStore();
    const id = { device: DEVICE, ip: '1.1.1.1' };
    await consume(store, policy, id, 'unit-aaaa', 's');
    await refund(store, policy, id, 'unit-aaaa', 's');
    expect(await consume(store, policy, id, 'unit-bbbb', 's')).toMatchObject({ ok: true, used: 1 });
  });

  it('never stores raw identifiers', async () => {
    const store = new MemoryQuotaStore();
    const write = vi.spyOn(store, 'write');
    await consume(store, policy, { device: DEVICE, ip: '9.9.9.9' }, 'unit-aaaa', 's');
    for (const [key] of write.mock.calls) {
      expect(key).not.toContain(DEVICE);
      expect(key).not.toContain('9.9.9.9');
    }
  });
});

describe('api handlers', () => {
  it('reports availability and remaining quota', async () => {
    const response = await handleApi(
      new Request('http://localhost/api/status', { headers: { 'X-Simetrics-Device': DEVICE } }),
      context(),
    );
    expect(await response!.json()).toMatchObject({
      deepseek: { available: true, model: 'deepseek-flash' },
      jev: { available: true },
      simi: { limit: 10, used: 0, remaining: 10 },
      hybrid: { limit: 3, used: 0, remaining: 3 },
    });
  });

  it('returns null for unknown paths', async () => {
    expect(await handleApi(new Request('http://localhost/api/other'), context())).toBeNull();
  });

  it('proxies to DeepSeek with the server key, forced model and capped tokens', async () => {
    const fetchMock = vi.fn(async () => new Response('data: {}\n\n', { headers: { 'content-type': 'text/event-stream' } }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await handleApi(chat('unit-aaaa'), context());
    expect(response!.status).toBe(200);
    expect(response!.headers.get('X-Free-Used')).toBe('1');

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://api.deepseek.com/chat/completions');
    expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer sk-test');
    const body = JSON.parse(String(init.body));
    expect(body.model).toBe('deepseek-flash');
    expect(body.max_tokens).toBe(8192);
  });

  it('stops after the free questions with a 429 in the OpenAI error format', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}')));
    const ctx = context({ SIMI_FREE_QUESTIONS: '2' });
    expect((await handleApi(chat('unit-aaaa'), ctx))!.status).toBe(200);
    expect((await handleApi(chat('unit-bbbb'), ctx))!.status).toBe(200);
    const blocked = await handleApi(chat('unit-cccc'), ctx);
    expect(blocked!.status).toBe(429);
    const body = (await blocked!.json()) as { error: { message: string; code: string } };
    expect(body.error.code).toBe('quota_device');
    expect(body.error.message).toContain('2 perguntas gratuitas');
  });

  it('refuses free use without a server key', async () => {
    const response = await handleApi(chat('unit-aaaa'), context({ DEEPSEEK_API_KEY: '' }));
    expect(response!.status).toBe(503);
  });

  it('forwards Jev calls with the server key and without Origin', async () => {
    const fetchMock = vi.fn(async () => new Response('{"answers":{}}', { headers: { 'content-type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const response = await handleApi(
      new Request('http://localhost/api/jev/systemone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Origin: 'http://localhost:5173' },
        body: JSON.stringify({ state: 'x', model: 'jev-latest', questions: { q: { type: 'noul', instructions: 'x?' } } }),
      }),
      context(),
    );
    expect(response!.status).toBe(200);
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer ts-test');
    expect(headers['Origin']).toBeUndefined();
  });
});
