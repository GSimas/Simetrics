import { afterEach, describe, expect, it, vi } from 'vitest';

import { readServerEnv } from '../server/env';
import { handleApi, jevCallBudget, type ApiContext } from '../server/handlers';
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

function openRun(unit: string, docCount: number): Request {
  return new Request('http://localhost/api/hybrid/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Simetrics-Device': DEVICE, 'X-Simetrics-Unit': unit },
    body: JSON.stringify({ docCount }),
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
      freeMaxDocs: 1000,
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
    expect(body.thinking).toBeUndefined();
  });

  it('forwards only a valid thinking switch', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const ctx = context();
    const send = (thinking: unknown, unit: string) =>
      handleApi(
        new Request('http://localhost/api/hybrid/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Simetrics-Device': DEVICE, 'X-Simetrics-Unit': unit },
          body: JSON.stringify({ messages: [{ role: 'user', content: 'oi' }], thinking }),
        }),
        ctx,
      );
    const sent = (call: number) => JSON.parse(String((fetchMock.mock.calls[call] as unknown as [string, RequestInit])[1].body));

    await send({ type: 'disabled', budget: 1e9 }, 'unit-aaaa');
    expect(sent(0).thinking).toEqual({ type: 'disabled' });
    await send({ type: 'max' }, 'unit-bbbb');
    expect(sent(1).thinking).toBeUndefined();
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

  it('forwards Jev calls with the server key and without Origin, only inside an open run', async () => {
    const fetchMock = vi.fn(async () => new Response('{"answers":{}}', { headers: { 'content-type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    const ctx = context();
    const jev = (unit: string) =>
      handleApi(
        new Request('http://localhost/api/jev/systemone', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Origin: 'http://localhost:5173',
            'X-Simetrics-Device': DEVICE,
            'X-Simetrics-Unit': unit,
          },
          body: JSON.stringify({ state: 'x', model: 'jev-latest', questions: { q: { type: 'noul', instructions: 'x?' } } }),
        }),
        ctx,
      );

    expect((await jev('unit-aaaa'))!.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();

    expect((await handleApi(openRun('unit-aaaa', 10), ctx))!.status).toBe(200);
    expect((await jev('unit-aaaa'))!.status).toBe(200);
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer ts-test');
    expect(headers['Origin']).toBeUndefined();
  });

  it('refuses free runs above the document cap and stops Jev at the run budget', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } })));
    const ctx = context({ HYBRID_FREE_MAX_DOCS: '1000' });
    expect((await handleApi(openRun('unit-bbbb', 1001), ctx))!.status).toBe(413);
    expect((await handleApi(openRun('unit-bbbb', 1), ctx))!.status).toBe(200);

    const budget = jevCallBudget(1);
    const call = () =>
      handleApi(
        new Request('http://localhost/api/jev/systemone', {
          method: 'POST',
          headers: { 'X-Simetrics-Device': DEVICE, 'X-Simetrics-Unit': 'unit-bbbb' },
          body: JSON.stringify({ questions: { q: {} } }),
        }),
        ctx,
      );
    for (let i = 0; i < budget; i += 1) expect((await call())!.status).toBe(200);
    const over = await call();
    expect(over!.status).toBe(429);
    expect(over!.headers.get('X-Free-Limit')).toBe(String(budget));

    // Reabrir a mesma execução conta como a mesma classificação e não zera o orçamento.
    expect((await handleApi(openRun('unit-bbbb', 1), ctx))!.status).toBe(200);
    expect((await call())!.status).toBe(429);
  });

  it('answers fixed error messages in English when the client asks for it', async () => {
    const bad = (locale: string) =>
      handleApi(
        new Request('http://localhost/api/hybrid/runs', {
          method: 'POST',
          headers: { 'X-Simetrics-Locale': locale },
          body: 'not json',
        }),
        context(),
      );
    expect(((await (await bad('en'))!.json()) as { error: { message: string } }).error.message).toBe(
      'The request body is not valid JSON.',
    );
    expect(((await (await bad('pt'))!.json()) as { error: { message: string } }).error.message).toBe(
      'Corpo da requisição não é um JSON válido.',
    );
  });

  it('counts free runs against the per-device classification quota', async () => {
    const ctx = context({ HYBRID_FREE_RUNS: '1' });
    expect((await handleApi(openRun('unit-cccc', 5), ctx))!.status).toBe(200);
    expect((await handleApi(openRun('unit-dddd', 5), ctx))!.status).toBe(429);
  });
});
