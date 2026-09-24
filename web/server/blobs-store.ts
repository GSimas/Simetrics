import { getStore } from '@netlify/blobs';

import type { QuotaRecord, QuotaStore } from './quota.ts';

/**
 * Cotas persistidas no Netlify Blobs, em produção. Consistência forte: com a eventual,
 * uma pergunta feita logo após outra poderia ler o contador antigo.
 */
export class BlobsQuotaStore implements QuotaStore {
  private readonly store = getStore({ name: 'simetrics-quota', consistency: 'strong' });

  async read(key: string): Promise<QuotaRecord | null> {
    const value = (await this.store.get(key, { type: 'json' })) as QuotaRecord | null;
    return value && typeof value === 'object' && value.units ? value : null;
  }

  async write(key: string, record: QuotaRecord): Promise<void> {
    await this.store.setJSON(key, record);
  }
}
