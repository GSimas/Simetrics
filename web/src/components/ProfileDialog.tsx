import { useState, type FormEvent, type ReactNode } from 'react';
import { Loader2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import {
  changePassword,
  deleteSimetricsData,
  hasPassword,
  updateDisplayName,
  updateEmail,
  useAuth,
} from '@/state/auth.store';
import { useLocale } from '@/state/locale.store';

const MIN_PASSWORD = 8;

type Status = { kind: 'ok' | 'error'; text: string } | null;

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Envia uma ação do formulário com estado de carregamento e mensagem de resultado. */
function useAction() {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status>(null);
  const run = async (action: () => Promise<string>) => {
    setBusy(true);
    setStatus(null);
    try {
      setStatus({ kind: 'ok', text: await action() });
    } catch (cause) {
      setStatus({ kind: 'error', text: errorText(cause) });
    } finally {
      setBusy(false);
    }
  };
  return { busy, status, run };
}

function StatusLine({ status }: { status: Status }) {
  if (!status) return null;
  return (
    <p role={status.kind === 'error' ? 'alert' : 'status'} className={cn('text-sm', status.kind === 'error' ? 'text-destructive' : 'text-highlight')}>
      {status.text}
    </p>
  );
}

/** Bloco numerado da janela, no traço das páginas de privacidade e termos. */
function Section({ index, title, children, danger }: { index: string; title: string; children: ReactNode; danger?: boolean }) {
  return (
    <section className="grid gap-3 border-t border-border pt-5 sm:grid-cols-[2.5rem_1fr]">
      <span className={cn('font-mono text-xs', danger ? 'text-destructive' : 'text-highlight')}>{index}</span>
      <div className="min-w-0 space-y-3">
        <h3 className={cn('text-base font-semibold tracking-[-0.02em]', danger && 'text-destructive')}>{title}</h3>
        {children}
      </div>
    </section>
  );
}

/**
 * Perfil da conta Scientata: nome de exibição, e-mail de acesso, senha e exclusão dos dados
 * do Simetrics. E-mail e senha só aparecem para quem entra com senha; quem entra com Google
 * ou Microsoft tem o e-mail gerido pelo provedor.
 */
export function ProfileDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const t = useLocale((state) => state.t);
  const user = useAuth((state) => state.user);

  const [name, setName] = useState((user?.user_metadata?.full_name as string | undefined) ?? '');
  const [email, setEmail] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmWord, setConfirmWord] = useState('');

  const nameAction = useAction();
  const emailAction = useAction();
  const passwordAction = useAction();
  const deleteAction = useAction();

  if (!user) return null;
  const withPassword = hasPassword(user);
  const providers = [...new Set((user.identities ?? []).map((identity) => identity.provider))];
  const providerLabel = (provider: string) =>
    ({ email: t('profile_provider_email'), google: 'Google', azure: 'Microsoft' })[provider] ?? provider;
  const deleteWord = t('profile_delete_word');

  const submitName = (event: FormEvent) => {
    event.preventDefault();
    void nameAction.run(async () => {
      await updateDisplayName(name.trim());
      return t('profile_saved');
    });
  };

  const submitEmail = (event: FormEvent) => {
    event.preventDefault();
    void emailAction.run(async () => {
      await updateEmail(email.trim());
      setEmail('');
      return t('profile_email_sent');
    });
  };

  const submitPassword = (event: FormEvent) => {
    event.preventDefault();
    void passwordAction.run(async () => {
      await changePassword(user.email ?? '', currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      return t('profile_password_changed');
    });
  };

  const submitDelete = (event: FormEvent) => {
    event.preventDefault();
    void deleteAction.run(async () => {
      await deleteSimetricsData();
      onOpenChange(false);
      return '';
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90dvh] max-w-xl overflow-y-auto">
        <DialogHeader>
          <p className="eyebrow">— {t('auth_account')}</p>
          <DialogTitle className="text-2xl font-medium tracking-[-0.04em]">{t('profile_title')}</DialogTitle>
          <DialogDescription>
            {user.email} · {t('profile_signs_in_with')} {providers.map(providerLabel).join(', ')}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          <Section index="01" title={t('profile_name')}>
            <form onSubmit={submitName} className="flex flex-col gap-2 sm:flex-row">
              <Label htmlFor="profile-name" className="sr-only">
                {t('profile_name')}
              </Label>
              <Input id="profile-name" required autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} />
              <Button type="submit" variant="outline" disabled={nameAction.busy || !name.trim()}>
                {nameAction.busy && <Loader2 className="size-4 animate-spin" />}
                {t('profile_save')}
              </Button>
            </form>
            <p className="text-xs text-muted-foreground">{t('profile_name_hint')}</p>
            <StatusLine status={nameAction.status} />
          </Section>

          <Section index="02" title={t('profile_email')}>
            {withPassword ? (
              <>
                <form onSubmit={submitEmail} className="flex flex-col gap-2 sm:flex-row">
                  <Label htmlFor="profile-email" className="sr-only">
                    {t('profile_email_new')}
                  </Label>
                  <Input
                    id="profile-email"
                    type="email"
                    required
                    autoComplete="email"
                    placeholder={t('profile_email_new')}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <Button type="submit" variant="outline" disabled={emailAction.busy || !email.trim()}>
                    {emailAction.busy && <Loader2 className="size-4 animate-spin" />}
                    {t('profile_change')}
                  </Button>
                </form>
                <StatusLine status={emailAction.status} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">{t('profile_email_oauth')}</p>
            )}
          </Section>

          <Section index="03" title={t('profile_password')}>
            {withPassword ? (
              <form onSubmit={submitPassword} className="grid gap-2">
                <Label htmlFor="profile-current-password">{t('profile_password_current')}</Label>
                <Input
                  id="profile-current-password"
                  type="password"
                  required
                  autoComplete="current-password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                />
                <Label htmlFor="profile-new-password" className="pt-1">
                  {t('profile_password_new')}
                </Label>
                <Input
                  id="profile-new-password"
                  type="password"
                  required
                  minLength={MIN_PASSWORD}
                  autoComplete="new-password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">{t('auth_password_min')}</p>
                <StatusLine status={passwordAction.status} />
                <Button type="submit" variant="outline" className="justify-self-start" disabled={passwordAction.busy}>
                  {passwordAction.busy && <Loader2 className="size-4 animate-spin" />}
                  {t('profile_password_change')}
                </Button>
              </form>
            ) : (
              <p className="text-sm text-muted-foreground">{t('profile_password_oauth')}</p>
            )}
          </Section>

          <Section index="04" title={t('profile_delete_title')} danger>
            <p className="text-sm leading-relaxed text-muted-foreground">{t('profile_delete_desc')}</p>
            <form onSubmit={submitDelete} className="grid gap-2">
              <Label htmlFor="profile-delete-confirm">{t('profile_delete_confirm').replace('{word}', deleteWord)}</Label>
              <Input
                id="profile-delete-confirm"
                autoComplete="off"
                value={confirmWord}
                onChange={(e) => setConfirmWord(e.target.value)}
              />
              <StatusLine status={deleteAction.status?.kind === 'error' ? deleteAction.status : null} />
              <Button
                type="submit"
                variant="destructive"
                className="justify-self-start"
                disabled={deleteAction.busy || confirmWord.trim().toLowerCase() !== deleteWord.toLowerCase()}
              >
                {deleteAction.busy && <Loader2 className="size-4 animate-spin" />}
                {t('profile_delete_button')}
              </Button>
            </form>
          </Section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
