import { useState, type FormEvent } from 'react';
import { Cloud, CloudOff, Loader2, MailCheck, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useCloudSync } from '@/lib/cloud-sync';
import { cn } from '@/lib/utils';
import {
  sendPasswordReset,
  signInWithOAuth,
  signInWithPassword,
  signUp,
  updatePassword,
  useAuth,
  type OAuthProvider,
} from '@/state/auth.store';
import { useLocale } from '@/state/locale.store';

type Mode = 'signin' | 'signup' | 'reset';

const MIN_PASSWORD = 8;

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z" />
      <path fill="#FBBC05" d="M5.84 14.1A6.6 6.6 0 0 1 5.5 12c0-.73.13-1.44.34-2.1V7.06H2.18A11 11 0 0 0 1 12c0 1.77.42 3.45 1.18 4.94l3.66-2.84z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15A10.96 10.96 0 0 0 12 1 11 11 0 0 0 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
    </svg>
  );
}

function MicrosoftIcon() {
  return (
    <svg viewBox="0 0 23 23" className="size-4" aria-hidden>
      <path fill="#F25022" d="M1 1h10v10H1z" />
      <path fill="#7FBA00" d="M12 1h10v10H12z" />
      <path fill="#00A4EF" d="M1 12h10v10H1z" />
      <path fill="#FFB900" d="M12 12h10v10H12z" />
    </svg>
  );
}

export function AuthDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const t = useLocale((state) => state.t);
  const redirectError = useAuth((state) => state.redirectError);
  const [mode, setMode] = useState<Mode>('signin');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState<Mode | OAuthProvider | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Link enviado (confirmação de cadastro ou nova senha): a janela troca de tela.
  const [sent, setSent] = useState<'signup' | 'reset' | null>(null);

  const switchMode = (next: Mode) => {
    setMode(next);
    setError(null);
    setSent(null);
  };

  const oauth = async (provider: OAuthProvider) => {
    setBusy(provider);
    setError(null);
    try {
      await signInWithOAuth(provider); // redireciona para o provedor
    } catch (cause) {
      setError(errorText(cause));
      setBusy(null);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(mode);
    setError(null);
    try {
      if (mode === 'signin') {
        await signInWithPassword(email.trim(), password);
        onOpenChange(false);
      } else if (mode === 'signup') {
        const needsConfirmation = await signUp(email.trim(), password, name.trim());
        if (needsConfirmation) setSent('signup');
        else onOpenChange(false);
      } else {
        await sendPasswordReset(email.trim());
        setSent('reset');
      }
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(null);
    }
  };

  const title = { signin: t('auth_title_signin'), signup: t('auth_title_signup'), reset: t('auth_title_reset') }[mode];
  const submitLabel = { signin: t('auth_submit_signin'), signup: t('auth_submit_signup'), reset: t('auth_submit_reset') }[mode];

  const handleOpenChange = (next: boolean) => {
    if (!next) setSent(null);
    onOpenChange(next);
  };

  if (sent) {
    // O endereço vai em destaque no meio da frase traduzida.
    const [before, after] = t(sent === 'signup' ? 'auth_sent_signup' : 'auth_sent_reset').split('{email}');
    return (
      <Dialog open={open} onOpenChange={handleOpenChange}>
        <DialogContent className="max-w-md">
          <div className="flex flex-col items-center gap-4 py-4 text-center duration-300 animate-in fade-in-0 zoom-in-95">
            <span className="grid size-14 place-items-center rounded-full border border-highlight/40 bg-highlight/10 text-highlight shadow-[0_0_32px_-8px_var(--highlight)]">
              <MailCheck className="size-6" aria-hidden />
            </span>
            <DialogTitle className="text-xl">{t('auth_sent_title')}</DialogTitle>
            <DialogDescription className="text-base leading-relaxed text-foreground/85">
              {before}
              <strong className="break-words font-semibold text-foreground">{email.trim()}</strong>
              {after}
            </DialogDescription>
            <p className="text-sm text-muted-foreground">{t('auth_sent_hint')}</p>
            <Button type="button" className="mt-2 w-full" onClick={() => handleOpenChange(false)}>
              {t('auth_sent_done')}
            </Button>
            <button
              type="button"
              onClick={() => setSent(null)}
              className="cursor-pointer text-sm text-muted-foreground hover:text-highlight"
            >
              {t('auth_sent_back')}
            </button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{mode === 'reset' ? t('auth_reset_desc') : t('auth_desc')}</DialogDescription>
        </DialogHeader>

        {mode !== 'reset' && (
          <>
            <div className="grid gap-2">
              <Button type="button" variant="outline" disabled={busy !== null} onClick={() => void oauth('google')}>
                {busy === 'google' ? <Loader2 className="size-4 animate-spin" /> : <GoogleIcon />}
                {t('auth_google')}
              </Button>
              <Button type="button" variant="outline" disabled={busy !== null} onClick={() => void oauth('azure')}>
                {busy === 'azure' ? <Loader2 className="size-4 animate-spin" /> : <MicrosoftIcon />}
                {t('auth_microsoft')}
              </Button>
            </div>
            <div className="eyebrow flex items-center gap-3 text-muted-foreground">
              <span className="h-px flex-1 bg-border" />
              {t('auth_or')}
              <span className="h-px flex-1 bg-border" />
            </div>
          </>
        )}

        <form onSubmit={(event) => void submit(event)} className="grid gap-3">
          {mode === 'signup' && (
            <div className="grid gap-1.5">
              <Label htmlFor="auth-name">{t('auth_name')}</Label>
              <Input id="auth-name" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
          )}
          <div className="grid gap-1.5">
            <Label htmlFor="auth-email">{t('auth_email')}</Label>
            <Input
              id="auth-email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          {mode !== 'reset' && (
            <div className="grid gap-1.5">
              <div className="flex items-center justify-between">
                <Label htmlFor="auth-password">{t('auth_password')}</Label>
                {mode === 'signin' && (
                  <button
                    type="button"
                    onClick={() => switchMode('reset')}
                    className="cursor-pointer text-xs text-muted-foreground hover:text-highlight"
                  >
                    {t('auth_forgot')}
                  </button>
                )}
              </div>
              <Input
                id="auth-password"
                type="password"
                required
                minLength={mode === 'signup' ? MIN_PASSWORD : undefined}
                autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              {mode === 'signup' && <p className="text-xs text-muted-foreground">{t('auth_password_min')}</p>}
            </div>
          )}

          {(error ?? redirectError) && (
            <p role="alert" className="text-sm text-destructive">
              {error ?? redirectError}
            </p>
          )}

          <Button type="submit" disabled={busy !== null}>
            {busy === mode && <Loader2 className="size-4 animate-spin" />}
            {submitLabel}
          </Button>
        </form>

        <div className="text-center text-sm text-muted-foreground">
          {mode === 'signin' ? (
            <button type="button" onClick={() => switchMode('signup')} className="cursor-pointer hover:text-highlight">
              {t('auth_to_signup')}
            </button>
          ) : (
            <button type="button" onClick={() => switchMode('signin')} className="cursor-pointer hover:text-highlight">
              {t('auth_to_signin')}
            </button>
          )}
        </div>
        <p className="text-center text-xs text-muted-foreground">{t('auth_local_note')}</p>
      </DialogContent>
    </Dialog>
  );
}

/** Volta do link de "esqueci a senha": pede a senha nova. */
export function NewPasswordDialog() {
  const t = useLocale((state) => state.t);
  const recovering = useAuth((state) => state.recovering);
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await updatePassword(password);
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={recovering} onOpenChange={(open) => !open && useAuth.setState({ recovering: false })}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{t('auth_new_password_title')}</DialogTitle>
          <DialogDescription>{t('auth_new_password_desc')}</DialogDescription>
        </DialogHeader>
        <form onSubmit={(event) => void submit(event)} className="grid gap-3">
          <div className="grid gap-1.5">
            <Label htmlFor="auth-new-password">{t('auth_password')}</Label>
            <Input
              id="auth-new-password"
              type="password"
              required
              minLength={MIN_PASSWORD}
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">{t('auth_password_min')}</p>
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <Button type="submit" disabled={busy}>
            {busy && <Loader2 className="size-4 animate-spin" />}
            {t('auth_new_password_save')}
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function CloudStatusLine() {
  const t = useLocale((state) => state.t);
  const status = useCloudSync((state) => state.status);
  if (status === 'off') return null;
  const Icon = status === 'syncing' ? RefreshCw : status === 'error' ? CloudOff : Cloud;
  const label = { syncing: t('cloud_status_syncing'), synced: t('cloud_status_synced'), error: t('cloud_status_error') }[status];
  return (
    <p className={cn('flex items-center gap-2 text-xs', status === 'error' ? 'text-destructive' : 'text-muted-foreground')}>
      <Icon className={cn('size-3.5 shrink-0', status === 'syncing' && 'animate-spin')} aria-hidden />
      {label}
    </p>
  );
}
