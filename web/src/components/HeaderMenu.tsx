import { Suspense, useEffect, useRef, useState, type ReactNode } from 'react';
import { ArrowLeft, ChevronDown, ChevronRight, FolderOpen, HelpCircle, LogIn, LogOut, Menu, Settings, UserRound } from 'lucide-react';

import type { User } from '@supabase/supabase-js';

import { AuthDialog, CloudStatusLine, NewPasswordDialog } from '@/components/AccountDialogs';
import { SettingsPanel } from '@/components/HeaderActions';
import { ProfileDialog } from '@/components/ProfileDialog';
import { lazyWithPreload } from '@/lib/lazy';
import { cn } from '@/lib/utils';
import { signOut, useAuth } from '@/state/auth.store';
import { useLocale } from '@/state/locale.store';

// Só baixado quando alguém abre as configurações de IA.
const AiSettingsModal = lazyWithPreload(() =>
  import('@/components/AiSettingsModal').then((module) => ({ default: module.AiSettingsModal })),
);

/** Iniciais do nome (ou do e-mail) para o botão do menu. */
function initialsOf(user: User): string {
  const name = (user.user_metadata?.full_name as string | undefined) || user.email || '';
  return name
    .split(/[\s@.]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]!.toUpperCase())
    .join('');
}

/** Linha do menu: ícone, rótulo e, quando abre outra tela do menu, uma seta. */
function MenuItem({
  icon,
  label,
  onClick,
  trailing,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  trailing?: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      className="flex w-full cursor-pointer items-center gap-3 px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent/50 hover:text-highlight focus-visible:bg-accent/50 focus-visible:outline-none"
    >
      <span className="text-muted-foreground" aria-hidden>
        {icon}
      </span>
      <span className="flex-1">{label}</span>
      {trailing}
    </button>
  );
}

/**
 * Menu único do cabeçalho: conta (entrar, ou nome, estado da nuvem e sair), Meus Projetos,
 * tutorial e configurações, que abrem dentro do próprio menu. Os itens de projetos e
 * tutorial só aparecem onde a tela passa a ação.
 */
export function HeaderMenu({
  onOpenProjects,
  onOpenTutorial,
}: {
  onOpenProjects?: () => void;
  onOpenTutorial?: () => void;
}) {
  const t = useLocale((state) => state.t);
  const user = useAuth((state) => state.user);
  const ready = useAuth((state) => state.ready);
  const redirectError = useAuth((state) => state.redirectError);

  const [open, setOpen] = useState(false);
  const [view, setView] = useState<'main' | 'settings'>('main');
  const [authOpen, setAuthOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  // Montado desde a primeira abertura, para a animação de fechar continuar funcionando.
  const [aiMounted, setAiMounted] = useState(false);

  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const close = (): void => {
    setOpen(false);
    setView('main');
  };

  // Fecha com clique fora ou Esc (e devolve o foco ao botão).
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) close();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      close();
      triggerRef.current?.focus();
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const run = (action: () => void) => () => {
    close();
    action();
  };

  const name = (user?.user_metadata?.full_name as string | undefined) || '';
  const initials = user ? initialsOf(user) : '';

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        data-tour="menu"
        onClick={() => (open ? close() : setOpen(true))}
        className={cn('header-chip cursor-pointer gap-2', open && 'border-highlight text-highlight')}
        aria-label={t('menu_btn')}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {initials ? (
          <span className="grid size-6 place-items-center rounded-full bg-primary text-[0.625rem] font-semibold text-primary-foreground">
            {initials}
          </span>
        ) : (
          <Menu className="size-3.5" aria-hidden />
        )}
        <span className="hidden sm:inline">{t('menu_btn')}</span>
        <ChevronDown className={cn('size-3.5 transition-transform duration-200', open && 'rotate-180')} aria-hidden />
      </button>

      {open && (
        <div
          role="menu"
          aria-label={view === 'settings' ? t('settings_title') : t('menu_btn')}
          className="absolute right-0 top-full z-50 mt-2 max-h-[calc(100dvh-6rem)] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto border border-border bg-card text-left shadow-xl duration-200 animate-in fade-in-0 slide-in-from-top-1"
        >
          {view === 'main' ? (
            <>
              <div className="space-y-2 border-b border-border p-4">
                {user ? (
                  <>
                    <div className="min-w-0">
                      {name && <p className="truncate text-sm font-semibold">{name}</p>}
                      <p className="truncate text-xs text-muted-foreground">{user.email}</p>
                    </div>
                    <CloudStatusLine />
                  </>
                ) : (
                  ready && (
                    <button
                      type="button"
                      role="menuitem"
                      onClick={run(() => setAuthOpen(true))}
                      className="flex w-full cursor-pointer items-center gap-3 text-left"
                    >
                      <span className="grid size-8 place-items-center rounded-full border border-highlight/40 bg-highlight/10 text-highlight">
                        <LogIn className="size-4" aria-hidden />
                      </span>
                      <span>
                        <span className="block text-sm font-semibold">{t('auth_sign_in_btn')}</span>
                        <span className="block text-xs text-muted-foreground">{t('menu_sign_in_hint')}</span>
                      </span>
                    </button>
                  )
                )}
              </div>

              <div className="py-1">
                {user && (
                  <MenuItem icon={<UserRound className="size-4" />} label={t('profile_title')} onClick={run(() => setProfileOpen(true))} />
                )}
                {onOpenProjects && (
                  <MenuItem icon={<FolderOpen className="size-4" />} label={t('nav_projects_btn')} onClick={run(onOpenProjects)} />
                )}
                {onOpenTutorial && (
                  <MenuItem icon={<HelpCircle className="size-4" />} label={t('tutorial_btn')} onClick={run(onOpenTutorial)} />
                )}
                <MenuItem
                  icon={<Settings className="size-4" />}
                  label={t('settings_title')}
                  onClick={() => setView('settings')}
                  trailing={<ChevronRight className="size-4 text-muted-foreground" aria-hidden />}
                />
              </div>

              {user && (
                <div className="border-t border-border py-1">
                  <MenuItem icon={<LogOut className="size-4" />} label={t('auth_sign_out')} onClick={run(() => void signOut())} />
                </div>
              )}
            </>
          ) : (
            <div className="p-4">
              <button
                type="button"
                onClick={() => setView('main')}
                className="mb-4 flex cursor-pointer items-center gap-2 text-base font-semibold transition-colors hover:text-highlight"
              >
                <ArrowLeft className="size-4" aria-hidden />
                {t('settings_title')}
              </button>
              <SettingsPanel
                onPreloadAi={() => void AiSettingsModal.preload()}
                onConfigureAi={run(() => {
                  setAiMounted(true);
                  setAiOpen(true);
                })}
              />
            </div>
          )}
        </div>
      )}

      <AuthDialog
        // Erro na volta do Google/Microsoft: o login reabre mostrando o motivo.
        open={!user && (authOpen || Boolean(redirectError))}
        onOpenChange={(next) => {
          setAuthOpen(next);
          if (!next) useAuth.setState({ redirectError: null });
        }}
      />
      <NewPasswordDialog />
      {/* Montado só aberto: os campos nascem com os dados atuais da conta. */}
      {user && profileOpen && <ProfileDialog open onOpenChange={setProfileOpen} />}
      {aiMounted && (
        <Suspense fallback={null}>
          <AiSettingsModal open={aiOpen} onOpenChange={setAiOpen} />
        </Suspense>
      )}
    </div>
  );
}
