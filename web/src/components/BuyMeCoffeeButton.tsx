import { Coffee, Heart } from 'lucide-react';
import { useLocale } from '@/state/locale.store';

export function BuyMeCoffeeButton() {
  const t = useLocale((state) => state.t);

  return (
    <div className="fixed bottom-4 right-5 z-50 flex items-center justify-end">
      <a
        href="https://link.mercadopago.com.br/strangerhits"
        target="_blank"
        rel="noopener noreferrer"
        title={t('buy_me_coffee_tooltip')}
        aria-label={t('buy_me_coffee')}
        className="group relative flex items-center gap-2 rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-amber-600 px-3.5 py-2 sm:px-4 sm:py-2 text-xs font-bold text-white shadow-xl shadow-amber-500/30 ring-2 ring-amber-400/60 transition-all duration-300 hover:scale-105 hover:shadow-amber-500/55 hover:ring-amber-300 active:scale-95 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-offset-2"
      >
        <div className="relative flex items-center justify-center">
          <Coffee className="size-4 sm:size-4.5 transition-transform duration-300 group-hover:-rotate-12" />
          <span className="absolute -top-1 -right-1 flex size-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-yellow-200 opacity-80" />
            <span className="relative inline-flex size-2 rounded-full bg-yellow-300 shadow-xs" />
          </span>
        </div>

        <span className="tracking-wide drop-shadow-xs">
          {t('buy_me_coffee')}
        </span>

        <Heart className="size-3 sm:size-3.5 fill-rose-100 text-rose-100 transition-transform duration-300 group-hover:scale-125 group-hover:fill-rose-300" />
      </a>
    </div>
  );
}
