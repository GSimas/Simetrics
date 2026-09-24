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
        className="group relative flex items-center gap-0 rounded-full bg-[#e56d45] p-2.5 text-xs font-bold text-ink shadow-[0_0_28px_-10px_#e56d45] transition-all duration-300 hover:scale-105 hover:gap-2 hover:shadow-[0_0_32px_-6px_#e56d45] active:scale-95 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-offset-2"
      >
        <div className="relative flex shrink-0 items-center justify-center">
          <Coffee className="size-4 sm:size-4.5 transition-transform duration-300 group-hover:-rotate-12" />
          <span className="absolute -top-1 -right-1 flex size-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-paper opacity-80" />
            <span className="relative inline-flex size-2 rounded-full bg-paper" />
          </span>
        </div>

        <span className="grid grid-cols-[0fr] transition-[grid-template-columns] duration-300 ease-out group-hover:grid-cols-[1fr]">
          <span className="flex items-center gap-1.5 overflow-hidden">
            <span className="tracking-wide whitespace-nowrap drop-shadow-xs">
              {t('buy_me_coffee')}
            </span>
            <Heart className="size-3 sm:size-3.5 shrink-0 fill-ink text-ink transition-transform duration-300 group-hover:scale-125 " />
          </span>
        </span>
      </a>
    </div>
  );
}
