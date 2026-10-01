import type { CSSProperties } from 'react';

import { cn } from '@/lib/utils';

/**
 * Gerador pseudoaleatório com semente fixa: as partículas ficam espalhadas de forma
 * irregular, mas iguais a cada render (Math.random mudaria tudo a cada atualização).
 */
function seeded(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

interface Particle {
  x: number;
  y: number;
  r: number;
  duration: number;
  delay: number;
  dx: number;
  dy: number;
  tone: 'muted' | 'signal' | 'cyan';
}

const PARTICLES: Particle[] = (() => {
  const random = seeded(7);
  return Array.from({ length: 28 }, (_, index) => ({
    x: 20 + random() * 1160,
    y: 20 + random() * 600,
    r: 1.1 + random() * 1.8,
    duration: 5 + random() * 7,
    delay: random() * 8,
    dx: (random() - 0.5) * 36,
    dy: -10 - random() * 26,
    tone: index % 6 === 0 ? 'signal' : index % 7 === 0 ? 'cyan' : 'muted',
  }));
})();

const PARTICLE_FILL = { muted: 'currentColor', signal: 'var(--highlight)', cyan: 'var(--cyan)' };

/** Constelação: uma pequena rede — nós e arestas — no canto oposto às órbitas. */
const NODES = [
  [120, 430],
  [210, 372],
  [300, 452],
  [255, 540],
  [390, 395],
  [150, 560],
  [440, 505],
] as const;
const EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [1, 4],
  [2, 4],
  [0, 5],
  [3, 5],
  [4, 6],
  [2, 6],
] as const;

/**
 * Órbitas, mira, constelação e partículas do hero, no traço fino do Scientata: o
 * movimento de fundo da tela inicial. Animações em index.css, paradas com movimento
 * reduzido.
 */
export function HeroOrbits({ className }: { className?: string }) {
  return (
    <svg
      className={cn('pointer-events-none absolute inset-0 hidden h-full w-full text-border md:block', className)}
      viewBox="0 0 1200 640"
      preserveAspectRatio="xMidYMid slice"
      fill="none"
      aria-hidden
    >
      {/* Diagonal com um feixe de luz percorrendo-a. */}
      <line x1="0" y1="560" x2="1200" y2="80" stroke="currentColor" />
      <line
        className="beam"
        x1="0"
        y1="560"
        x2="1200"
        y2="80"
        stroke="var(--highlight)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Mira com pulso de radar. */}
      <path d="M930 190v120M870 250h120" stroke="currentColor" />
      <circle className="radar-ping" cx="930" cy="250" r="24" stroke="var(--highlight)" />
      <circle className="radar-ping radar-ping--late" cx="930" cy="250" r="24" stroke="var(--cyan)" />

      <g className="orbit-spin orbit-spin--outer">
        <circle cx="930" cy="250" r="260" stroke="currentColor" />
        <circle cx="930" cy="-10" r="4.5" fill="var(--highlight)" />
        <circle cx="930" cy="510" r="2.5" fill="currentColor" />
      </g>
      <g className="orbit-spin orbit-spin--middle">
        <circle cx="930" cy="250" r="205" stroke="currentColor" strokeDasharray="2 10" />
        <circle cx="725" cy="250" r="3" fill="var(--cyan)" />
      </g>
      <g className="orbit-spin orbit-spin--inner">
        <circle cx="930" cy="250" r="150" stroke="currentColor" />
        <circle cx="1080" cy="250" r="3.5" fill="var(--highlight)" />
      </g>

      {/* Constelação com fluxo nas arestas. */}
      {EDGES.map(([from, to], index) => (
        <line
          key={`${from}-${to}`}
          className="edge-flow"
          x1={NODES[from]![0]}
          y1={NODES[from]![1]}
          x2={NODES[to]![0]}
          y2={NODES[to]![1]}
          stroke="currentColor"
          style={{ animationDelay: `-${index * 0.7}s` }}
        />
      ))}
      {NODES.map(([cx, cy], index) => (
        <circle
          key={`${cx}-${cy}`}
          className="node-pulse"
          cx={cx}
          cy={cy}
          r={index % 3 === 0 ? 4 : 3}
          fill={index % 3 === 0 ? 'var(--highlight)' : 'currentColor'}
          style={{ animationDelay: `-${index * 0.9}s` }}
        />
      ))}

      {PARTICLES.map((particle) => (
        <circle
          key={`${particle.x.toFixed(1)}-${particle.y.toFixed(1)}`}
          className="particle-float"
          cx={particle.x}
          cy={particle.y}
          r={particle.r}
          fill={PARTICLE_FILL[particle.tone]}
          style={
            {
              animationDuration: `${particle.duration.toFixed(2)}s`,
              animationDelay: `-${particle.delay.toFixed(2)}s`,
              '--dx': `${particle.dx.toFixed(1)}px`,
              '--dy': `${particle.dy.toFixed(1)}px`,
            } as CSSProperties
          }
        />
      ))}
    </svg>
  );
}

/**
 * Fundo vivo da tela inicial: a grade milimetrada desliza e três brilhos difusos
 * derivam. Fica fixo atrás de todo o conteúdo e cobre a grade estática do body.
 */
export function AmbientBackground() {
  return (
    <div className="landing-ambient" aria-hidden>
      <div className="ambient-glow ambient-glow--a" />
      <div className="ambient-glow ambient-glow--b" />
      <div className="ambient-glow ambient-glow--c" />
    </div>
  );
}

/**
 * O fundo da tela inicial atrás de todas as telas do workspace, desfocado: a grade, os
 * brilhos e as órbitas continuam se movendo, mas como luz ao fundo, sem disputar a leitura
 * com gráficos e tabelas.
 */
export function WorkspaceBackdrop() {
  return (
    <div className="workspace-backdrop" aria-hidden>
      <AmbientBackground />
      {/* Na tela inteira, e não só no topo: o movimento acompanha a rolagem. */}
      <HeroOrbits className="fixed" />
    </div>
  );
}
