import { atom } from 'recoil';

/**
 * Prefill progress emitted by mlx-proxy as SSE events during prompt processing.
 *
 * mlx-lm format:   { phase: 'prefilling', done: 4096, total: 9385 }
 * LM Studio format: { phase: 'prefilling', pct: 44.3 }
 * Completion:       { phase: 'done' }
 */
export interface MlxPrefillStatus {
  phase: 'prefilling' | 'done';
  /** Tokens processed so far (mlx-lm only) */
  done?: number;
  /** Total prompt tokens (mlx-lm only) */
  total?: number;
  /** Percentage 0–100 (LM Studio only) */
  pct?: number;
}

/**
 * Keyed by conversationId. Cleared when phase becomes 'done'
 * or when the SSE stream closes.
 */
export const mlxPrefillStatusState = atom<Record<string, MlxPrefillStatus>>({
  key: 'mlxPrefillStatus',
  default: {},
});
