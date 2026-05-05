import { useState, useEffect, useRef } from 'react';

interface PrefillStatus {
  phase: 'prefilling' | 'idle';
  done?: number;
  total?: number;
  pct?: number;
  input_tokens?: number;
}

interface ProgressSample {
  pct: number;
  ts: number; // performance.now()
}

const POLL_INTERVAL_MS = 300;
const PROXY_URL = '/api/prefill_status';
// Use up to the last N samples for the speed estimate
const MAX_SAMPLES = 8;

type Props = {
  isLatestMessage: boolean;
};

/**
 * Shows prompt prefill progress while mlx-proxy is processing the input tokens.
 * Polls GET /prefill_status every 300ms whenever this is the latest message.
 * Estimates remaining time from recent progress samples and counts it down live.
 * Disappears as soon as the proxy reports phase=idle (first token generated).
 */
export default function PrefillProgress({ isLatestMessage }: Props) {
  const [status, setStatus] = useState<PrefillStatus | null>(null);
  // Remaining seconds — updated by the estimation + live countdown ticker
  const [secsLeft, setSecsLeft] = useState<number | null>(null);

  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const samplesRef = useRef<ProgressSample[]>([]);
  // When the current ETA was last computed (performance.now)
  const etaComputedAtRef = useRef<number>(0);
  // The raw estimated seconds at the time of last computation
  const etaRawRef = useRef<number | null>(null);

  // Compute ETA from recent samples; returns seconds or null if not enough data
  const computeEta = (currentPct: number): number | null => {
    const samples = samplesRef.current;
    if (samples.length < 2) return null;
    const oldest = samples[0];
    const newest = samples[samples.length - 1];
    const deltaPct = newest.pct - oldest.pct;
    const deltaMs = newest.ts - oldest.ts;
    if (deltaPct <= 0 || deltaMs <= 0) return null;
    const pctPerMs = deltaPct / deltaMs;
    const remainingPct = 100 - currentPct;
    return (remainingPct / pctPerMs) / 1000; // seconds
  };

  useEffect(() => {
    if (!isLatestMessage) {
      setStatus(null);
      setSecsLeft(null);
      samplesRef.current = [];
      return;
    }

    const poll = async () => {
      try {
        const res = await fetch(PROXY_URL, { cache: 'no-store' });
        const data: PrefillStatus = await res.json();
        if (data.phase !== 'prefilling') {
          setStatus(null);
          setSecsLeft(null);
          samplesRef.current = [];
          return;
        }

        setStatus(data);

        // Derive a 0–100 percentage for ETA estimation
        let currentPct: number | null = null;
        if (data.done != null && data.total != null && data.total > 0) {
          currentPct = (data.done / data.total) * 100;
        } else if (data.pct != null) {
          currentPct = data.pct;
        }

        if (currentPct != null) {
          const now = performance.now();
          const samples = samplesRef.current;
          // Only add sample if pct actually moved
          if (samples.length === 0 || currentPct !== samples[samples.length - 1].pct) {
            samples.push({ pct: currentPct, ts: now });
            if (samples.length > MAX_SAMPLES) samples.shift();
          }
          const eta = computeEta(currentPct);
          if (eta != null) {
            etaRawRef.current = eta;
            etaComputedAtRef.current = now;
            setSecsLeft(Math.ceil(eta));
          }
        }
      } catch {
        // ignore — keep showing last known state
      }
    };

    samplesRef.current = [];
    poll();
    pollTimerRef.current = setInterval(poll, POLL_INTERVAL_MS);

    // Live countdown: tick every second, decrement based on elapsed time since last ETA compute
    tickTimerRef.current = setInterval(() => {
      if (etaRawRef.current == null) return;
      const elapsedSecs = (performance.now() - etaComputedAtRef.current) / 1000;
      const remaining = etaRawRef.current - elapsedSecs;
      setSecsLeft(remaining > 0 ? Math.ceil(remaining) : 0);
    }, 1000);

    return () => {
      if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
      if (tickTimerRef.current) { clearInterval(tickTimerRef.current); tickTimerRef.current = null; }
    };
  }, [isLatestMessage]);

  if (!status) return null;

  const hasDeterminate = status.done != null && status.total != null && status.total > 0;
  const pct = hasDeterminate
    ? (status.done! / status.total!) * 100
    : status.pct ?? null;
  const pctRounded = pct != null ? Math.round(pct) : null;
  const tokenLabel = status.input_tokens != null
    ? status.input_tokens.toLocaleString() + ' tok'
    : null;

  // Build the right-hand label
  let rightLabel = '';
  if (hasDeterminate) {
    rightLabel = `${status.done!.toLocaleString()} / ${status.total!.toLocaleString()} tokens`;
    if (secsLeft != null) rightLabel += ` · ~${secsLeft}s`;
  } else if (pctRounded != null) {
    rightLabel = tokenLabel
      ? `${pctRounded}% (of ${tokenLabel})`
      : `${pctRounded}%`;
    if (secsLeft != null) rightLabel += ` · ~${secsLeft}s`;
  } else if (tokenLabel) {
    rightLabel = tokenLabel;
  }

  return (
    <div className="my-1 rounded-md border border-border-light bg-surface-secondary px-3 py-2 text-xs text-text-secondary">
      <div className="mb-1 flex items-center justify-between">
        <span className="font-medium">Processing prompt…</span>
        <span className="tabular-nums">{rightLabel}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-border-light">
        {pct != null ? (
          <div
            className="h-full rounded-full bg-text-secondary transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full w-full rounded-full bg-text-secondary opacity-40 animate-pulse" />
        )}
      </div>
    </div>
  );
}
