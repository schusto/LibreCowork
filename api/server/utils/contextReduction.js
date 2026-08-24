'use strict';

/**
 * contextReduction.js — LIVE-path wrapper around LibreCowork's deterministic
 * context reduction algorithm.
 *
 * [cowork] The actual reduction algorithm (which turns are kept verbatim,
 * which tool tiers get deduped/truncated, and the `CWK_*` env vars that
 * configure it) now lives in `packages/api/src/utils/contextReduction.ts`
 * (`applyContextReductionCore`, exported from `@librechat/api`) so the LIVE
 * path here and the context-usage GAUGE path
 * (`packages/api/src/endpoints/projection.ts`) share exactly one
 * implementation and cannot drift apart. This file adds the pieces that are
 * specific to the live request path: the `CWK_CONTEXT_REDUCTION` on/off
 * switch, the `CWK_DUMP_CONTEXT` debug dump, and the console.log summary
 * line. See docs/18_context_reduction.md and
 * docs/22_librecowork_fork_changes.md for the full writeup, including the
 * bug this split fixed (the gauge showing an inflated, pre-reduction token
 * estimate because it had no knowledge of this algorithm at all).
 *
 * Applies two transforms to the formatted message array produced by buildMessages,
 * before it is sent to the backend.
 *
 * MESSAGE STRUCTURE (agents pipeline)
 * ------------------------------------
 * In the LibreCowork agents pipeline, tool calls and their results are NOT stored as
 * separate role:"tool" messages. They live as content parts inside assistant messages:
 *
 *   {
 *     role: 'assistant',
 *     content: [
 *       { type: 'think',     think: '...' },
 *       { type: 'text',      text: '...', tool_call_ids: ['call_abc'] },
 *       { type: 'tool_call', tool_call: { id: 'call_abc', name: 'read_file',
 *                                         args: '{"path":"/foo"}', output: '...' } },
 *       { type: 'text',      text: 'final prose answer' }
 *     ]
 *   }
 *
 * Transform 1 — strip old thinking blocks:
 *   Remove { type: 'think' } parts from assistant messages outside the keep window.
 *
 * Transform 2 — truncate/dedup old tool_call results (three independent tiers):
 *
 *   Filesystem tier (DEDUP_TOOL_NAMES):
 *     Dedup identical reads of the same path + truncate to TOOL_TRUNCATE_CHARS
 *     outside the last TOOL_TRUNCATE_TURNS user turns.
 *
 *   RAG tier (RAG_TOOL_NAMES):
 *     Knowledge-base search results stay verbatim for RAG_TRUNCATE_TURNS turns
 *     (default 5 — RAG chunks are often referenced several turns later).
 *     Outside that window, truncated to RAG_TRUNCATE_CHARS.
 *     No dedup (queries are unique).
 *
 *   Web search tier (WEB_SEARCH_TOOL_NAMES):
 *     Web results stay verbatim for WEB_SEARCH_TRUNCATE_TURNS turns (default 2).
 *     Outside that window, truncated to WEB_SEARCH_TRUNCATE_CHARS.
 *     No dedup.
 *
 * Configuration (environment variables):
 *
 *   CWK_THINK_KEEP_TURNS          Keep thinking blocks in the last N user turns (default: 2).
 *   CWK_TOOL_TRUNCATE_TURNS       Pass filesystem tool results verbatim in the last N user turns (default: 3).
 *   CWK_TOOL_TRUNCATE_CHARS       Max chars for a filesystem tool result outside the keep window (default: 2000).
 *   CWK_TOOL_DEDUP_MIN_CHARS      Never dedup results shorter than this (default: 300).
 *   CWK_RAG_TRUNCATE_TURNS        Pass RAG results verbatim in the last N user turns (default: 5).
 *   CWK_RAG_TRUNCATE_CHARS        Max chars for a RAG result outside its keep window (default: 2000).
 *   CWK_WEB_TRUNCATE_TURNS        Pass web search results verbatim in the last N user turns (default: 2).
 *   CWK_WEB_TRUNCATE_CHARS        Max chars for a web result outside its keep window (default: 1000).
 */

const fs = require('fs');
const path = require('path');
const { CONTEXT_REDUCTION_ENABLED, applyContextReductionCore } = require('@librechat/api');

// ── Configuration (live-path-only; algorithm config lives in the shared core) ──

// Debug dump. Set CWK_DUMP_CONTEXT=1 to write the raw formattedMessages array
// (before any reduction) to a timestamped JSON file in CWK_DUMP_DIR.
// One file is written per request. Clear the flag when done.
const DUMP_CONTEXT = Boolean(parseInt(process.env.CWK_DUMP_CONTEXT ?? '0', 10));
const DUMP_DIR = process.env.CWK_DUMP_DIR ?? '/app/uploads/cwk-context-dumps';

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Apply deterministic context reduction to the formatted message array produced
 * by buildMessages, before it becomes the request payload.
 *
 * Pure transform (aside from the optional debug dump below): no DB writes, no
 * network calls. The actual reduction algorithm lives in
 * `applyContextReductionCore` (`packages/api/src/utils/contextReduction.ts`,
 * exported from `@librechat/api`) — this function only adds the on/off switch,
 * debug dump, and console.log summary that are specific to the live request path.
 *
 * @param {object[]} formattedMessages  - Output of the formatMessage map in buildMessages.
 * @param {object[]} orderedMessages    - Parallel DB message array (accepted for API stability;
 *                                        unused in Phase 1 — Phase 2 will add compressed field lookup).
 * @param {string}   [conversationId]  - Optional, for log correlation.
 * @returns {object[]} Reduced message array.
 */
function applyContextReduction(formattedMessages, orderedMessages, conversationId) {
  if (!CONTEXT_REDUCTION_ENABLED) return formattedMessages;
  if (!formattedMessages || formattedMessages.length === 0) return formattedMessages;

  // ── Debug dump ──────────────────────────────────────────────────────────────
  if (DUMP_CONTEXT) {
    try {
      fs.mkdirSync(DUMP_DIR, { recursive: true });
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const conv = conversationId ? conversationId.slice(-8) : 'unknown';
      const filePath = path.join(DUMP_DIR, `context-${ts}-${conv}.json`);
      fs.writeFileSync(filePath, JSON.stringify(formattedMessages, null, 2));
      console.log(`[contextReduction] dumped ${formattedMessages.length} messages → ${filePath}`);
    } catch (err) {
      console.warn(`[contextReduction] dump failed: ${err.message}`);
    }
  }

  const { messages, stats } = applyContextReductionCore(formattedMessages);

  const anyReduction = stats.thinkStripped > 0 || stats.toolDeduped > 0 || stats.toolTruncated > 0;
  if (anyReduction) {
    const convLabel = conversationId ? `conv=${conversationId.slice(-8)} ` : '';
    console.log(
      `[contextReduction] ${convLabel}` +
      `think_stripped=${stats.thinkStripped} ` +
      `tool_deduped=${stats.toolDeduped} ` +
      `tool_truncated=${stats.toolTruncated} ` +
      `~saved=${stats.savedChars.toLocaleString()} chars`,
    );
  }

  return messages;
}

module.exports = { applyContextReduction };
