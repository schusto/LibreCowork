'use strict';

/**
 * contextReduction.js — deterministic context reduction for LibreCowork
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

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// ── Configuration ─────────────────────────────────────────────────────────────

// Master on/off switch. Set CWK_CONTEXT_REDUCTION=false in docker-compose.cowork.yml
// to disable all reduction and pass the raw message array straight through.
// Any value other than 'false' (case-insensitive) leaves it enabled.
const CONTEXT_REDUCTION_ENABLED =
  (process.env.CWK_CONTEXT_REDUCTION ?? 'true').toLowerCase() !== 'false';

// Debug dump. Set CWK_DUMP_CONTEXT=1 to write the raw formattedMessages array
// (before any reduction) to a timestamped JSON file in CWK_DUMP_DIR.
// One file is written per request. Clear the flag when done.
const DUMP_CONTEXT = Boolean(parseInt(process.env.CWK_DUMP_CONTEXT ?? '0', 10));
const DUMP_DIR = process.env.CWK_DUMP_DIR ?? '/app/uploads/cwk-context-dumps';

const THINK_KEEP_TURNS        = parseInt(process.env.CWK_THINK_KEEP_TURNS        ?? '2',    10);
const TOOL_TRUNCATE_TURNS     = parseInt(process.env.CWK_TOOL_TRUNCATE_TURNS     ?? '3',    10);
const TOOL_TRUNCATE_CHARS     = parseInt(process.env.CWK_TOOL_TRUNCATE_CHARS     ?? '2000', 10);
const TOOL_DEDUP_MIN_CHARS    = parseInt(process.env.CWK_TOOL_DEDUP_MIN_CHARS    ?? '300',  10);
const RAG_TRUNCATE_TURNS      = parseInt(process.env.CWK_RAG_TRUNCATE_TURNS      ?? '5',    10);
const RAG_TRUNCATE_CHARS      = parseInt(process.env.CWK_RAG_TRUNCATE_CHARS      ?? '2000', 10);
const WEB_TRUNCATE_TURNS      = parseInt(process.env.CWK_WEB_TRUNCATE_TURNS      ?? '2',    10);
const WEB_TRUNCATE_CHARS      = parseInt(process.env.CWK_WEB_TRUNCATE_CHARS      ?? '1000', 10);

// Filesystem tools: candidates for deduplication + standard truncation.
const DEDUP_TOOL_NAMES = new Set([
  'read_file',
  'read_multiple_files',
  'read_text_file_mcp_filesystem',
  'list_directory',
  'list_directory_mcp_filesystem',
  'directory_tree',
  'directory_tree_mcp_filesystem',
]);

// RAG / knowledge-base tools: longer keep window (referenced across multiple turns).
const RAG_TOOL_NAMES = new Set([
  'search_knowledge_base_mcp_rag',
  'search_knowledge_base',
]);

// Web search tools: shorter keep window (surfaced results are discussed quickly).
const WEB_SEARCH_TOOL_NAMES = new Set([
  'web_search',
  'search',
  'searxng_search',
  'brave_search',
]);

// ── Helpers ───────────────────────────────────────────────────────────────────

/** MD5 of a string — fast enough, good enough for dedup comparison. */
function md5(text) {
  return crypto.createHash('md5').update(text).digest('hex');
}

/**
 * Find the message-array index of the first message in the "keep verbatim" window.
 * The window is defined as the last `n` user messages (by role) and everything after them.
 *
 * Returns -1 if there are fewer than n user messages (keep everything).
 * Returns 0 if n === 0 (strip everything).
 *
 * @param {object[]} messages
 * @param {number}   n
 * @returns {number}
 */
function keepBoundaryIndex(messages, n) {
  if (n <= 0) return 0;

  const userIndices = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'user') userIndices.push(i);
  }

  if (userIndices.length <= n) return -1; // not enough turns — keep everything

  return userIndices[userIndices.length - n];
}

/**
 * Extract the path argument from a tool_call part's args, for use as a dedup key.
 * Returns '' if the tool is not in DEDUP_TOOL_NAMES or no path-like arg is found.
 *
 * @param {{ name: string, args: string|object }} toolCall
 * @returns {string}
 */
function extractPath(toolCall) {
  const { name } = toolCall;
  if (!DEDUP_TOOL_NAMES.has(name)) return '';

  let args = {};
  try {
    args = typeof toolCall.args === 'string' ? JSON.parse(toolCall.args) : (toolCall.args ?? {});
  } catch {
    return '';
  }

  if (name === 'read_multiple_files') {
    const paths = args.paths;
    return Array.isArray(paths) && paths.length > 0 ? String(paths[0]) : '';
  }

  return String(args.path ?? args.file_path ?? args.directory ?? '');
}

// ── Transform 1 — Strip old thinking blocks ───────────────────────────────────

/**
 * Remove { type: 'think' } parts from assistant messages in the strip zone
 * (message index < thinkBoundary).
 *
 * @param {object[]} messages  Shallow-copy of formattedMessages (will be mutated).
 * @param {number}   thinkBoundary
 * @returns {number} Count of assistant messages modified.
 */
function stripOldThinkingBlocks(messages, thinkBoundary) {
  if (thinkBoundary < 0) return 0;

  let stripped = 0;
  for (let i = 0; i < thinkBoundary; i++) {
    const msg = messages[i];
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;

    const before = msg.content.length;
    const filtered = msg.content.filter((part) => part?.type !== 'think');

    if (filtered.length < before) {
      messages[i] = {
        ...msg,
        content: filtered.length > 0 ? filtered : [{ type: 'text', text: '' }],
      };
      stripped++;
    }
  }
  return stripped;
}

// ── Transform 2 — Tool result truncation and deduplication ───────────────────

/**
 * Truncate and deduplicate tool_call output strings in assistant messages.
 *
 * Three independent tiers:
 *
 *   Filesystem tier (DEDUP_TOOL_NAMES):
 *     - Dedup identical reads of the same path across the conversation.
 *     - Truncate to TOOL_TRUNCATE_CHARS outside the last TOOL_TRUNCATE_TURNS user turns.
 *
 *   RAG tier (RAG_TOOL_NAMES):
 *     - No dedup (queries are always unique).
 *     - Truncate to RAG_TRUNCATE_CHARS outside the last RAG_TRUNCATE_TURNS user turns.
 *     - Default window is 5 turns — RAG chunks are often referenced several turns later.
 *
 *   Web search tier (WEB_SEARCH_TOOL_NAMES):
 *     - No dedup.
 *     - Truncate to WEB_TRUNCATE_CHARS outside the last WEB_TRUNCATE_TURNS user turns.
 *     - Default window is 2 turns — web results are discussed and moved on from quickly.
 *
 * The dedup map is built left-to-right including the keep window so we correctly
 * identify which older reads are redundant.
 *
 * @param {object[]} messages
 * @param {number}   toolBoundary    First index of filesystem keep window (-1 = keep all).
 * @param {number}   ragBoundary     First index of RAG keep window (-1 = keep all).
 * @param {number}   webBoundary     First index of web search keep window (-1 = keep all).
 * @returns {{ deduped: number, truncated: number, savedChars: number }}
 */
function reduceToolResults(messages, toolBoundary, ragBoundary, webBoundary) {
  const stats = { deduped: 0, truncated: 0, savedChars: 0 };

  /** path → md5 of last seen output (filesystem dedup only) */
  const lastSeen = new Map();

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;

    const inToolWindow = toolBoundary < 0 || i >= toolBoundary;
    const inRagWindow  = ragBoundary  < 0 || i >= ragBoundary;
    const inWebWindow  = webBoundary  < 0 || i >= webBoundary;

    let msgModified = false;
    const newContent = msg.content.map((part) => {
      if (part?.type !== 'tool_call' || !part.tool_call) return part;

      const tc = part.tool_call;
      const output = tc.output ?? '';
      const outputLen = output.length;
      const isRag    = RAG_TOOL_NAMES.has(tc.name);
      const isWeb    = WEB_SEARCH_TOOL_NAMES.has(tc.name);
      const isFileOp = DEDUP_TOOL_NAMES.has(tc.name);

      // ── Filesystem tier ──────────────────────────────────────────────────
      if (isFileOp) {
        const filePath = extractPath(tc);
        const isDeduplicable = filePath && outputLen >= TOOL_DEDUP_MIN_CHARS;

        if (inToolWindow) {
          // Inside keep window — update dedup map but don't rewrite.
          if (isDeduplicable) lastSeen.set(filePath, md5(output));
          return part;
        }

        // Strip zone: dedup first, then truncate remainder.
        if (isDeduplicable) {
          const hash = md5(output);
          if (lastSeen.has(filePath) && lastSeen.get(filePath) === hash) {
            const marker =
              `[${tc.name}(${filePath}): identical to earlier call in this session — ` +
              `output omitted to reduce context size. Refer to the previous result above.]`;
            msgModified = true;
            stats.deduped++;
            stats.savedChars += outputLen - marker.length;
            return { ...part, tool_call: { ...tc, output: marker } };
          }
          lastSeen.set(filePath, hash);
        }

        if (outputLen > TOOL_TRUNCATE_CHARS) {
          const omitted = outputLen - TOOL_TRUNCATE_CHARS;
          const truncated =
            output.slice(0, TOOL_TRUNCATE_CHARS) +
            `\n[... ${omitted.toLocaleString()} chars omitted — truncated by context reducer]`;
          msgModified = true;
          stats.truncated++;
          stats.savedChars += omitted;
          return { ...part, tool_call: { ...tc, output: truncated } };
        }

        return part;
      }

      // ── RAG tier ─────────────────────────────────────────────────────────
      if (isRag) {
        if (inRagWindow) return part; // keep verbatim

        if (outputLen > RAG_TRUNCATE_CHARS) {
          const omitted = outputLen - RAG_TRUNCATE_CHARS;
          const truncated =
            output.slice(0, RAG_TRUNCATE_CHARS) +
            `\n[... ${omitted.toLocaleString()} chars omitted — RAG result truncated by context reducer]`;
          msgModified = true;
          stats.truncated++;
          stats.savedChars += omitted;
          return { ...part, tool_call: { ...tc, output: truncated } };
        }

        return part;
      }

      // ── Web search tier ───────────────────────────────────────────────────
      if (isWeb) {
        if (inWebWindow) return part; // keep verbatim

        if (outputLen > WEB_TRUNCATE_CHARS) {
          const omitted = outputLen - WEB_TRUNCATE_CHARS;
          const truncated =
            output.slice(0, WEB_TRUNCATE_CHARS) +
            `\n[... ${omitted.toLocaleString()} chars omitted — web search result truncated by context reducer]`;
          msgModified = true;
          stats.truncated++;
          stats.savedChars += omitted;
          return { ...part, tool_call: { ...tc, output: truncated } };
        }

        return part;
      }

      // ── Unknown tools — apply filesystem truncation if outside window ────
      if (!inToolWindow && outputLen > TOOL_TRUNCATE_CHARS) {
        const omitted = outputLen - TOOL_TRUNCATE_CHARS;
        const truncated =
          output.slice(0, TOOL_TRUNCATE_CHARS) +
          `\n[... ${omitted.toLocaleString()} chars omitted — truncated by context reducer]`;
        msgModified = true;
        stats.truncated++;
        stats.savedChars += omitted;
        return { ...part, tool_call: { ...tc, output: truncated } };
      }

      return part;
    });

    if (msgModified) {
      messages[i] = { ...msg, content: newContent };
    }
  }

  return stats;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Apply deterministic context reduction to the formatted message array produced
 * by buildMessages, before it becomes the request payload.
 *
 * Pure transform: no DB reads/writes, no network calls, no side effects.
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

  // Shallow-copy the array; individual messages are copied on modification (see transforms).
  const messages = [...formattedMessages];

  const thinkBoundary = keepBoundaryIndex(messages, THINK_KEEP_TURNS);
  const toolBoundary  = keepBoundaryIndex(messages, TOOL_TRUNCATE_TURNS);
  const ragBoundary   = keepBoundaryIndex(messages, RAG_TRUNCATE_TURNS);
  const webBoundary   = keepBoundaryIndex(messages, WEB_TRUNCATE_TURNS);

  const thinkStripped = stripOldThinkingBlocks(messages, thinkBoundary);
  const toolStats = reduceToolResults(messages, toolBoundary, ragBoundary, webBoundary);

  const anyReduction = thinkStripped > 0 || toolStats.deduped > 0 || toolStats.truncated > 0;
  if (anyReduction) {
    const convLabel = conversationId ? `conv=${conversationId.slice(-8)} ` : '';
    console.log(
      `[contextReduction] ${convLabel}` +
      `think_stripped=${thinkStripped} ` +
      `tool_deduped=${toolStats.deduped} ` +
      `tool_truncated=${toolStats.truncated} ` +
      `~saved=${toolStats.savedChars.toLocaleString()} chars`,
    );
  }

  return messages;
}

module.exports = { applyContextReduction };
