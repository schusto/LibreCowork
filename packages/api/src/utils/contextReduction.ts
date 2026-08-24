import crypto from 'crypto';
import { ContentTypes } from 'librechat-data-provider';
import type { FormattedMessageContentPart, FormattedMessageWithContent } from '~/agents/client';

/**
 * contextReduction.ts — shared core of LibreCowork's deterministic context
 * reduction algorithm.
 *
 * This is the single source of truth for the reduction algorithm itself
 * (which turns are kept verbatim, which tool tiers get deduped/truncated, and
 * by how much) and for the `CWK_*` environment variables that configure it.
 * Two callers consume it:
 *
 *   - `api/server/utils/contextReduction.js` — the LIVE path. Wraps
 *     `applyContextReductionCore` with the `CWK_CONTEXT_REDUCTION` on/off
 *     switch (via `CONTEXT_REDUCTION_ENABLED` below), the `CWK_DUMP_CONTEXT`
 *     debug dump, and a console.log summary line. Runs on every agent request
 *     in `buildMessages` (client.js) to build the actual `payload` sent to
 *     the model.
 *   - `packages/api/src/endpoints/projection.ts` — the GAUGE path. Calls
 *     `applyContextReductionCore` directly (no dump/log) so the context-usage
 *     projection shown in the UI for snapshot-less branches (page load,
 *     branch/model switch) reflects the same reduction the live path would
 *     actually apply, instead of the full unreduced conversation size.
 *
 * Keeping the algorithm and its config in exactly one place means the two
 * paths cannot silently drift apart. See docs/18_context_reduction.md for
 * the full design writeup and docs/22_librecowork_fork_changes.md for the
 * incident this fixes (summarization trigger firing at ~90% estimated usage
 * — 144k+ tokens — while only ~30k of a 256k budget was actually being sent,
 * and the context gauge showing the same inflated estimate).
 */

// ── Configuration (environment variables, read once at import) ─────────────

/**
 * Master on/off switch. Set `CWK_CONTEXT_REDUCTION=false` to disable all
 * reduction everywhere it's consumed. Any value other than 'false'
 * (case-insensitive) leaves it enabled.
 */
export const CONTEXT_REDUCTION_ENABLED: boolean =
  (process.env.CWK_CONTEXT_REDUCTION ?? 'true').toLowerCase() !== 'false';

export const THINK_KEEP_TURNS: number = parseInt(process.env.CWK_THINK_KEEP_TURNS ?? '2', 10);
export const TOOL_TRUNCATE_TURNS: number = parseInt(
  process.env.CWK_TOOL_TRUNCATE_TURNS ?? '3',
  10,
);
export const TOOL_TRUNCATE_CHARS: number = parseInt(
  process.env.CWK_TOOL_TRUNCATE_CHARS ?? '2000',
  10,
);
export const TOOL_DEDUP_MIN_CHARS: number = parseInt(
  process.env.CWK_TOOL_DEDUP_MIN_CHARS ?? '300',
  10,
);
export const RAG_TRUNCATE_TURNS: number = parseInt(process.env.CWK_RAG_TRUNCATE_TURNS ?? '5', 10);
export const RAG_TRUNCATE_CHARS: number = parseInt(
  process.env.CWK_RAG_TRUNCATE_CHARS ?? '2000',
  10,
);
export const WEB_TRUNCATE_TURNS: number = parseInt(process.env.CWK_WEB_TRUNCATE_TURNS ?? '2', 10);
export const WEB_TRUNCATE_CHARS: number = parseInt(
  process.env.CWK_WEB_TRUNCATE_CHARS ?? '1000',
  10,
);

/** Filesystem tools: candidates for deduplication + standard truncation. */
export const DEDUP_TOOL_NAMES: ReadonlySet<string> = new Set([
  'read_file',
  'read_multiple_files',
  'read_text_file_mcp_filesystem',
  'list_directory',
  'list_directory_mcp_filesystem',
  'directory_tree',
  'directory_tree_mcp_filesystem',
]);

/** RAG / knowledge-base tools: longer keep window (referenced across multiple turns). */
export const RAG_TOOL_NAMES: ReadonlySet<string> = new Set([
  'search_knowledge_base_mcp_rag',
  'search_knowledge_base',
]);

/** Web search tools: shorter keep window (surfaced results are discussed quickly). */
export const WEB_SEARCH_TOOL_NAMES: ReadonlySet<string> = new Set([
  'web_search',
  'search',
  'searxng_search',
  'brave_search',
]);

// ── Types ────────────────────────────────────────────────────────────────

/**
 * Minimal shape the reducer needs: a role plus formatted content parts
 * (matching the `{ role, content }` shape `formatMessage` produces in the
 * live path, and the same shape callers already track elsewhere in this
 * package — see `FormattedMessageWithContent` in `agents/client.ts`).
 */
export type ReducibleMessage = FormattedMessageWithContent & {
  role?: string;
};

export interface ContextReductionStats {
  thinkStripped: number;
  toolDeduped: number;
  toolTruncated: number;
  savedChars: number;
}

export interface ContextReductionResult<T> {
  messages: T[];
  stats: ContextReductionStats;
}

interface ToolCallPart {
  id?: string;
  name?: string;
  args?: string | Record<string, unknown>;
  output?: string;
}

interface ToolCallArgsShape {
  path?: string;
  file_path?: string;
  directory?: string;
  paths?: string[];
}

// ── Helpers ─────────────────────────────────────────────────────────────

/** MD5 of a string — fast enough, good enough for dedup comparison. */
function md5(text: string): string {
  return crypto.createHash('md5').update(text).digest('hex');
}

/**
 * Find the message-array index of the first message in the "keep verbatim"
 * window. The window is defined as the last `n` user messages (by role) and
 * everything after them.
 *
 * Returns -1 if there are fewer than n user messages (keep everything).
 * Returns 0 if n === 0 (strip everything).
 */
function keepBoundaryIndex(messages: ReducibleMessage[], n: number): number {
  if (n <= 0) {
    return 0;
  }

  const userIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'user') {
      userIndices.push(i);
    }
  }

  if (userIndices.length <= n) {
    return -1;
  }

  return userIndices[userIndices.length - n];
}

/**
 * Extract the path argument from a tool_call part's args, for use as a dedup
 * key. Returns '' if the tool is not in DEDUP_TOOL_NAMES or no path-like arg
 * is found.
 */
function extractPath(toolCall: ToolCallPart): string {
  const name = toolCall.name ?? '';
  if (!DEDUP_TOOL_NAMES.has(name)) {
    return '';
  }

  let args: ToolCallArgsShape = {};
  try {
    args =
      typeof toolCall.args === 'string'
        ? (JSON.parse(toolCall.args) as ToolCallArgsShape)
        : ((toolCall.args ?? {}) as ToolCallArgsShape);
  } catch {
    return '';
  }

  if (name === 'read_multiple_files') {
    const paths = args.paths;
    return Array.isArray(paths) && paths.length > 0 ? String(paths[0]) : '';
  }

  return String(args.path ?? args.file_path ?? args.directory ?? '');
}

// ── Transform 1 — strip old thinking blocks ────────────────────────────────

/**
 * Removes `{ type: 'think' }` parts from assistant messages in the strip
 * zone (message index < thinkBoundary). Mutates `messages` in place, only
 * replacing entries it actually modifies.
 */
function stripOldThinkingBlocks(messages: ReducibleMessage[], thinkBoundary: number): number {
  if (thinkBoundary < 0) {
    return 0;
  }

  let stripped = 0;
  for (let i = 0; i < thinkBoundary; i++) {
    const msg = messages[i];
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) {
      continue;
    }

    const before = msg.content.length;
    const filtered = msg.content.filter((part) => part?.type !== ContentTypes.THINK);

    if (filtered.length < before) {
      messages[i] = {
        ...msg,
        content:
          filtered.length > 0
            ? filtered
            : ([{ type: ContentTypes.TEXT, text: '' }] as FormattedMessageContentPart[]),
      };
      stripped++;
    }
  }
  return stripped;
}

// ── Transform 2 — tool result truncation and deduplication ───────────────

/**
 * Truncates and deduplicates tool_call output strings in assistant messages,
 * across three independent tiers (filesystem/dedup, RAG, web search — see
 * module docstring). Mutates `messages` in place, only replacing entries it
 * actually modifies. The dedup map is built left-to-right including the keep
 * window so older reads outside it can be correctly identified as redundant.
 */
function reduceToolResults(
  messages: ReducibleMessage[],
  toolBoundary: number,
  ragBoundary: number,
  webBoundary: number,
): { deduped: number; truncated: number; savedChars: number } {
  const stats = { deduped: 0, truncated: 0, savedChars: 0 };

  /** path → md5 of last seen output (filesystem dedup only) */
  const lastSeen = new Map<string, string>();

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) {
      continue;
    }

    const inToolWindow = toolBoundary < 0 || i >= toolBoundary;
    const inRagWindow = ragBoundary < 0 || i >= ragBoundary;
    const inWebWindow = webBoundary < 0 || i >= webBoundary;

    let msgModified = false;
    const newContent: FormattedMessageContentPart[] = msg.content.map((part) => {
      if (part?.type !== ContentTypes.TOOL_CALL || part.tool_call == null) {
        return part;
      }

      const tc = part.tool_call as ToolCallPart;
      const output = tc.output ?? '';
      const outputLen = output.length;
      const name = tc.name ?? '';
      const isRag = RAG_TOOL_NAMES.has(name);
      const isWeb = WEB_SEARCH_TOOL_NAMES.has(name);
      const isFileOp = DEDUP_TOOL_NAMES.has(name);

      // ── Filesystem tier ──────────────────────────────────────────────
      if (isFileOp) {
        const filePath = extractPath(tc);
        const isDeduplicable = filePath !== '' && outputLen >= TOOL_DEDUP_MIN_CHARS;

        if (inToolWindow) {
          // Inside keep window — update dedup map but don't rewrite.
          if (isDeduplicable) {
            lastSeen.set(filePath, md5(output));
          }
          return part;
        }

        // Strip zone: dedup first, then truncate remainder.
        if (isDeduplicable) {
          const hash = md5(output);
          if (lastSeen.has(filePath) && lastSeen.get(filePath) === hash) {
            const marker =
              `[${name}(${filePath}): identical to earlier call in this session — ` +
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

      // ── RAG tier ─────────────────────────────────────────────────────
      if (isRag) {
        if (inRagWindow) {
          return part;
        }

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

      // ── Web search tier ────────────────────────────────────────────────
      if (isWeb) {
        if (inWebWindow) {
          return part;
        }

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

      // ── Unknown tools — apply filesystem truncation if outside window ──
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

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Applies deterministic context reduction to an array of formatted messages
 * (`{ role, content }`, where `content` is either a plain string or an array
 * of parts including `{ type: 'think' }` and `{ type: 'tool_call', tool_call:
 * { name, args, output } }`).
 *
 * Pure transform: no I/O, no logging, no env-var on/off gate — callers check
 * `CONTEXT_REDUCTION_ENABLED` themselves and decide whether to invoke this at
 * all (the live path also layers on debug-dump/logging; the gauge path calls
 * this directly).
 *
 * Messages the algorithm does not touch keep their exact input object
 * reference. Callers may rely on this (`result.messages[i] !==
 * formattedMessages[i]`) to cheaply detect which messages actually changed —
 * e.g. to skip re-tokenizing the (usually large) majority of a long
 * conversation that reduction left alone.
 */
export function applyContextReductionCore<T extends ReducibleMessage>(
  formattedMessages: T[],
): ContextReductionResult<T> {
  if (!formattedMessages || formattedMessages.length === 0) {
    return {
      messages: formattedMessages,
      stats: { thinkStripped: 0, toolDeduped: 0, toolTruncated: 0, savedChars: 0 },
    };
  }

  // Shallow-copy the array; individual messages are copied on modification (see transforms).
  const messages = [...formattedMessages] as ReducibleMessage[];

  const thinkBoundary = keepBoundaryIndex(messages, THINK_KEEP_TURNS);
  const toolBoundary = keepBoundaryIndex(messages, TOOL_TRUNCATE_TURNS);
  const ragBoundary = keepBoundaryIndex(messages, RAG_TRUNCATE_TURNS);
  const webBoundary = keepBoundaryIndex(messages, WEB_TRUNCATE_TURNS);

  const thinkStripped = stripOldThinkingBlocks(messages, thinkBoundary);
  const toolStats = reduceToolResults(messages, toolBoundary, ragBoundary, webBoundary);

  return {
    messages: messages as T[],
    stats: {
      thinkStripped,
      toolDeduped: toolStats.deduped,
      toolTruncated: toolStats.truncated,
      savedChars: toolStats.savedChars,
    },
  };
}
