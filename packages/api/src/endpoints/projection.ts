import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { Providers, createTokenCounter, projectAgentContextUsage } from '@librechat/agents';
import type { TContextProjectionRequest, TContextUsageEvent } from 'librechat-data-provider';
import type { BaseMessage } from '@langchain/core/messages';
import { QUOTE_MAX_COUNT, mergeQuotedText } from '~/utils/quotes';
import { CONTEXT_REDUCTION_ENABLED, applyContextReductionCore } from '~/utils/contextReduction';
import type { ReducibleMessage } from '~/utils/contextReduction';
import { countFormattedMessageTokens } from '~/agents/client';
import type { FormattedMessageContentPart } from '~/agents/client';

const MAX_PROJECTION_MESSAGES = 512;
const MAX_PROJECTION_BRANCH_MESSAGES = 256;
const MAX_PROJECTION_BRANCH_TEXT_BYTES = 512 * 1024;
const PROJECTION_GRAPH_SELECT = 'messageId parentMessageId metadata.summaryUsedTokens';
/**
 * [cowork] `content` is included so the gauge can run the same deterministic
 * context reduction (`applyContextReductionCore`) the live path applies —
 * without it, this endpoint only ever sees `text` and reconstructs plain
 * HumanMessage/AIMessage turns, so a branch with heavy tool_call output would
 * project a token count far higher than what `buildMessages` actually sends.
 * See docs/18_context_reduction.md.
 */
const PROJECTION_BODY_SELECT =
  'messageId parentMessageId tokenCount isCreatedByUser text quotes content';

interface ProjectionMessage {
  messageId: string;
  parentMessageId?: string | null;
  tokenCount?: number;
  isCreatedByUser?: boolean;
  text?: string;
  /** Quoted excerpts merged into the model-facing text by the live path; must be
   *  included here so the context gauge counts the same prompt the model sees. */
  quotes?: string[];
  /**
   * [cowork] Raw content-part array as stored by the agents pipeline (think /
   * tool_call / text parts) — the same shape `formatMessage` produces on the
   * live path. Absent for plain text-only turns. Used only for the context
   * reduction + recount pass below; the LangChain messages built for
   * `projectAgentContextUsage` still use the flattened `text`.
   */
  content?: FormattedMessageContentPart[] | string;
  /** Compaction marker written by the live path (`agents/usage.ts`); its
   *  presence means the next call sends the summary + tail, not this raw chain. */
  metadata?: { summaryUsedTokens?: number };
}

interface ProjectionMessageFilter {
  conversationId: string;
  user?: string;
  messageId?: string | { $in: string[] };
}

interface ProjectionMessageQueryOptions {
  limit?: number;
  sort?: false;
}

interface ProjectionMessageTextStats {
  messageId: string;
  textBytes: number;
  quoteCount: number;
  quoteBytes: number;
  quoteLineCount: number;
  nonStringQuoteCount: number;
}

interface ProjectionMessageTextStatsOptions {
  limit?: number;
}

export interface ContextProjectionDeps {
  /** Authenticated requester — branch lookups are scoped to this user. */
  userId?: string;
  getMessages: (
    filter: ProjectionMessageFilter,
    select?: string,
    options?: ProjectionMessageQueryOptions,
  ) => Promise<ProjectionMessage[]>;
  getMessageTextStats: (
    filter: ProjectionMessageFilter,
    options?: ProjectionMessageTextStatsOptions,
  ) => Promise<ProjectionMessageTextStats[]>;
}

/**
 * Walks the parent chain from `tailId` to root and returns the branch messages
 * oldest→newest. The visited set guards against cycles / self-referential links.
 */
function resolveBranch(messages: ProjectionMessage[], tailId: string): ProjectionMessage[] {
  const byId = new Map<string, ProjectionMessage>();
  for (const message of messages) {
    byId.set(message.messageId, message);
  }
  const branch: ProjectionMessage[] = [];
  const seen = new Set<string>();
  let currentId: string | null | undefined = tailId;
  while (currentId != null && !seen.has(currentId)) {
    const message = byId.get(currentId);
    if (message == null) {
      break;
    }
    seen.add(currentId);
    branch.push(message);
    currentId = message.parentMessageId;
  }
  return branch.reverse();
}

function hasValidProjectionIds(params: TContextProjectionRequest): boolean {
  return typeof params.conversationId === 'string' && typeof params.messageId === 'string';
}

function getProjectionText(message: ProjectionMessage): string | null {
  const hasQuotes =
    message.isCreatedByUser === true && Array.isArray(message.quotes) && message.quotes.length > 0;
  if (!hasQuotes) {
    return message.text ?? '';
  }
  if (message.quotes == null || message.quotes.length > QUOTE_MAX_COUNT) {
    return null;
  }
  for (const quote of message.quotes) {
    if (typeof quote !== 'string') {
      return null;
    }
  }
  return mergeQuotedText(message.text ?? '', message.quotes);
}

function hasExceededBranchTextLimit(branch: ProjectionMessage[]): boolean {
  let bytes = 0;
  for (const message of branch) {
    const text = getProjectionText(message);
    if (text == null) {
      return true;
    }
    bytes += Buffer.byteLength(text, 'utf8');
    if (bytes > MAX_PROJECTION_BRANCH_TEXT_BYTES) {
      return true;
    }
  }
  return false;
}

function getEstimatedMergedTextBytes(stats: ProjectionMessageTextStats): number | null {
  if (
    stats.nonStringQuoteCount > 0 ||
    stats.quoteCount > QUOTE_MAX_COUNT ||
    stats.quoteLineCount < stats.quoteCount
  ) {
    return null;
  }
  if (stats.quoteCount === 0) {
    return stats.textBytes;
  }

  const quotePrefixBytes = stats.quoteLineCount * 2;
  const quoteLineBreakBytes = stats.quoteLineCount - stats.quoteCount;
  const quoteSeparatorBytes = (stats.quoteCount - 1) * 2;
  const bodySeparatorBytes = stats.textBytes > 0 ? 2 : 0;
  return (
    stats.textBytes +
    stats.quoteBytes +
    quotePrefixBytes +
    quoteLineBreakBytes +
    quoteSeparatorBytes +
    bodySeparatorBytes
  );
}

function hasExceededBranchTextStatsLimit(stats: ProjectionMessageTextStats[]): boolean {
  let bytes = 0;
  for (const messageStats of stats) {
    const messageBytes = getEstimatedMergedTextBytes(messageStats);
    if (messageBytes == null) {
      return true;
    }
    bytes += messageBytes;
    if (bytes > MAX_PROJECTION_BRANCH_TEXT_BYTES) {
      return true;
    }
  }
  return false;
}

/** Maps an endpoint/provider string to the agents `Providers` enum. */
function resolveProvider(value?: string): Providers {
  if (value == null || value === '') {
    return Providers.OPENAI;
  }
  const lower = value.toLowerCase();
  for (const provider of Object.values(Providers)) {
    if (provider.toLowerCase() === lower) {
      return provider;
    }
  }
  if (lower.includes('anthropic') || lower.includes('claude')) {
    return Providers.ANTHROPIC;
  }
  if (lower.includes('google') || lower.includes('gemini') || lower.includes('vertex')) {
    return Providers.GOOGLE;
  }
  if (lower.includes('bedrock')) {
    return Providers.BEDROCK;
  }
  return Providers.OPENAI;
}

async function getBranchMessages(
  deps: ContextProjectionDeps,
  baseFilter: ProjectionMessageFilter,
  branch: ProjectionMessage[],
): Promise<ProjectionMessage[] | null> {
  const branchIds = branch.map((message) => message.messageId);
  const stats = await deps.getMessageTextStats(
    { ...baseFilter, messageId: { $in: branchIds } },
    { limit: branchIds.length },
  );
  if (stats.length !== branchIds.length || hasExceededBranchTextStatsLimit(stats)) {
    return null;
  }

  const stored = await deps.getMessages(
    { ...baseFilter, messageId: { $in: branchIds } },
    PROJECTION_BODY_SELECT,
    { limit: branchIds.length, sort: false },
  );
  if (stored.length !== branchIds.length) {
    return null;
  }
  const byId = new Map<string, ProjectionMessage>();
  for (const message of stored) {
    byId.set(message.messageId, message);
  }
  const ordered: ProjectionMessage[] = [];
  for (const messageId of branchIds) {
    const message = byId.get(messageId);
    if (message == null) {
      return null;
    }
    ordered.push(message);
  }
  return ordered;
}

/**
 * Server-side context-usage projection: reconstructs the viewed branch and asks
 * the agents SDK what the next call's context would be, WITHOUT invoking the
 * model. Provider/model/window come from the (client-resolved) request — no
 * agent or model-spec config is loaded here, so there is no cross-user config
 * exposure. Reuses LibreChat's already-calibrated per-message `tokenCount` for
 * any message [cowork] `applyContextReductionCore` leaves untouched (no
 * re-tokenizing); messages the reducer strips/truncates/dedupes are recounted
 * from the reduced content so the projection matches what the live path would
 * actually send — see docs/18_context_reduction.md. Returns null when there is
 * no resolvable context window.
 * NOTE: this first cut targets message-windowing accuracy — instruction and
 * tool-schema tokens (agent instructions, `promptPrefix`, model-spec presets,
 * tool schemas) are NOT yet included; a follow-up will reuse the full
 * `initializeAgent`/send path for exact overhead and proper access control.
 */
export async function resolveContextProjection(
  deps: ContextProjectionDeps,
  params: TContextProjectionRequest,
): Promise<TContextUsageEvent | null> {
  if (!hasValidProjectionIds(params)) {
    return null;
  }

  const maxContextTokens = params.maxContextTokens;
  if (maxContextTokens == null || maxContextTokens <= 0) {
    return null;
  }

  const baseFilter = { conversationId: params.conversationId, user: deps.userId };
  const stored = await deps.getMessages(baseFilter, PROJECTION_GRAPH_SELECT, {
    limit: MAX_PROJECTION_MESSAGES + 1,
    sort: false,
  });
  if (stored.length > MAX_PROJECTION_MESSAGES) {
    return null;
  }

  const branch = resolveBranch(stored, params.messageId);
  if (branch.length === 0) {
    return null;
  }
  if (branch.length > MAX_PROJECTION_BRANCH_MESSAGES) {
    return null;
  }

  /** A summarized/compacted branch's next call sends the saved summary + the
   *  post-summary tail, NOT this raw parent chain — projecting from the full
   *  history would prune/count the wrong context and omit the summary. Detect it
   *  via the live path's `metadata.summaryUsedTokens` marker and fall back (null)
   *  so the client's summary-baseline-aware estimate handles these branches until
   *  a follow-up replays the summary boundary. */
  if (branch.some((message) => (message.metadata?.summaryUsedTokens ?? 0) > 0)) {
    return null;
  }

  const bodyBranch = await getBranchMessages(deps, baseFilter, branch);
  if (bodyBranch == null || hasExceededBranchTextLimit(bodyBranch)) {
    return null;
  }

  const model = params.model;
  const encoding = (model ?? '').toLowerCase().includes('claude') ? 'claude' : 'o200k_base';
  const tokenCounter = await createTokenCounter(encoding);

  const messages: BaseMessage[] = [];
  /** [cowork] Parallel array of `{ role, content }` objects mirroring what the
   *  live path's `formatMessage` produces, fed through the same
   *  `applyContextReductionCore` the live path uses so this projection
   *  reflects deduped/truncated tool output and stripped thinking blocks
   *  instead of the full original conversation. See docs/18_context_reduction.md. */
  const reducibleMessages: ReducibleMessage[] = [];
  /** Per-message flags carried alongside `reducibleMessages` (same index) —
   *  kept out of the `ReducibleMessage` shape itself since they're gauge-only
   *  bookkeeping, not part of what the reducer or `countFormattedMessageTokens`
   *  operate on. */
  const hasQuotesByIndex: boolean[] = [];
  for (let i = 0; i < bodyBranch.length; i++) {
    const message = bodyBranch[i];
    /** Mirror the live path: prepend quoted excerpts into the user text the model
     *  receives so the gauge counts the same prompt. */
    const hasQuotes =
      message.isCreatedByUser === true &&
      Array.isArray(message.quotes) &&
      message.quotes.length > 0;
    const text = getProjectionText(message);
    if (text == null) {
      return null;
    }
    const lcMessage =
      message.isCreatedByUser === true ? new HumanMessage(text) : new AIMessage(text);
    messages.push(lcMessage);
    hasQuotesByIndex.push(hasQuotes);

    /**
     * [cowork] Quoted turns use the merged text (quotes are folded into `text`
     * via `getProjectionText`/`mergeQuotedText`, not represented as a
     * structured content part here) as a single text-part fallback. Otherwise
     * prefer the raw stored `content` array (think/tool_call/text parts) when
     * present — that's what actually determines what the live path sends —
     * falling back to the flattened `text` for plain turns that never had a
     * structured `content` field.
     */
    const contentForReduction: FormattedMessageContentPart[] | string =
      !hasQuotes && Array.isArray(message.content) && message.content.length > 0
        ? message.content
        : [{ type: 'text', text }];
    reducibleMessages.push({
      role: message.isCreatedByUser === true ? 'user' : 'assistant',
      content: contentForReduction,
    });
  }

  const reducedMessages = CONTEXT_REDUCTION_ENABLED
    ? applyContextReductionCore(reducibleMessages).messages
    : reducibleMessages;

  const indexTokenCountMap: Record<string, number> = {};
  for (let i = 0; i < reducedMessages.length; i++) {
    const message = bodyBranch[i];
    const hasQuotes = hasQuotesByIndex[i];
    /** Untouched by the reducer — the exact same object reference survives
     *  `applyContextReductionCore` (see its docstring) — so the stored
     *  canonical count is still accurate and can be reused instead of paying
     *  for a re-tokenization pass. Recount messages with no stored count
     *  (imported / pre-feature) rather than charging 0 — a real 0 and
     *  "unknown" must not collapse, or the snapshot-less histories this
     *  endpoint targets would under-report. Also always recount quoted
     *  messages: a text-only Save edit leaves a stale text-only `tokenCount`
     *  that omits the quote block, so trust the merged recount. Anything the
     *  reducer actually stripped/truncated/deduped must be recounted from the
     *  reduced content — that's the whole point of this pass. */
    const wasReduced = reducedMessages[i] !== reducibleMessages[i];
    indexTokenCountMap[String(i)] =
      !hasQuotes && !wasReduced && message.tokenCount != null && message.tokenCount > 0
        ? message.tokenCount
        : countFormattedMessageTokens(reducedMessages[i] as Record<string, unknown>, encoding);
  }

  return projectAgentContextUsage({
    agent: {
      agentId: params.agentId ?? 'projection',
      provider: resolveProvider(params.endpoint),
      maxContextTokens,
    },
    messages,
    tokenCounter,
    indexTokenCountMap,
    calibrationRatio: params.calibrationRatio,
  });
}
