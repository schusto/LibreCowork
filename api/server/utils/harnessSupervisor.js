'use strict';

/**
 * harnessSupervisor.js — active harness-level supervision for memory/work-graph
 * retrieval and stuck-pattern detection, applied to LibreCowork's agent request
 * pipeline. Two independent pieces, both called from `buildMessages` in
 * `api/server/controllers/agents/client.js`, pushed into the shared
 * `sharedRunContextParts` pool (visible identically to every agent in the run —
 * this is deliberate, see docs/31_harness_memory_supervisor.md §3 and
 * docs/36_harness_supervisor_implementation_report.md):
 *
 *   1. getProactiveMemoryContext() — fires BEFORE the model's first turn.
 *      Embeds the latest user message and queries the `work_graph` and
 *      `chats` Qdrant collections directly (same infra `consolidate.js` and
 *      `mcp/memory/server.js` already use), then returns a labelled context
 *      block. This does not depend on the model deciding to call
 *      `search_work_graph` / `search_memory` — it happens whether or not the
 *      model would have remembered to. The existing tools stay available and
 *      still carry their "CALL THIS TOOL BEFORE STARTING ANY TASK" mandate
 *      for deeper, differently-phrased follow-up queries; this is a floor,
 *      not a replacement.
 *
 *      Gated by a topic-shift check so a multi-turn task on the same subject
 *      doesn't re-inject near-identical snippets every turn: an exponential
 *      moving-average (EMA) centroid of the conversation's recent message
 *      embeddings is tracked per conversation, and injection is skipped when
 *      the current message's cosine similarity to that centroid stays above
 *      CWK_SUPERVISOR_TOPIC_SHIFT_THRESHOLD for fewer than
 *      CWK_SUPERVISOR_TOPIC_SHIFT_MAX_STALE_TURNS turns since the last actual
 *      injection. Cosine similarity is the same distance metric the
 *      `work_graph`/`chats` Qdrant collections themselves use (`Cosine`, see
 *      mcp/memory/server.js), and reuses the same embedding already computed
 *      for the search below rather than a second embed call.
 *
 *   2. trackAndDetectStuckPattern() — fires AFTER context reduction runs,
 *      using `stats.toolDeduped` (already computed by
 *      applyContextReductionWithStats, see contextReduction.js) to detect a
 *      session that keeps re-deriving/re-requesting content it already has.
 *      When repetition crosses a threshold across consecutive turns, it
 *      returns a nudge string to inject — the closest analogue in this stack
 *      to the "supervisor" component described in the NVIDIA harness
 *      article: not a smarter model, a piece of the harness that notices the
 *      agent going in circles and says something about it. The threshold is
 *      configurable per agent (see CWK_SUPERVISOR_STUCK_TURNS_BY_AGENT below)
 *      since a Coder agent doing rapid file edits naturally trips
 *      `toolDeduped`-based detection more often than a Writer agent.
 *
 *   3. Stall-recovery continuation (getStallProbeTagInstruction /
 *      shouldFireStallProbe / createStallRecoveryStopHook /
 *      tryReserveStallProbeSlot / resetAutoContinueCounter) — catches the
 *      opposite failure mode from
 *      (2): the agent goes IDLE with NO further tool call, mid-task,
 *      instead of repeating one. Observed live with the local qwen3.6-35b
 *      model: it narrates an intended next action ("now executing Test
 *      2...") and then simply stops generating, no error anywhere.
 *      `trackAndDetectStuckPattern` cannot catch this — it fires on
 *      `toolDeduped > 0`, i.e. repetition, not silence. See
 *      docs/37_stall_recovery_handover.md and
 *      docs/39_stall_recovery_implementation_report.md for the full design
 *      history and the reasoning trail (in particular why this continues
 *      the same run rather than making a separate isolated classifier call
 *      — docs/39 §2 — and a real over-triggering flaw found only through
 *      live testing — docs/39 §3), and
 *      docs/41_stall_recovery_stop_hook_report.md for the mechanism that
 *      actually delivers the continuation.
 *
 *      Detection is anchored on an explicit completion marker
 *      (`DONE_TAG`, default `<---DONE--->`) rather than trying to infer
 *      completion from message shape alone: `getStallProbeTagInstruction()`
 *      is pushed into `sharedRunContextParts` every turn (like components 1
 *      and 2, this one DOES run before the model's turn for this part),
 *      asking the model to append the tag whenever a response is genuinely
 *      final. `shouldFireStallProbe()` then keys off that tag — present →
 *      done, absent → continuation candidate — applied consistently to the
 *      original response and every continuation round by the same
 *      predicate. (`isDoneMessage()` applied the identical test to the
 *      probe's reply back when the two rounds were checked separately; it
 *      is kept and still unit-tested, but the live path no longer needs it
 *      — one predicate now covers every round.) DONE_TAG absence is the
 *      SOLE signal — an earlier length-based fallback ("only fire if the
 *      final text was also short") was tried and removed after live
 *      testing showed it silently suppressed the probe on a genuinely
 *      incomplete response that just happened to be moderately long (see
 *      docs/39_stall_recovery_implementation_report.md §10 for the
 *      incident and shouldFireStallProbe()'s own docstring for the
 *      reasoning behind removing it rather than tuning the threshold).
 *
 *      The continuation itself runs as a `Stop` hook
 *      (`createStallRecoveryStopHook()`), registered on the run by
 *      `createRun` and dispatched by the SDK at natural completion (no HITL
 *      interrupt, no hook halt). The hook inspects the run's messages via
 *      `shouldFireStallProbe()`; when true it returns
 *      `{ decision: 'block', reason: STALL_PROBE_MESSAGE }`, which makes the
 *      SDK re-enter the stream IN PLACE — inside the same `processStream()`
 *      call — with that nudge appended, asking the model to either print the
 *      tag or continue for real. Because nothing resets between rounds, the
 *      continuation streams to the client exactly like an ordinary extra
 *      model turn. An earlier design called `processStream()` a second time
 *      from the host instead; that could not render correctly and is
 *      dissected in docs/41 §2. `tryReserveStallProbeSlot()` gates each attempt against a
 *      per-conversation cap on CONSECUTIVE UNPRODUCTIVE rounds
 *      (CWK_SUPERVISOR_STALL_PROBE_MAX_CONTINUATIONS), plus an absolute
 *      per-turn ceiling (CWK_SUPERVISOR_STALL_PROBE_MAX_TOTAL_CONTINUATIONS),
 *      stored in the same `harness_supervisor_state` document as the other
 *      two components' state, so a conversation that keeps stalling is
 *      auto-continued a bounded number of times (mirroring Erik's own
 *      manual "regenerate 2-3 times" recovery habit) and then left alone.
 *      `resetAutoContinueCounter()` resets that count once per real
 *      user-initiated turn (called from `buildMessages`, NOT from inside
 *      the hook itself, so continuation rounds don't reset their own cap).
 *
 * Both functions fail open: on timeout, network error, Mongo error, or
 * missing config, they return null and the turn proceeds exactly as it does
 * today. Neither writes to Qdrant; both read/write a small per-conversation
 * state document in MongoDB (see "Durable state" below) via the same
 * mongoose connection every other model in this app already shares — no new
 * DB connection is opened.
 *
 * Deliberate scope cut, carried over from v1: dense-vector search only, no
 * keyword/RRF fusion. `mcp/memory/lib/hybrid-search.js` already implements
 * keywordSearch() + reciprocalRankFusion() for the MCP tools' own queries;
 * porting that here would close some query-quality gap on exact-term queries
 * but roughly doubles this file's size. If sparse-injection quality turns
 * out to be the bottleneck (measurable via the CWK_SUPERVISOR log lines
 * below), port it then.
 *
 * Durable state: a single `harness_supervisor_state` collection, one
 * document per conversationId (`_id`), holding both components' state
 * (stuck-pattern counters, topic-shift centroid). A MongoDB TTL index on
 * `lastSeen` expires stale conversations automatically after
 * CWK_SUPERVISOR_STATE_TTL_MS — this replaces the earlier draft's in-memory
 * `Map`, which reset on every container restart. Concurrent updates from the
 * two components use field-scoped `$set`s against the same document, so they
 * cannot clobber each other's fields.
 *
 * Config (env vars, following the CWK_* convention already used by
 * contextReduction.js):
 *
 *   CWK_SUPERVISOR_ENABLED                    Master on/off switch for both components (default: true)
 *   CWK_SUPERVISOR_MIN_MESSAGE_CHARS          Skip proactive search below this length (default: 15)
 *   CWK_SUPERVISOR_WORK_GRAPH_LIMIT           Max work-graph hits to inject (default: 2)
 *   CWK_SUPERVISOR_WORK_GRAPH_THRESHOLD       Cosine score floor for work-graph (default: 0.3 — matches search_work_graph's own default)
 *   CWK_SUPERVISOR_CHATS_LIMIT                Max past-chat hits to inject (default: 2)
 *   CWK_SUPERVISOR_CHATS_THRESHOLD            Cosine score floor for chats (default: 0.25 — matches search_memory's own default)
 *   CWK_SUPERVISOR_TIMEOUT_MS                 Hard timeout for the whole proactive-context pass (default: 2000)
 *   CWK_SUPERVISOR_STUCK_TURNS                Consecutive turns with tool_deduped>0 before nudging, global default (default: 3)
 *   CWK_SUPERVISOR_STUCK_TURNS_BY_AGENT       JSON object mapping agent id (suffix-stripped, e.g. `{"agent_abc123":2}`) to a
 *                                             per-agent override of CWK_SUPERVISOR_STUCK_TURNS (default: '{}', i.e. no overrides).
 *                                             NOTE: keyed by the agent's own persisted id (same suffix-stripping
 *                                             `getMemoryAgentId`/`stripAgentIdSuffix` use), NOT by calling
 *                                             `getMemoryAgentId(agent)` itself — that function returns `undefined` for
 *                                             any agent that hasn't opted into isolated memory (`memory_scope: 'agent'`),
 *                                             which is most agents, so it collapses to one bucket for almost every run
 *                                             and can't actually differentiate a Coder agent from a Writer agent.
 *   CWK_SUPERVISOR_TOPIC_SHIFT_ENABLED        Master on/off switch for the topic-shift injection gate (default: true)
 *   CWK_SUPERVISOR_TOPIC_SHIFT_THRESHOLD      Cosine similarity to the rolling topic centroid above which the current
 *                                             message counts as "same topic" (default: 0.92)
 *   CWK_SUPERVISOR_TOPIC_SHIFT_EMA_ALPHA      Exponential moving-average weight given to each new message's embedding
 *                                             when updating the rolling topic centroid (default: 0.5)
 *   CWK_SUPERVISOR_TOPIC_SHIFT_MAX_STALE_TURNS  Force re-injection after this many consecutive gated turns regardless
 *                                             of topic similarity, so a long single-topic task doesn't get permanently
 *                                             silenced (default: 6)
 *   CWK_SUPERVISOR_STATE_TTL_MS                TTL (ms) for the per-conversation state document; also the MongoDB TTL
 *                                             index's expireAfterSeconds (default: 21600000 — 6h)
 *   CWK_SUPERVISOR_STALL_PROBE_ENABLED        Master on/off switch for the stall-recovery probe specifically
 *                                             (default: true — still gated by CWK_SUPERVISOR_ENABLED above)
 *   CWK_SUPERVISOR_STALL_PROBE_MAX_CONTINUATIONS  Max CONSECUTIVE UNPRODUCTIVE auto-continue rounds before giving
 *                                             up and leaving the turn as-is (default: 2). "Unproductive" means the
 *                                             round produced no new tool call; a round that did real work resets
 *                                             this to zero, so a long task that stalls between genuine steps is
 *                                             carried to the end while an agent that is truly spinning is cut off
 *                                             just as fast as a flat cap would. Both counters reset on the next real
 *                                             user-initiated turn, not on every internal continuation round.
 *                                             (Was a flat per-turn budget until a live run of the 16-test regression
 *                                             suite exhausted it three-quarters of the way through a turn that was
 *                                             still making steady progress — see 41 §10.)
 *   CWK_SUPERVISOR_STALL_PROBE_MAX_TOTAL_CONTINUATIONS  Absolute per-turn ceiling regardless of progress (default: 8).
 *                                             Must stay BELOW the MAX_STOP_HOOK_CONTINUATIONS backstop in
 *                                             patches/@librechat+agents+3.4.7.patch (10), which ends the turn with
 *                                             no log line at all — raise both together or not at all.
 *   CWK_SUPERVISOR_STALL_PROBE_DONE_TAG       The explicit completion marker the model is asked, every turn, to
 *                                             append to a genuinely final response (default: `<---DONE--->`).
 *                                             Presence/absence of this exact tag is the PRIMARY stall signal —
 *                                             see getStallProbeTagInstruction() / shouldFireStallProbe() —
 *                                             checked consistently on the original response
 *                                             and every continuation round. Added after live testing
 *                                             (docs/39_stall_recovery_implementation_report.md §3, §7) showed the
 *                                             earlier message-shape-only trigger (any tool call this run +
 *                                             plain-text end) fires on EVERY normal tool-using turn's legitimate
 *                                             conclusion, not just genuine stalls — a real finished answer is
 *                                             structurally identical to "narrated intent, then went idle" from
 *                                             the message shape alone. An explicit marker the model is
 *                                             instructed to emit is a direct signal instead of an inferred one.
 *   CWK_SUPERVISOR_STALL_PROBE_MESSAGE        The nudge text appended as a message when the tag was
 *                                             absent (default: see STALL_PROBE_MESSAGE below — reinforces the
 *                                             same tag convention rather than a separate literal-"DONE" ask)
 *   CWK_SUPERVISOR_STALL_PROBE_REENTRY_ENABLED  Whether to register the `Stop` hook that actually performs the
 *                                             continuation (default: false). See STALL_PROBE_REENTRY_ENABLED
 *                                             below for why it is still opt-in, and docs/41 for the mechanism.
 *
 *   (No length-based fallback exists — DONE_TAG absence is the sole trigger
 *   signal. A CWK_SUPERVISOR_STALL_PROBE_MAX_FINAL_TEXT_CHARS fallback was
 *   tried and REMOVED: live testing showed it silently suppressed the probe
 *   on a genuinely incomplete 537-character response, since length alone
 *   cannot distinguish "complete" from "incomplete but substantial." See
 *   docs/39_stall_recovery_implementation_report.md §10 and
 *   shouldFireStallProbe()'s docstring.)
 *
 *   EMBED_URL, QDRANT_URL, QDRANT_COLL_WORK_GRAPH, QDRANT_COLL_CHATS —
 *   intentionally reuse the exact same env var names consolidate.sh already
 *   sets, so the two processes agree on where memory lives without a second
 *   place to configure it. Defaults mirror mcp/memory/lib/config.js.
 */

const axios = require('axios');
const mongoose = require('mongoose');
const { logger } = require('@librechat/data-schemas');

// ── Config ───────────────────────────────────────────────────────────────────

const SUPERVISOR_ENABLED =
  (process.env.CWK_SUPERVISOR_ENABLED ?? 'true').toLowerCase() !== 'false';

const MIN_MESSAGE_CHARS   = parseInt(process.env.CWK_SUPERVISOR_MIN_MESSAGE_CHARS ?? '15', 10);
const WORK_GRAPH_LIMIT    = parseInt(process.env.CWK_SUPERVISOR_WORK_GRAPH_LIMIT ?? '2', 10);
const WORK_GRAPH_THRESHOLD = parseFloat(process.env.CWK_SUPERVISOR_WORK_GRAPH_THRESHOLD ?? '0.3');
const CHATS_LIMIT         = parseInt(process.env.CWK_SUPERVISOR_CHATS_LIMIT ?? '2', 10);
const CHATS_THRESHOLD     = parseFloat(process.env.CWK_SUPERVISOR_CHATS_THRESHOLD ?? '0.25');
const TIMEOUT_MS          = parseInt(process.env.CWK_SUPERVISOR_TIMEOUT_MS ?? '2000', 10);
const STUCK_TURNS         = parseInt(process.env.CWK_SUPERVISOR_STUCK_TURNS ?? '3', 10);

const TOPIC_SHIFT_ENABLED =
  (process.env.CWK_SUPERVISOR_TOPIC_SHIFT_ENABLED ?? 'true').toLowerCase() !== 'false';
const TOPIC_SHIFT_THRESHOLD = parseFloat(process.env.CWK_SUPERVISOR_TOPIC_SHIFT_THRESHOLD ?? '0.92');
const TOPIC_SHIFT_EMA_ALPHA = parseFloat(process.env.CWK_SUPERVISOR_TOPIC_SHIFT_EMA_ALPHA ?? '0.5');
const TOPIC_SHIFT_MAX_STALE_TURNS =
  parseInt(process.env.CWK_SUPERVISOR_TOPIC_SHIFT_MAX_STALE_TURNS ?? '6', 10);

const STATE_TTL_MS = parseInt(
  process.env.CWK_SUPERVISOR_STATE_TTL_MS ?? String(6 * 60 * 60 * 1000),
  10,
);

const STALL_PROBE_ENABLED =
  (process.env.CWK_SUPERVISOR_STALL_PROBE_ENABLED ?? 'true').toLowerCase() !== 'false';
/**
 * Separate, narrower switch from STALL_PROBE_ENABLED above: it gates whether
 * the `Stop` hook that performs the continuation is registered at all.
 * Detection-side behaviour (the DONE_TAG instruction pushed into every turn's
 * context) is governed by STALL_PROBE_ENABLED and stays active regardless.
 *
 * Still defaults to DISABLED, but for a different reason than before. The
 * mechanism it gates has CHANGED: it used to re-enter `run.processStream()`
 * from the host after the first call returned, which live testing showed
 * corrupts message ordering and drops content from the live view until a
 * reload (docs/40). That approach is gone — root-caused (docs/41 §2) to
 * `Graph.clearHeavyState()` running in `processStream`'s `finally` and wiping
 * the content bookkeeping and handler registry the second call needed. The
 * continuation now happens inside the SDK's own stream loop via a `Stop`
 * hook, which is mechanically verified to append content in the correct
 * order (docs/41 §3). The default stays `false` only because that has not yet
 * been confirmed against a LIVE conversation — every prior live run in this
 * feature's history surfaced a real bug the mechanical tests missed (docs/39
 * §3, §8, §10). Flip to `true` for that live run.
 *
 * Requires the local `@librechat/agents` patch (`patches/`, docs/41 §4). On an
 * unpatched SDK the hook still runs and logs, but its decision is discarded
 * and the run finalizes normally — inert, never corrupting.
 */
const STALL_PROBE_REENTRY_ENABLED =
  (process.env.CWK_SUPERVISOR_STALL_PROBE_REENTRY_ENABLED ?? 'false').toLowerCase() === 'true';
const STALL_PROBE_MAX_CONTINUATIONS = parseInt(
  process.env.CWK_SUPERVISOR_STALL_PROBE_MAX_CONTINUATIONS ?? '2',
  10,
);
/**
 * Absolute per-turn ceiling on continuations, regardless of how much progress
 * the agent is making. STALL_PROBE_MAX_CONTINUATIONS above only bounds
 * CONSECUTIVE UNPRODUCTIVE rounds, so on its own it would let a model that
 * stalls after every single tool call continue indefinitely.
 *
 * Default 8 is deliberately BELOW the `MAX_STOP_HOOK_CONTINUATIONS = 10`
 * runaway backstop in `patches/@librechat+agents+3.4.7.patch`. That backstop
 * just breaks the SDK's loop with no log line, so if it ever became the
 * binding limit the turn would end silently — precisely the class of failure
 * docs/39 §10 was written about. Keep this below the patch's value, or raise
 * both together.
 */
const STALL_PROBE_MAX_TOTAL_CONTINUATIONS = parseInt(
  process.env.CWK_SUPERVISOR_STALL_PROBE_MAX_TOTAL_CONTINUATIONS ?? '8',
  10,
);
/**
 * Explicit completion marker the model is asked (via getStallProbeTagInstruction(),
 * injected into sharedRunContextParts every turn) to append when a response is
 * genuinely final. Its PRESENCE/ABSENCE is the primary stall signal — see
 * shouldFireStallProbe() and isDoneMessage() — replacing the earlier pass's
 * literal-"DONE"-reply / final-text-length heuristics with one direct signal
 * checked consistently on the original response and every probe round.
 * Chosen to be distinctive (unlikely to appear in normal prose) and cheap to
 * scan for with a plain substring check.
 */
const DONE_TAG = process.env.CWK_SUPERVISOR_STALL_PROBE_DONE_TAG ?? '<---DONE--->';
const STALL_PROBE_MESSAGE =
  process.env.CWK_SUPERVISOR_STALL_PROBE_MESSAGE ??
  `If the work is done and no instructions are missing just print ${DONE_TAG}, ` +
    'otherwise continue your tasks and achieve your goal. When you are fully ' +
    `done add ${DONE_TAG} at the end.`;

/**
 * Text shown in the chat when a continuation fires. Set to an empty string to
 * suppress the indicator while leaving the continuation itself enabled.
 *
 * Rendered as an `activity_label` content part, NOT as text in the response:
 * `packages/api/src/agents/client.ts` classifies activity labels as "UI-only
 * progress headers — never model input, never billed output", so the marker
 * cannot leak back into the model's context on a later turn or be charged as
 * output tokens. Streaming a raw string into the response instead would do
 * both. `%d` is replaced with the continuation number — the live run that
 * motivated this could not tell two restarts from three (docs/41 §10).
 */
const STALL_PROBE_INDICATOR =
  process.env.CWK_SUPERVISOR_STALL_PROBE_INDICATOR ??
  'Continuing automatically — the response stopped before signalling completion (continuation %d)';

const EMBED_URL  = process.env.EMBED_URL  || 'http://host.docker.internal:9000/v1/embeddings';
const QDRANT_URL = process.env.QDRANT_URL || 'http://host.docker.internal:6333';
const QDRANT_COLL_WORK_GRAPH = process.env.QDRANT_COLL_WORK_GRAPH || 'work_graph';
const QDRANT_COLL_CHATS      = process.env.QDRANT_COLL_CHATS      || 'chats';

function parseAgentThresholdMap(raw) {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch (e) {
    logger.warn(
      `[harnessSupervisor] CWK_SUPERVISOR_STUCK_TURNS_BY_AGENT is not valid JSON, ignoring: ${e.message}`,
    );
    return {};
  }
}

const STUCK_TURNS_BY_AGENT = parseAgentThresholdMap(process.env.CWK_SUPERVISOR_STUCK_TURNS_BY_AGENT);

/**
 * Stable per-agent key for threshold lookups: the agent's own persisted id,
 * with the `____N` parallel-run suffix stripped (same normalization
 * `stripAgentIdSuffix` in packages/data-provider/src/parsers.ts applies) so
 * the same agent resolves to the same key across single- and multi-agent
 * runs. Deliberately NOT `getMemoryAgentId(agent)` — see the module docstring.
 */
function getAgentThresholdKey(agent) {
  const id = agent?.id;
  if (typeof id !== 'string' || !id) return 'default';
  return id.replace(/____\d+$/, '');
}

function resolveStuckTurnsThreshold(agent) {
  const override = STUCK_TURNS_BY_AGENT[getAgentThresholdKey(agent)];
  return Number.isFinite(override) && override > 0 ? override : STUCK_TURNS;
}

// ── Durable per-conversation state (MongoDB) ────────────────────────────────

const STATE_COLLECTION_NAME = 'harness_supervisor_state';
let stateIndexEnsured = false;

function getStateCollection() {
  const collection = mongoose.connection.collection(STATE_COLLECTION_NAME);
  if (!stateIndexEnsured) {
    stateIndexEnsured = true;
    // Fire-and-forget: index creation is idempotent and safe to race across
    // concurrent requests; a failure here only means state never expires,
    // not that reads/writes fail.
    collection
      .createIndex({ lastSeen: 1 }, { expireAfterSeconds: Math.floor(STATE_TTL_MS / 1000) })
      .catch((e) => logger.warn(`[harnessSupervisor] failed to ensure state TTL index: ${e.message}`));
  }
  return collection;
}

async function loadSupervisorState(conversationId) {
  try {
    return await getStateCollection().findOne({ _id: conversationId });
  } catch (e) {
    logger.warn(`[harnessSupervisor] state load failed: ${e.message}`);
    return null;
  }
}

async function saveSupervisorState(conversationId, patch) {
  try {
    await getStateCollection().updateOne(
      { _id: conversationId },
      { $set: { ...patch, lastSeen: new Date() } },
      { upsert: true },
    );
  } catch (e) {
    logger.warn(`[harnessSupervisor] state save failed: ${e.message}`);
  }
}

// ── Shared: embedding + cosine similarity ───────────────────────────────────

async function embedQuery(query) {
  const res = await axios.post(
    EMBED_URL,
    { input: [`search_query: ${query}`] },
    { timeout: TIMEOUT_MS },
  );
  const vector = res.data?.data?.[0]?.embedding;
  if (!vector) throw new Error('embed server returned no embedding');
  return vector;
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** EMA update: `alpha` weight on the new vector, `1 - alpha` on the prior centroid. */
function updateEmaCentroid(prevCentroid, embedding, alpha) {
  if (!Array.isArray(prevCentroid) || prevCentroid.length !== embedding.length) {
    return embedding.slice();
  }
  return embedding.map((v, i) => alpha * v + (1 - alpha) * prevCentroid[i]);
}

// ── Component 1: proactive memory / work-graph injection ───────────────────

async function searchCollection({ vector, collection, limit, scoreThreshold, filter }) {
  const res = await axios.post(
    `${QDRANT_URL}/collections/${collection}/points/search`,
    {
      vector,
      limit,
      with_payload: true,
      score_threshold: scoreThreshold,
      ...(filter ? { filter } : {}),
    },
    { timeout: TIMEOUT_MS },
  );
  return res.data?.result || [];
}

function formatWorkGraphHit(hit) {
  const { name, entity_type, summary } = hit.payload || {};
  const body = summary ? summary.trim().slice(0, 400) : '(no summary)';
  return `- [work graph: ${name || '(unnamed)'}, ${entity_type || '?'}] ${body}`;
}

function formatChatHit(hit) {
  const { title, date, summary } = hit.payload || {};
  const body = summary ? summary.trim().slice(0, 300) : '(no summary)';
  return `- [past conversation: "${title || '(untitled)'}", ${date || '?'}] ${body}`;
}

/**
 * @param {object} opts
 * @param {string} opts.userMessage       Latest user turn's text.
 * @param {string} [opts.conversationId]  Also keys the topic-shift gate's state; omit to skip gating.
 * @param {string} [opts.userId]          Scopes the `chats` search — omit to skip chat search.
 * @param {object} [opts.agent]           Unused today (threshold config only applies to component 2),
 *                                        accepted for a consistent call signature between both components.
 * @returns {Promise<string|null>}     A labelled context block to push into
 *   sharedRunContextParts, or null if nothing crossed threshold / gated by
 *   topic-shift / supervisor is disabled / the pass failed or timed out.
 */
async function getProactiveMemoryContext({ userMessage, conversationId, userId }) {
  if (!SUPERVISOR_ENABLED) return null;
  if (!userMessage || userMessage.trim().length < MIN_MESSAGE_CHARS) return null;

  const convLabel = conversationId ? conversationId.slice(-8) : 'unknown';

  const run = async () => {
    const vector = await embedQuery(userMessage);

    // ── Topic-shift gate ──────────────────────────────────────────────────
    // Skip injection when the current message is still cosine-similar to the
    // conversation's recent-topic centroid AND we injected recently — a
    // multi-turn task on the same subject shouldn't re-inject near-identical
    // snippets every turn. The centroid always advances via EMA regardless of
    // gating outcome, and gating is bypassed after TOPIC_SHIFT_MAX_STALE_TURNS
    // so a long single-topic task doesn't get permanently silenced.
    let nextCentroid = null;
    if (TOPIC_SHIFT_ENABLED && conversationId) {
      const state = await loadSupervisorState(conversationId);
      const centroid = state?.topicCentroid || null;
      const turnsSinceLastInjection = state?.turnsSinceLastInjection ?? 0;
      nextCentroid = updateEmaCentroid(centroid, vector, TOPIC_SHIFT_EMA_ALPHA);

      if (centroid) {
        const similarity = cosineSimilarity(vector, centroid);
        const gated =
          similarity >= TOPIC_SHIFT_THRESHOLD && turnsSinceLastInjection < TOPIC_SHIFT_MAX_STALE_TURNS;
        if (gated) {
          await saveSupervisorState(conversationId, {
            topicCentroid: nextCentroid,
            turnsSinceLastInjection: turnsSinceLastInjection + 1,
          });
          logger.info(
            `[harnessSupervisor] conv=${convLabel} proactive_context gated ` +
              `(topic_similarity=${similarity.toFixed(3)}, stale_turns=${turnsSinceLastInjection})`,
          );
          return null;
        }
      }
    }

    const searches = [
      searchCollection({
        vector,
        collection: QDRANT_COLL_WORK_GRAPH,
        limit: WORK_GRAPH_LIMIT,
        scoreThreshold: WORK_GRAPH_THRESHOLD,
      }).catch((e) => {
        logger.warn(`[harnessSupervisor] work_graph search failed: ${e.message}`);
        return [];
      }),
    ];

    if (userId) {
      searches.push(
        searchCollection({
          vector,
          collection: QDRANT_COLL_CHATS,
          limit: CHATS_LIMIT,
          scoreThreshold: CHATS_THRESHOLD,
          filter: { must: [{ key: 'user', match: { value: userId } }] },
        }).catch((e) => {
          logger.warn(`[harnessSupervisor] chats search failed: ${e.message}`);
          return [];
        }),
      );
    }

    const [workGraphHits, chatHits = []] = await Promise.all(searches);

    if (workGraphHits.length === 0 && chatHits.length === 0) {
      // Nothing relevant found — still advance the topic centroid (we did see
      // this message), but don't reset the staleness counter: no injection
      // actually happened, so the "hasn't injected recently" clock keeps ticking.
      if (TOPIC_SHIFT_ENABLED && conversationId) {
        const turnsSinceLastInjection = (await loadSupervisorState(conversationId))
          ?.turnsSinceLastInjection ?? 0;
        await saveSupervisorState(conversationId, {
          topicCentroid: nextCentroid,
          turnsSinceLastInjection: turnsSinceLastInjection + 1,
        });
      }
      return null;
    }

    const lines = [
      '# Auto-surfaced background (harness supervisor)',
      '',
      'The following were retrieved automatically based on similarity to your ' +
        'current message — not requested by you. Treat them as a starting ' +
        'point, not a complete answer: call `search_work_graph` or ' +
        '`search_memory` directly for a deeper or differently-phrased look ' +
        'before relying on this alone.',
      '',
      ...workGraphHits.map(formatWorkGraphHit),
      ...chatHits.map(formatChatHit),
    ];

    const block = lines.join('\n');

    if (TOPIC_SHIFT_ENABLED && conversationId) {
      await saveSupervisorState(conversationId, {
        topicCentroid: nextCentroid,
        turnsSinceLastInjection: 0,
      });
    }

    logger.info(
      `[harnessSupervisor] conv=${convLabel} proactive_context ` +
        `work_graph=${workGraphHits.length} chats=${chatHits.length} ` +
        `~chars=${block.length}`,
    );
    return block;
  };

  const timeout = new Promise((resolve) =>
    setTimeout(() => resolve(null), TIMEOUT_MS),
  );

  try {
    return await Promise.race([run(), timeout]);
  } catch (e) {
    logger.warn(`[harnessSupervisor] conv=${convLabel} proactive context failed: ${e.message}`);
    return null;
  }
}

// ── Component 2: dead-end / stuck-pattern detector ──────────────────────────

/**
 * @param {object} opts
 * @param {string} opts.conversationId
 * @param {{toolDeduped?: number}} opts.stats  From applyContextReductionWithStats().
 * @param {object} [opts.agent]  Current agent, used to look up a per-agent threshold
 *   override (CWK_SUPERVISOR_STUCK_TURNS_BY_AGENT) — see getAgentThresholdKey().
 * @returns {Promise<string|null>}  A nudge string to inject, or null if nothing to say.
 */
async function trackAndDetectStuckPattern({ conversationId, stats, agent }) {
  if (!SUPERVISOR_ENABLED || !conversationId || !stats) return null;

  const threshold = resolveStuckTurnsThreshold(agent);
  const state = await loadSupervisorState(conversationId);
  const deduped = stats.toolDeduped || 0;

  let consecutiveDedupTurns = state?.consecutiveDedupTurns || 0;
  let cumulativeDeduped = state?.cumulativeDeduped || 0;

  if (deduped > 0) {
    consecutiveDedupTurns += 1;
    cumulativeDeduped += deduped;
  } else {
    consecutiveDedupTurns = 0;
  }

  if (consecutiveDedupTurns < threshold) {
    await saveSupervisorState(conversationId, { consecutiveDedupTurns, cumulativeDeduped });
    return null;
  }

  // Reset so the nudge fires once per episode, not every turn thereafter.
  await saveSupervisorState(conversationId, { consecutiveDedupTurns: 0, cumulativeDeduped });

  logger.info(
    `[harnessSupervisor] conv=${conversationId.slice(-8)} stuck-pattern nudge fired ` +
      `(threshold=${threshold}, cumulative_deduped=${cumulativeDeduped})`,
  );

  return (
    '# Harness note\n\n' +
    "You've re-requested content you already retrieved earlier in this " +
    'session several turns in a row. Before continuing the same approach: ' +
    '(1) check `search_work_graph` for a documented pattern on this task ' +
    'type — this exact loop may already be a recorded dead end, ' +
    '(2) re-read your own plan or todo list to confirm the next step is ' +
    'actually new work, (3) if neither helps, say so and ask the user for ' +
    'a course correction rather than repeating the same action again.'
  );
}

// ── Component 3: stall-recovery continuation probe ──────────────────────────

function getMessageType(message) {
  if (!message) return undefined;
  if (typeof message._getType === 'function') return message._getType();
  return message.type;
}

function extractMessageText(message) {
  const content = message?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((part) => part && (part.type === 'text' || typeof part.text === 'string'))
      .map((part) => part.text || '')
      .join('');
  }
  return '';
}

/**
 * True when `text` contains the DONE_TAG completion marker the model was
 * asked (via getStallProbeTagInstruction()) to append to a genuinely final
 * response. A plain substring check — the tag is distinctive enough not to
 * need anchoring to start/end.
 */
function containsDoneTag(text) {
  return typeof text === 'string' && text.includes(DONE_TAG);
}

/**
 * Continuation candidate check, applied to the messages a single
 * `run.processStream()` call just produced (`run.getRunMessages()` —
 * scoped to that one call, not the whole conversation; see the module
 * docstring's Component 3 section). True only when this run made at least
 * one real tool call (a ToolMessage is present), ended on a plain-text
 * AIMessage with no further tool_calls, AND that final message does not
 * carry the DONE_TAG completion marker — DONE_TAG absence is now the sole
 * and authoritative signal, no length-based override.
 *
 * A length-based fallback (fire only if the final text was also short) was
 * tried and then REMOVED after live testing found it caused the opposite
 * failure from the one it was meant to prevent: a genuinely INCOMPLETE
 * response (cut off mid-task, no closing summary) that happened to be
 * moderately long — 537 characters in the observed case, comfortably over
 * a 400-character threshold that was meant to catch only short narrated-
 * intent stalls — got silently treated as complete and never probed at
 * all, with no log trace to explain why (see
 * docs/39_stall_recovery_implementation_report.md §10 for the full
 * incident). A length threshold cannot reliably distinguish "genuinely
 * complete" from "genuinely incomplete but substantial" — both land in the
 * same few-hundred-to-few-thousand character range — so it was actively
 * undermining the tag, which is supposed to be the authoritative signal.
 * The residual risk this removal reopens (a genuinely complete, long
 * answer that simply forgot the tag triggers one unnecessary probe round)
 * is bounded by the auto-continue cap and self-corrects on the very next
 * round, since the probe message reinforces the same tag instruction — a
 * strictly smaller and more visible failure than the silent one this
 * caused.
 *
 * @param {Array<{_getType?: () => string, type?: string, tool_calls?: unknown[], content?: unknown}>} runMessages
 * @returns {boolean}
 */
function shouldFireStallProbe(runMessages) {
  if (!Array.isArray(runMessages) || runMessages.length === 0) return false;
  const hadToolCall = runMessages.some((m) => getMessageType(m) === 'tool');
  if (!hadToolCall) return false;
  const last = runMessages[runMessages.length - 1];
  if (getMessageType(last) !== 'ai') return false;
  const toolCalls = last?.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) return false;
  return !containsDoneTag(extractMessageText(last));
}

/**
 * True when `message` is the probe's reply AND it resolved to "nothing
 * further to do" — a plain-text AIMessage (no tool_calls) that carries the
 * DONE_TAG completion marker. Same tag, same check as shouldFireStallProbe
 * uses on the original response — one consistent signal applied uniformly
 * to every round, not a separate literal-"DONE" match. Any other AI reply
 * (including one that makes a new tool call, per shouldFireStallProbe's own
 * logic on the next round) counts as a real continuation, not a DONE
 * resolution.
 *
 * @param {{_getType?: () => string, type?: string, tool_calls?: unknown[], content?: unknown}} [message]
 * @returns {boolean}
 */
function isDoneMessage(message) {
  if (getMessageType(message) !== 'ai') return false;
  const toolCalls = message?.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) return false;
  return containsDoneTag(extractMessageText(message));
}

/**
 * Atomically-enough (single Node process, sequential within one request)
 * reserve one stall-probe attempt for `conversationId`: reads the current
 * `autoContinueCount`, and if under CWK_SUPERVISOR_STALL_PROBE_MAX_CONTINUATIONS,
 * increments and persists it and returns true. Returns false when the cap is
 * reached, the feature is disabled, or state read/write fails — fails CLOSED
 * (declines to probe) rather than open, since probing spends extra model/tool
 * calls, unlike components 1 and 2 which fail open by skipping a no-op context
 * injection.
 *
 * @param {string} [conversationId]
 * @returns {Promise<boolean>}
 */
async function tryReserveStallProbeSlot(conversationId, { toolCount } = {}) {
  if (!SUPERVISOR_ENABLED || !STALL_PROBE_ENABLED || !STALL_PROBE_REENTRY_ENABLED || !conversationId) {
    return false;
  }
  try {
    const state = await loadSupervisorState(conversationId);
    const unproductive = state?.consecutiveUnproductiveContinuations || 0;
    const total = state?.totalContinuations || 0;
    const lastToolCount = state?.lastToolCount || 0;

    /** Absolute ceiling first: it holds no matter how productive the agent is. */
    if (total >= STALL_PROBE_MAX_TOTAL_CONTINUATIONS) {
      return false;
    }

    /**
     * Progress signal: did the agent actually DO anything since the previous
     * continuation? Tool calls are the honest measure for this workload — the
     * observed stall is narrated intent with no tool call ("...then use
     * edit_block:"), while a round that really ran a test leaves ToolMessages
     * behind. Raw message count would be useless here: the nudge this feature
     * injects is itself a message, so the count grows every round by
     * construction. ToolMessages cannot be inflated that way.
     *
     * `toolCount` omitted (older callers, and the pure-function tests) means
     * "no progress information" and falls back to the flat-cap behaviour.
     */
    const madeProgress = typeof toolCount === 'number' && toolCount > lastToolCount;
    const nextUnproductive = madeProgress ? 0 : unproductive + 1;

    /** Only CONSECUTIVE unproductive rounds count against the main cap, so a
     *  long task that keeps stalling between real steps is carried to the end,
     *  while an agent that is genuinely spinning is cut off just as fast as
     *  before. Mirrors the manual habit this replaced: regenerate a couple of
     *  times, give up if it isn't moving. */
    if (!madeProgress && unproductive >= STALL_PROBE_MAX_CONTINUATIONS) {
      return false;
    }

    await saveSupervisorState(conversationId, {
      consecutiveUnproductiveContinuations: nextUnproductive,
      totalContinuations: total + 1,
      ...(typeof toolCount === 'number' && { lastToolCount: toolCount }),
    });
    return true;
  } catch (e) {
    logger.warn(`[harnessSupervisor] stall-probe reserve failed, declining to probe: ${e.message}`);
    return false;
  }
}

/**
 * Resets the per-conversation auto-continue counter. Call once per real
 * user-initiated turn (from `buildMessages`, alongside the other two
 * components) — NOT from inside the probe loop itself, or every internal
 * probe round would reset its own cap and the loop could never stop.
 *
 * @param {string} [conversationId]
 * @returns {Promise<void>}
 */
async function resetAutoContinueCounter(conversationId) {
  if (!conversationId) return;
  await saveSupervisorState(conversationId, {
    consecutiveUnproductiveContinuations: 0,
    totalContinuations: 0,
    lastToolCount: 0,
  });
}

/**
 * Context block establishing the DONE_TAG completion-marker convention,
 * pushed into `sharedRunContextParts` every turn alongside the other two
 * components (see `buildMessages` in client.js) — NOT into the agent's
 * persistent stored system prompt, so it only applies while the harness
 * supervisor / stall-probe feature is enabled and carries no effect on any
 * other agent or conversation. Establishing the convention here, in the
 * FIRST round's own context, costs nothing extra: `shouldFireStallProbe`
 * and `isDoneMessage` both read this same tag on every subsequent round, so
 * detection is one consistent check applied uniformly, not a separate
 * mechanism for "was the original response done" vs "was the probe's reply
 * done".
 *
 * @returns {string|null} null when the feature is disabled.
 */
function getStallProbeTagInstruction() {
  if (!SUPERVISOR_ENABLED || !STALL_PROBE_ENABLED) return null;
  return (
    '# Harness note — completion signal\n\n' +
    'When you have fully completed the current request and there is ' +
    `nothing further to do, end your response with the exact tag ${DONE_TAG} ` +
    'on its own line. If more work remains, do not include this tag — just ' +
    'continue working.'
  );
}

/**
 * Build the `Stop`-hook that drives stall-recovery continuation, or undefined
 * when the feature is off. Registered on the run's HookRegistry by
 * `createRun` and dispatched by the SDK at natural run completion; returning
 * `decision: 'block'` re-enters the stream IN PLACE, inside the same
 * `processStream()` call, so the continuation streams to the client exactly
 * like an ordinary extra model turn.
 *
 * This replaced an earlier design that called `run.processStream()` a SECOND
 * time from the host after the first returned. That could never render
 * correctly: `processStream`'s `finally` calls `Graph.clearHeavyState()`,
 * which wipes `contentData`/`contentIndexMap` (so the next call's content
 * indices restart at 0 and collide with content already on screen) and drops
 * the Graph's `handlerRegistry` (so the continuation never reaches the
 * client at all). `streamOptions.keepContent` does not help — it only guards
 * `resetValues()`, which runs long after the damage. See
 * `docs/41_stall_recovery_stop_hook_report.md` §2.
 *
 * The hook is deliberately the ONLY policy site: it reuses
 * `shouldFireStallProbe()` unchanged and `tryReserveStallProbeSlot()` for the
 * cap, so detection semantics are identical to every prior pass.
 *
 * One scoping difference worth knowing: the SDK hands the hook
 * `graph.getRunMessages()`, which — because a continuation never resets the
 * Graph's run boundary — accumulates across rounds rather than being just the
 * latest round's output. `shouldFireStallProbe` handles that correctly (it
 * tests the LAST message for plain-text-with-no-DONE_TAG and only asks
 * whether a tool call happened at all), and it makes the "did this turn use
 * tools" test span the whole turn instead of one round, which is the more
 * faithful reading. It also subsumes the old `isDoneMessage()` check on the
 * probe reply: one predicate now covers every round.
 *
 * @param {string} [conversationId]
 * @returns {((input: {messages?: unknown[], stopHookActive?: boolean}) => Promise<object>)|undefined}
 */
function createStallRecoveryStopHook(conversationId, { onContinue } = {}) {
  if (!SUPERVISOR_ENABLED || !STALL_PROBE_ENABLED || !STALL_PROBE_REENTRY_ENABLED || !conversationId) {
    return undefined;
  }
  const convLabel = conversationId.slice(-8);
  /** Per-run counter for the indicator. The hook closure is created once per
   *  `createRun`, so this counts continuations within THIS turn — which is the
   *  number worth showing — without a second Mongo read per round. */
  let continuationNumber = 0;
  return async function stallRecoveryStopHook(input) {
    if (!shouldFireStallProbe(input?.messages)) {
      /** Debug-level either way: a silent non-firing decision was previously
       *  diagnosable only by querying Mongo directly (docs/39 §10). */
      logger.debug(
        `[harnessSupervisor] conv=${convLabel} stall-recovery Stop hook: no continuation needed`,
      );
      return {};
    }
    /**
     * Progress signal for the cap: how many tool calls this turn has produced
     * SO FAR. Because a continuation never resets the run boundary, the
     * messages the SDK hands us accumulate across rounds, so this count only
     * grows when the agent actually did something — which is exactly what
     * distinguishes "stalled but making headway between stalls" from
     * "spinning". See tryReserveStallProbeSlot.
     */
    const toolCount = (input?.messages ?? []).filter((m) => getMessageType(m) === 'tool').length;
    /** Cap is authoritative and fails closed; see tryReserveStallProbeSlot. */
    const reserved = await tryReserveStallProbeSlot(conversationId, { toolCount });
    if (!reserved) {
      /** Distinguish the two ways of declining — "it was spinning" and "this
       *  turn has simply had enough continuations" are different diagnoses,
       *  and telling them apart from the logs alone was the point of docs/39
       *  §10's fix. */
      const state = await loadSupervisorState(conversationId).catch(() => undefined);
      const reason =
        (state?.totalContinuations || 0) >= STALL_PROBE_MAX_TOTAL_CONTINUATIONS
          ? `absolute per-turn ceiling reached (${STALL_PROBE_MAX_TOTAL_CONTINUATIONS})`
          : `no progress in ${STALL_PROBE_MAX_CONTINUATIONS} consecutive continuations`;
      logger.info(
        `[harnessSupervisor] conv=${convLabel} stall detected but not continuing — ${reason}`,
      );
      return {};
    }
    continuationNumber += 1;
    logger.info(
      `[harnessSupervisor] conv=${convLabel} stall detected — blocking Stop to continue ` +
        `(stopHookActive=${input?.stopHookActive === true}, toolCalls=${toolCount}, ` +
        `continuation=${continuationNumber})`,
    );
    /**
     * Surface the restart in the chat. Best-effort and deliberately never
     * allowed to block or fail the continuation: an indicator that couldn't be
     * drawn is a cosmetic loss, whereas throwing here would abort a recovery
     * that is otherwise ready to run.
     */
    if (typeof onContinue === 'function' && STALL_PROBE_INDICATOR) {
      try {
        await onContinue({
          text: STALL_PROBE_INDICATOR.replace('%d', String(continuationNumber)),
          continuationNumber,
        });
      } catch (e) {
        logger.warn(`[harnessSupervisor] conv=${convLabel} indicator emit failed: ${e.message}`);
      }
    }
    return { decision: 'block', reason: STALL_PROBE_MESSAGE };
  };
}

module.exports = {
  getProactiveMemoryContext,
  createStallRecoveryStopHook,
  trackAndDetectStuckPattern,
  shouldFireStallProbe,
  isDoneMessage,
  tryReserveStallProbeSlot,
  resetAutoContinueCounter,
  getStallProbeTagInstruction,
  STALL_PROBE_MESSAGE,
  DONE_TAG,
  STALL_PROBE_REENTRY_ENABLED,
};
