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
 * [cowork] 2026-11-13. Legacy gate kept behind a flag: when true, restore the
 * old precondition that a turn must contain >=1 LangChain 'tool'-typed message
 * before the stall probe may fire. Default false — see shouldFireStallProbe for
 * why the precondition was removed (blind to symptom-B drift; unreliable here
 * because tool results ride assistant content parts rather than ToolMessages).
 */
const STALL_PROBE_REQUIRE_TOOL_CALL =
  (process.env.CWK_SUPERVISOR_STALL_PROBE_REQUIRE_TOOL_CALL ?? 'false').toLowerCase() === 'true';
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
  `You stopped without signalling completion. If the work is genuinely done ` +
    `and nothing is missing, reply exactly ${DONE_TAG}. Otherwise continue: ` +
    'call the next tool for the next step — describing an intended action ' +
    'instead of executing it is a failure mode. If a todo/task-list tool is ' +
    'available, keep it current while you work (mark the active step ' +
    'in_progress, completed steps completed) and finish every item before ' +
    `signalling; when all items are completed, clear or close the list, then ` +
    `end your final response with ${DONE_TAG} on its own line.`;

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

/**
 * Text shown in the chat when proactive work-graph/chat injection actually
 * fires — i.e. getProactiveMemoryContext() found something and pushed it into
 * sharedRunContextParts, not on every turn regardless of outcome. Set to an
 * empty string to suppress the indicator while leaving injection itself on:
 * the same "turn it on to check, off once trusted" switch as
 * CWK_SUPERVISOR_STALL_PROBE_INDICATOR above.
 *
 * Rendered as an `activity_label` content part, NOT as text in the response —
 * same reasoning as STALL_PROBE_INDICATOR: `packages/api/src/agents/client.ts`
 * classifies activity labels as "UI-only progress headers — never model
 * input, never billed output", so the marker can't leak into a later turn's
 * context or be charged as output tokens. This closes the visibility gap
 * docs/42_proactive_injection_visibility_report.md was written to close:
 * without it, "fired and came back thin" and "never fired" look identical
 * from the chat. The first `%d` is the work-graph hit count, the second the
 * chat hit count.
 */
const INJECTION_INDICATOR =
  process.env.CWK_SUPERVISOR_INJECTION_INDICATOR ??
  'Consulted memory + work graph for this task (%d work-graph, %d chat results)';

/**
 * [cowork] Component 4: tool-voice reminder (2026-11).
 * Long-context symptom observed on Qwen3.8-Flash-Next: after 2-3 purely
 * conversational assistant turns in a tool-using conversation, the model's
 * prior for emitting tool calls decays (in-context behavioral momentum) —
 * it narrates intended actions instead of executing them. A turn reminding
 * it restores the behavior; re-running the same message sometimes does,
 * which shows the prior dipped rather than vanished. Prevention layer for
 * exactly the failure the stall probe cannot catch BY DESIGN: the probe
 * requires >=1 tool call in the turn (docs 2026-08-stall-recovery §2), so a
 * tool-free turn that SHOULD have contained a call is invisible to it.
 * Fires only when BOTH hold: (a) the most recent TOOL_VOICE_GAP assistant
 * turns contain no tool call, (b) a tool call exists within the previous
 * TOOL_VOICE_RECENT_TURNS assistant turns — pure-chat conversations are
 * therefore never nagged. Stateless predicate over the reduced payload
 * (no Mongo, no embedding, no latency). See docs/changes/2026-11-tool-voice-reminder.md.
 */
const TOOL_VOICE_ENABLED =
  (process.env.CWK_TOOL_VOICE_ENABLED ?? 'true').toLowerCase() !== 'false';
const TOOL_VOICE_GAP = parseInt(process.env.CWK_TOOL_VOICE_GAP ?? '2', 10);
const TOOL_VOICE_RECENT_TURNS = parseInt(
  process.env.CWK_TOOL_VOICE_RECENT_TURNS ?? '10',
  10,
);

/**
 * [cowork] Component 4b (S1, 2026-11-16): claim-based trigger variant for the
 * tool-voice reminder. Spike finding F5: the GAP=2 trailing-tool-free
 * condition misses real fabrication points — contamination rides INSIDE turns
 * that also carry calls, so condition (a) never sees it. Additional fire path
 * (the existing one stays): any of the last TOOL_VOICE_CLAIM_WINDOW assistant
 * messages that carries no tool_calls AND past-tense completion claims in its
 * text. False-positive guard built into the condition: only messages WITHOUT
 * own-turn tool_calls can count, so "I ran the tests" riding the very turn
 * that ran them is never flagged. Default OFF until validated (ship dark,
 * same rollout discipline as components 4–6). See
 * docs/changes/2026-11-tool-voice-reminder.md §9.
 */
const TOOL_VOICE_CLAIM_ENABLED =
  (process.env.CWK_TOOL_VOICE_CLAIM_ENABLED ?? 'false').toLowerCase() === 'true';
const TOOL_VOICE_CLAIM_WINDOW = parseInt(
  process.env.CWK_TOOL_VOICE_CLAIM_WINDOW ?? '4',
  10,
);
// Stateless anti-nag cap (S6b), mirroring TOOL_ERROR_VOICE_MAX_STREAK's
// reasoning: while claim-drift persists, the trailing run of tool-free
// claiming assistant messages grows monotonically (1,2,3,...) and exactly one
// reminder rode each preceding request, so streak > MAX implies MAX nudges
// already delivered. Suppressing above the cap reproduces "stop nagging after
// N" without state. Caveat (undershoot, documented honestly as in component
// 6): several assistant rounds inside ONE user turn advance the streak
// without an intervening request, so the implied nudge count can undershoot;
// the cap then errs toward silence, the safe direction.
const TOOL_VOICE_CLAIM_MAX_STREAK = parseInt(
  process.env.CWK_TOOL_VOICE_CLAIM_MAX_STREAK ?? '3',
  10,
);

/**
 * [cowork] Component 7 (S5, 2026-11-16): claim-vs-call reconciliation.
 * Post-run check attacking the contamination SOURCE (findings F3/F4): a
 * fabricated completion claim entering history poisons later turns regardless
 * of depth, and it is invisible to the proxy (no markup to parse) and to the
 * stall probe BY DESIGN when the fabrication closes with the DONE tag or a
 * question (both exempt the probe). Fires at run end via the Stop hook: final
 * assistant message carries strong completion claims while the run contains
 * ZERO tool calls → inject a corrective message once through the existing
 * stall-probe slot machinery (its Mongo caps give once-per-round and
 * per-turn-ceiling safety for free). Summarize-guard: ANY tool call anywhere
 * in the run suppresses — we cannot map a claim to a specific call, so the
 * possibility that the model legitimately summarizes past work wins. Default
 * OFF until validated. See docs/changes/2026-11-claim-reconciliation.md.
 */
const CLAIM_RECONCILE_ENABLED =
  (process.env.CWK_CLAIM_RECONCILE_ENABLED ?? 'false').toLowerCase() === 'true';

/** Corrective injection body (component 7). Names the mismatch factually and
 *  offers exactly two exits — produce the call, or retract the claim. */
const CLAIM_RECONCILE_MESSAGE =
  '[CLAIM RECONCILIATION] Your closing message states that work was completed, ' +
  'but no tool call performed it in this turn or the previous one. Either make ' +
  'the actual tool call now, or correct your statement and say plainly what has ' +
  'not been done. Do not repeat the claim unchanged.';

/**
 * [cowork] Component 6: tool-error voice (2026-11).
 * The served chat template used to be a community fork
 * (qwen3.8-froggeric-v22.5) that classified consecutive tool errors WHILE
 * RENDERING and injected "⚠️ SYSTEM WARNING" text inside tool_response
 * blocks. Two structural flaws: re-rendered history makes every warning
 * permanent (a turn-5 failure scolds the model hundreds of turns later), and
 * the warning's authorship is invisible in transcripts (looks like tool
 * output). Those nudges now belong HERE: injected per-request into the
 * ephemeral proactiveContextMessage slot, so they expire the moment the
 * error streak ends. Template side patched separately
 * (prompts/qwen38-chat-template-cwk-patched.jinja, warnings removed) — one
 * owner per nudge class. Classifier below is a faithful port of the removed
 * Jinja logic (/Users/erik/Downloads/chat_template.jinja.txt lines 413-424).
 * See docs/changes/2026-11-tool-error-voice.md.
 */
const TOOL_ERROR_VOICE_ENABLED =
  (process.env.CWK_TOOL_ERROR_VOICE_ENABLED ?? 'true').toLowerCase() !== 'false';
// Stateless anti-nag cap: during an ongoing error run the trailing streak
// grows monotonically (1,2,3,...) and exactly one nudge rides each request,
// so streak > MAX_STREAK implies MAX_STREAK nudges were already delivered in
// preceding turns of the same run. Suppressing above the cap therefore
// reproduces "stop nagging after N consecutive nudges" without any state.
// Caveat (documented honestly): several tool batches inside ONE assistant
// turn advance the streak without an intervening request, so the implied
// nudge count can undershoot; the cap then errs toward silence, which is
// the safe direction.
const TOOL_ERROR_VOICE_MAX_STREAK = parseInt(
  process.env.CWK_TOOL_ERROR_VOICE_MAX_STREAK ?? '3',
  10,
);

/**
 * [cowork] Component 5: todo-list voice (2026-11). Agents use the scoped
 * tasklist sparsely even though its own tool description asks for it — tool
 * descriptions lose weight mid-conversation, the same discount docs/42 showed
 * for the system-prompt tail. Fix rides the proven per-turn instruction slot:
 * getStallProbeTagInstruction() (pushed into sharedRunContextParts every turn)
 * gains a compact progress-tracking section with one REAL example call
 * (registry name todo_update_mcp_todo_list, schema from mcp/todo_list/server.js
 * v2), phrased conditionally ("if you have such a tool") so agents without
 * the tool attached are never told to call something they lack. The
 * continuation probe message carries the same reminder — see
 * STALL_PROBE_MESSAGE. See docs/changes/2026-11-tool-voice-reminder.md §7.
 */
const TODO_VOICE_ENABLED =
  (process.env.CWK_TODO_VOICE_ENABLED ?? 'true').toLowerCase() !== 'false';

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
async function getProactiveMemoryContext({ userMessage, conversationId, userId, onInjected }) {
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
        // Person-knowledge lifecycle: assumptions (inferred, not yet confirmed by
        // recurrence) are no longer hard-excluded here. They surface in proactive
        // context like any other entry and carry a rank penalty instead
        // (ASSUMPTION_RANK_FACTOR in mcp/memory/lib/hybrid-search.js): a wrong
        // assumption visible in conversation gets corrected on the spot, which
        // beats a manual review cycle. No filter key is sent for this leg now.
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
      '# Instruction: relevant background was found automatically — check it before you respond',
      '',
      'The system searched memory and the work graph based on your message above ' +
        '(not something you asked for) and found the items below. Before writing ' +
        'your response: decide whether any of this is actually relevant. If it ' +
        'is, use it AND explicitly tell the user what you found and how it ' +
        'applies — do not silently factor it into your reasoning without ' +
        'surfacing it. If none of it is relevant, ignore it and answer normally; ' +
        'do not force a connection that is not there. For a deeper or ' +
        'differently-phrased look than what is shown here, call ' +
        '`search_work_graph` or `search_memory` directly.',
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

    /**
     * Surface the injection in the chat — same best-effort, never-blocking
     * contract as the stall-recovery indicator: a marker that fails to draw
     * is a cosmetic loss, not a reason to drop content that already passed
     * the relevance threshold.
     */
    if (typeof onInjected === 'function' && INJECTION_INDICATOR) {
      try {
        await onInjected({
          text: INJECTION_INDICATOR.replace('%d', String(workGraphHits.length)).replace(
            '%d',
            String(chatHits.length),
          ),
          workGraphCount: workGraphHits.length,
          chatCount: chatHits.length,
        });
      } catch (e) {
        logger.warn(
          `[harnessSupervisor] conv=${convLabel} injection indicator emit failed: ${e.message}`,
        );
      }
    }
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
 * True when the FINAL paragraph of `text` reads as a question directed at
 * the user — "which approach do you want?", "ska jag göra X eller Y?" — i.e.
 * the model is legitimately waiting for a decision, not stalled. Absence of
 * DONE_TAG alone can't distinguish these: neither case is "done," but only
 * one should be auto-continued. Observed live (2026-08-26): a genuine
 * clarifying question got nudged to "continue your tasks," which the model
 * then read as permission to proceed without an answer, assuming the user
 * either didn't know or didn't want to respond.
 *
 * Checked on the LAST paragraph only (split on blank lines) so an earlier
 * rhetorical question in a long answer can't suppress a genuine stall later
 * in the same response — only the actual last thing the model said matters.
 * A trailing `?` (optionally wrapped in closing markdown/quote punctuation)
 * is deliberately a blunt, cheap check in the same spirit as DONE_TAG itself
 * — not sentence-level NLP, which is more machinery than this needs.
 */
const TRAILING_QUESTION_PATTERN = /[?？][)\]"'*_~`]*$/;
function endsWithQuestionToUser(text) {
  if (typeof text !== 'string') return false;
  const paragraphs = text.trim().split(/\n\s*\n/);
  const lastParagraph = paragraphs[paragraphs.length - 1]?.trim() ?? '';
  return TRAILING_QUESTION_PATTERN.test(lastParagraph);
}

/**
 * Continuation candidate check, applied to the messages a single
 * `run.processStream()` call just produced (`run.getRunMessages()` —
 * scoped to that one call, not the whole conversation; see the module
 * docstring's Component 3 section). True only when this run made at least
 * one real tool call (a ToolMessage is present), ended on a plain-text
 * AIMessage with no further tool_calls, AND that final message does not
 * carry the DONE_TAG completion marker, AND that final message does not
 * read as a question directed at the user (see endsWithQuestionToUser) —
 * DONE_TAG absence is the primary signal, no length-based override, but a
 * trailing question is a second, independent reason not to fire: the model
 * isn't stalled, it's waiting for the user's answer, and nudging it to
 * "continue" there is actively wrong (2026-08-26 live finding).
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
  // [cowork] 2026-11-13: the former `>=1 'tool'-typed message` precondition
  // is OFF by default. It made the probe structurally blind to the drift
  // failure (a tool-free turn that should have contained a call — symptom B)
  // and, worse, unreliable on this architecture: tool results ride assistant
  // content parts, not LangChain ToolMessages, so `getMessageType(m)==='tool'`
  // can stay false even after genuine tool work (suspected cause of the
  // 2026-11-13 live miss). DONE_TAG presence/absence plus the final-question
  // exemption carry the whole decision now — the tag instruction is injected
  // EVERY turn, so even pure-chat answers are covered by the same contract.
  // Residual cost: a complete pure-chat reply that forgot the tag burns one
  // bounded probe round and self-corrects (same trade-off already accepted
  // for the removed length gate, docs 2026-08-stall-recovery §8). Set
  // CWK_SUPERVISOR_STALL_PROBE_REQUIRE_TOOL_CALL=true for legacy behavior.
  if (STALL_PROBE_REQUIRE_TOOL_CALL) {
    const hadToolCall = runMessages.some((m) => getMessageType(m) === 'tool');
    if (!hadToolCall) return false;
  }
  const last = runMessages[runMessages.length - 1];
  if (getMessageType(last) !== 'ai') return false;
  const toolCalls = last?.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) return false;
  const text = extractMessageText(last);
  if (containsDoneTag(text)) return false;
  return !endsWithQuestionToUser(text);
}

/**
 * [cowork] Component 7 predicate (S5): did this run END on a fabricated
 * completion claim? Operates on the Stop hook's accumulated runMessages
 * (whole user turn, all rounds). True only when ALL hold:
 *   - flag CWK_CLAIM_RECONCILE_ENABLED is on (default off);
 *   - the final message is an AI reply with no tool_calls;
 *   - NO tool call exists anywhere in the run (any typed ToolMessage, any
 *     tool_calls array, any tool_call content part). This single check IS
 *     the summarize-guard: if the run did real work, the closing claim may
 *     legitimately summarize it, so silence wins. With zero calls in the
 *     whole turn, "final turn clean AND previous turn clean" holds trivially.
 *   - the final text carries a STRONG completion claim (weak tier skipped —
 *     honest summaries hit weak phrasings constantly; see classifier doc).
 * Deliberately NOT gated on DONE_TAG / trailing question: those exempt the
 * stall probe, and F4 fabrications frequently carry both — that exemption is
 * precisely the blind spot this component exists to cover. When BOTH this
 * and the stall probe qualify, the probe runs first (more general nudge);
 * see createStallRecoveryStopHook.
 * @param {Array<object>} runMessages
 * @returns {boolean}
 */
function shouldReconcileClaims(runMessages) {
  if (!SUPERVISOR_ENABLED || !CLAIM_RECONCILE_ENABLED) return false;
  if (!Array.isArray(runMessages) || runMessages.length === 0) return false;
  const last = runMessages[runMessages.length - 1];
  if (getMessageType(last) !== 'ai') return false;
  const toolCalls = last?.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) return false;
  const anyToolWork = runMessages.some(
    (m) =>
      getMessageType(m) === 'tool' ||
      assistantHasToolCall(m),
  );
  if (anyToolWork) return false;
  return hasCompletionClaimContent(extractMessageText(last), { strongOnly: true });
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
/**
 * Cheap normalization for round-over-round text comparison: collapse
 * whitespace and cap length, so trivial formatting drift doesn't defeat the
 * repeated-text check and a pathologically long reply doesn't bloat Mongo
 * state. Not a similarity metric — exact match on the normalized string only.
 */
function normalizeResponseText(text) {
  if (typeof text !== 'string') return '';
  return text.trim().replace(/\s+/g, ' ').slice(0, 500);
}

async function tryReserveStallProbeSlot(conversationId, { toolCount, responseText } = {}) {
  if (!SUPERVISOR_ENABLED || !STALL_PROBE_ENABLED || !STALL_PROBE_REENTRY_ENABLED || !conversationId) {
    return false;
  }
  try {
    const state = await loadSupervisorState(conversationId);
    const unproductive = state?.consecutiveUnproductiveContinuations || 0;
    const total = state?.totalContinuations || 0;
    const lastToolCount = state?.lastToolCount || 0;
    const lastResponseText = state?.lastResponseText || '';

    /** Absolute ceiling first: it holds no matter how productive the agent is. */
    if (total >= STALL_PROBE_MAX_TOTAL_CONTINUATIONS) {
      return false;
    }

    /**
     * Progress signal: did the agent actually DO anything since the previous
     * continuation? Tool calls are the primary measure for this workload — the
     * observed stall is narrated intent with no tool call ("...then use
     * edit_block:"), while a round that really ran a test leaves ToolMessages
     * behind. Raw message count would be useless here: the nudge this feature
     * injects is itself a message, so the count grows every round by
     * construction. ToolMessages cannot be inflated that way.
     *
     * Tool count alone is not sufficient, though: a model that keeps issuing
     * NEW but equally unproductive tool calls (e.g. retrying a search with
     * different phrasing, never finding anything) looks like it's progressing
     * by this measure alone, while its own concluding text repeats round over
     * round — observed live (docs/42 §diagnosis-followup) as several
     * consecutive continuations closing on byte-identical apology text while
     * still burning fresh `web_search` calls each round. `responseText` is
     * the stalled round's own final message (the same text
     * `shouldFireStallProbe` just read to decide THIS round counts as a
     * stall) — an exact repeat of the previous round's stalled text means no
     * real progress regardless of tool count.
     *
     * `toolCount`/`responseText` omitted (older callers, and the pure-function
     * tests) means "no progress information" and falls back to the flat-cap
     * behaviour.
     */
    const madeProgressByTools = typeof toolCount === 'number' && toolCount > lastToolCount;
    const normalizedResponseText = normalizeResponseText(responseText);
    const isRepeatedText =
      normalizedResponseText.length > 0 && normalizedResponseText === lastResponseText;
    const madeProgress = madeProgressByTools && !isRepeatedText;
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
      ...(normalizedResponseText.length > 0 && { lastResponseText: normalizedResponseText }),
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
  const completion =
    '# Harness note — completion signal\n\n' +
    'When you have fully completed the current request and there is ' +
    `nothing further to do, end your response with the exact tag ${DONE_TAG} ` +
    'on its own line. If more work remains, do not include the tag — just ' +
    'continue working. If instead you are asking a question or need the ' +
    'user\'s decision before continuing, end normally (with the question) ' +
    'and omit the tag.';
  // [cowork] Component 5: todo-list voice. Exact callable names ride the
  // instruction because tool descriptions lose weight mid-conversation and
  // agents were observed searching the registry for the tasklist tool at
  // runtime instead of using it (2026-11-13 live check). Conditional wording:
  // agents without these tools attached must not be told to call them.
  const tracking = TODO_VOICE_ENABLED
    ? '\n\n# Progress tracking\n\n' +
      'For any multi-step task keep a visible task list with the scoped ' +
      'tasklist MCP tools. Use these EXACT tool names (searching the ' +
      'registry for them wastes a turn):\n' +
      '- `todo_update_mcp_todo_list` — full snapshot every time; fields per ' +
      'item: content (imperative), activeForm (present continuous), status ' +
      '(pending | in_progress | completed)\n' +
      '- `todo_get_mcp_todo_list` — read the current list\n' +
      '- `todo_clear_mcp_todo_list` — dismiss the widget once every item is ' +
      'completed\n\n' +
      'Create the list before the first real step, resend the updated ' +
      'snapshot after each step, keep exactly one item in_progress while ' +
      'working, set every item completed when finished, then clear the list ' +
      `before ending with ${DONE_TAG}. Example first call: {"todos": [{"content": "Run tests", "activeForm": "Running tests", "status": "in_progress"}, {"content": "Write report", "activeForm": "Writing report", "status": "pending"}]}`
    : '';
  return completion + tracking;
}

/**
 * [cowork] True when an assistant message (reduced-payload shape, see
 * contextReduction.js header) contains at least one tool_call content part.
 * Checks both the parts-array form ({ type: 'tool_call' }) and a plain
 * message.tool_calls array, so the predicate works on either representation.
 */
function assistantHasToolCall(message) {
  if (!message) return false;
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return true;
  }
  const content = message.content;
  if (Array.isArray(content)) {
    return content.some(
      (part) => part && typeof part === 'object' && part.type === 'tool_call',
    );
  }
  return false;
}

/**
 * [cowork] Completion-claim classifier (component 4b / component 7, S1).
 * Same tiered style as component 6's isToolErrorContent: strong signals fire
 * regardless; weak signals suppress on code-ish content or total length ≥600
 * (long summaries quote past work incidentally — same reasoning as the error
 * classifier's long-output suppression). Targets PAST-TENSE completion
 * claims ("I ran the tests", "Live docs updated…") — exactly the F4 shape.
 * Deliberately narrow: this gates a corrective injection, so false positives
 * cost an extra model round and erode trust in the harness voice.
 * @param {unknown} raw assistant message body (string or multipart array)
 * @param {{strongOnly?: boolean}} [opts] strongOnly skips the weak tier —
 *   component 7 uses it because honest end-of-run summaries hit the weak
 *   phrasings constantly, and a false positive there costs a model round.
 * @returns {boolean} true when the text asserts completed actions
 */
function hasCompletionClaimContent(raw, { strongOnly = false } = {}) {
  let content;
  if (typeof raw === 'string') {
    content = raw;
  } else if (raw == null) {
    content = '';
  } else if (Array.isArray(raw)) {
    content = raw
      .map((p) => (typeof p === 'string' ? p : p?.text ?? ''))
      .join('\n');
  } else {
    try {
      content = JSON.stringify(raw);
    } catch {
      content = String(raw);
    }
  }
  const lower = content.toLowerCase();
  // Strong: first-person past-tense action verbs, proof-quoting phrasing.
  const strongClaim =
    /\bi (just |have |had )?(ran|wrote|updated|created|edited|called|used|executed|appended|fixed|deployed)\b/.test(
      lower,
    ) ||
    /"proof:"/.test(lower) ||
    /grep (shows|output confirms)/.test(lower) ||
    // F4 verbatim shapes from the spike transcript ("Live docs updated…").
    /(docs?|file|scripts?|tests?) (were |has been |have been )?updated\b/.test(lower);
  if (strongClaim) return true;
  // Weak: bare phrasings that legitimate summaries use constantly.
  if (strongOnly) return false;
  const weakClaim = /updated .*\.md/.test(lower) || /tests? pass/.test(lower);
  if (!weakClaim) return false;
  const isCodeish =
    lower.includes('```') ||
    lower.includes('$ ') ||
    lower.includes('exit code') ||
    lower.includes('traceback');
  return !isCodeish && content.length < 600;
}

/**
 * [cowork] Tool-voice decision (component 4 + claim variant 4b): returns
 * `{ text, reason }` with reason 'gap' | 'claim', or null when silent.
 *
 * Fire path A (original, unchanged): scan assistant messages from the end;
 * fire only when the last TOOL_VOICE_GAP of them are all tool-free AND a tool
 * call appears within the next TOOL_VOICE_RECENT_TURNS assistant messages
 * behind them. Consequences: a conversation that never used tools stays
 * silent (condition b fails); a healthy tool-using conversation stays silent
 * (condition a fails); only the drift pattern nags.
 *
 * Fire path B (S1, flag CWK_TOOL_VOICE_CLAIM_ENABLED, default off): any of
 * the last TOOL_VOICE_CLAIM_WINDOW assistant messages carries no tool_calls
 * AND completion claims (see hasCompletionClaimContent), plus condition (b)
 * still required (recent real tool use exists — pure chat never nagged). The
 * own-turn guard is built in: claiming messages WITH their own calls cannot
 * count. Anti-nag cap TOOL_VOICE_CLAIM_MAX_STREAK over the trailing run of
 * tool-free claiming messages (stateless, see constant comment).
 *
 * Text design rules (learned the hard way during drafting, 2026-11-13;
 * S2 hardened after F5 measured example imitation live): injected examples
 * are imitated with full fidelity — so the reminder now names NO callable
 * registry tool and carries NO JSON payload at all. The todo-list example
 * lives exclusively in component 5 (different failure surface, its job IS
 * teaching that shape).
 *
 * @param {Array<{role?: string, content?: unknown, tool_calls?: unknown[]}>} payloadMessages
 * @returns {{text: string, reason: 'gap'|'claim'}|null}
 */
function decideToolVoice(payloadMessages) {
  if (!SUPERVISOR_ENABLED || !TOOL_VOICE_ENABLED) return null;
  if (!Array.isArray(payloadMessages) || payloadMessages.length === 0) {
    return null;
  }
  const assistants = payloadMessages.filter((m) => m?.role === 'assistant');
  if (assistants.length < TOOL_VOICE_GAP + 1) return null;

  const buildText = () =>
    '# Harness note — tool use\n\n' +
    'You have tools available and used them earlier in this conversation. ' +
    'If this request involves reading files, running commands, fetching ' +
    'data, or changing anything, act with a tool call instead of describing ' +
    'what you would do. Narrating intended actions without executing them ' +
    'is a failure mode. Never state that an action was completed unless the ' +
    'tool call performing it appears in your own previous turns. Before ' +
    'concluding a tool does not exist, search for it by exact name once more. ' +
    'A correct call names an available tool exactly and carries every ' +
    'required argument.';

  // Shared condition (b): recent REAL tool use (same span as path A's
  // lookback, measured over the last RECENT_TURNS assistant messages). Both
  // triggers need it: without it a pure-chat conversation whose participants
  // merely talk about past work would be nagged, and stale tool use means
  // this is a chat again, not a drifting agent.
  const hasRecentToolUse = assistants
    .slice(-TOOL_VOICE_RECENT_TURNS)
    .some(assistantHasToolCall);

  // Path B (evaluated FIRST): claim trigger (S1). Ordering rationale: the
  // claim diagnosis is the more specific reading of the same drift (mirror
  // of the error-voice-wins rule), and evaluating it first keeps the
  // stateless cap reachable — with path A first, every sustained-drift case
  // is answered identically by the uncapped gap path, leaving the cap dead
  // code. Window of the last CLAIM_WINDOW assistant messages; a message
  // counts when it lacks tool_calls AND claims past completions.
  if (TOOL_VOICE_CLAIM_ENABLED && hasRecentToolUse) {
    const window = assistants.slice(-TOOL_VOICE_CLAIM_WINDOW);
    const claiming = window.filter(
      (m) => !assistantHasToolCall(m) && hasCompletionClaimContent(m.content),
    );
    if (claiming.length > 0) {
      // Stateless anti-nag cap: trailing run of tool-free claiming
      // messages (newest-first walk, stop at first non-matching).
      let streak = 0;
      for (let i = assistants.length - 1; i >= 0; i--) {
        const m = assistants[i];
        if (!assistantHasToolCall(m) && hasCompletionClaimContent(m.content)) {
          streak++;
        } else break;
      }
      if (streak <= TOOL_VOICE_CLAIM_MAX_STREAK) {
        return { text: buildText(), reason: 'claim' };
      }
      // Over cap: go fully silent — including suppressing path A, whose
      // uncapped re-firing here would defeat the cap in exactly its target
      // scenario (sustained claim-drift). Pure-chat drift (no claims)
      // never reaches this branch and keeps path A's original dynamics.
      return null;
    }
  }

  // Path A: gap trigger (unchanged semantics — window-scoped lookback).
  const gapSlice = assistants.slice(-TOOL_VOICE_GAP);
  if (!gapSlice.some(assistantHasToolCall)) {
    const lookback = assistants.slice(
      -(TOOL_VOICE_GAP + TOOL_VOICE_RECENT_TURNS),
      -TOOL_VOICE_GAP,
    );
    if (lookback.some(assistantHasToolCall)) {
      return { text: buildText(), reason: 'gap' };
    }
  }
  return null;
}

/**
 * [cowork] Tool-voice reminder text (component 4), or null when it should not
 * fire. Thin back-compat wrapper over decideToolVoice() — kept because client
 * wiring and existing suites import the original name.
 *
 * @param {Array<{role?: string, content?: unknown, tool_calls?: unknown[]}>} payloadMessages
 * @returns {string|null}
 */
function getToolVoiceReminder(payloadMessages) {
  const decision = decideToolVoice(payloadMessages);
  return decision ? decision.text : null;
}

/**
 * [cowork] Tool-error classifier — faithful JS port of the consecutive-error
 * classifier removed from the community chat template fork
 * (source of truth: /Users/erik/Downloads/chat_template.jinja.txt lines
 * 413-424, variables _content_head / _is_code_or_grep / _exit_code_zero /
 * _error_field_ok / _strong_error / _weak_error / _weak_suppressed). Three
 * tiers, checked against the LOWERCASED FIRST 120 CHARS exactly as Jinja did:
 *   - strong signals: fire regardless of suppression unless code-excluded
 *   - weak signals ('error:', 'err!'): suppressed when '$ ' or 'took '
 *     present or total content length >= 600 (long outputs quote errors
 *     incidentally; shell echoes and timing lines are their own tell)
 *   - code-content exclusion: discussing errors in code is not failing with
 *     one
 * @param {unknown} raw a tool result body (usually a string)
 * @returns {boolean} true when the body reads as a tool FAILURE
 */
function isToolErrorContent(raw) {
  let content;
  if (typeof raw === 'string') {
    content = raw;
  } else if (raw == null) {
    content = '';
  } else if (Array.isArray(raw)) {
    // role:'tool' multipart content: join text parts (Jinja assumed a plain
    // string; this is the closest faithful generalisation).
    content = raw
      .map((p) => (typeof p === 'string' ? p : p?.text ?? ''))
      .join('\n');
  } else {
    try {
      content = JSON.stringify(raw);
    } catch {
      content = String(raw);
    }
  }
  const lower = content.toLowerCase();
  const head = lower.slice(0, 120);
  const isCodeOrGrep =
    lower.includes('throw new ') ||
    lower.includes('throw error') ||
    lower.includes('console.error') ||
    lower.includes('logger.error') ||
    lower.includes('logging.error') ||
    head.startsWith('import ') ||
    head.startsWith('def ') ||
    head.startsWith('function ');
  const exitCodeZero =
    head.includes('exit code: 0') || head.includes('process exited with code 0');
  const errorFieldOk =
    head.includes('"error": null') ||
    head.includes('"error":null') ||
    head.includes('"error": false') ||
    head.includes('"error":false') ||
    head.includes('"error": ""') ||
    head.includes('"error":""');
  const strongError =
    (head.includes('"error":') && !errorFieldOk) ||
    head.includes('"status": "error"') ||
    head.includes('"status":"error"') ||
    head.includes('traceback (most recent call last):') ||
    head.includes('command not found') ||
    head.includes('invalid syntax') ||
    head.includes('fatal:') ||
    ((head.includes('exit code: ') || head.includes('process exited with code')) &&
      !exitCodeZero) ||
    head.startsWith('exception:') ||
    head.startsWith('failed to ');
  const weakError = head.includes('error:') || head.includes('err!');
  const weakSuppressed =
    head.includes('$ ') || head.includes('took ') || content.length >= 600;
  return !isCodeOrGrep && (strongError || (weakError && !weakSuppressed));
}

/**
 * Collect the TRAILING run of tool-result bodies from the reduced payload,
 * newest first, stopping at the first event that closes the run. Handles both
 * transports a tool result can arrive as:
 *   a) role:'tool' messages (OpenAI-style replay / imported threads), and
 *   b) the native agents-pipeline shape, where results ride INSIDE assistant
 *      messages as { type:'tool_call', tool_call:{ ..., output } } content
 *      parts (see contextReduction.js header — there are no role:'tool'
 *      messages natively, so a literal role-only scan would never fire).
 * Run-closing events (stop the walk, streak considered closed):
 *   - assistant text/think content AFTER its last tool_call part: the model
 *     already saw the results and replied, whatever it concluded is history;
 *   - a plain assistant message without tool calls: same reason;
 *   - any user/system message between tool rounds.
 * Leading user messages (the CURRENT request rides at the end of the payload)
 * are skipped first: the predicate judges what the model INHERITS from the
 * previous run, not the fresh input.
 * @param {Array<object>} payloadMessages
 * @returns {unknown[]} tool-result bodies, newest first (possibly empty)
 */
function collectTrailingToolResults(payloadMessages) {
  const newestFirst = [];
  let i = payloadMessages.length - 1;
  while (
    i >= 0 &&
    payloadMessages[i]?.role !== 'assistant' &&
    payloadMessages[i]?.role !== 'tool'
  ) {
    i--;
  }
  while (i >= 0) {
    const m = payloadMessages[i];
    if (m?.role === 'tool') {
      newestFirst.push(m.content);
      i--;
      continue;
    }
    if (m?.role === 'assistant') {
      const parts = Array.isArray(m.content) ? m.content : null;
      if (!parts) break; // plain string reply → model already moved on
      let lastTc = -1;
      for (let p = parts.length - 1; p >= 0; p--) {
        if (parts[p]?.type === 'tool_call') {
          lastTc = p;
          break;
        }
      }
      if (lastTc === -1) break; // no tool calls in this message
      const answeredAfter = parts
        .slice(lastTc + 1)
        .some(
          (p) =>
            (p?.type === 'text' && typeof p.text === 'string' && p.text.trim()) ||
            (p?.type === 'think' && typeof p.think === 'string' && p.think.trim()),
        );
      if (answeredAfter) break; // replied after the batch → run closed
      for (let p = parts.length - 1; p >= 0; p--) {
        if (parts[p]?.type === 'tool_call') {
          newestFirst.push(parts[p]?.tool_call?.output);
        }
      }
      i--;
      continue;
    }
    break; // user/system boundary
  }
  return newestFirst;
}

/**
 * [cowork] Component 6: tool-error voice (component-4 sibling). Returns the
 * nudge text or null. Scan the trailing tool-result run (see
 * collectTrailingToolResults); count TRAILING consecutive error-classified
 * results (n) using the ported classifier. n == 1 → one-shot diagnosis
 * nudge; n >= 2 → escalation demanding a fundamentally different approach
 * (wording inherited from the removed template warnings, reframed with
 * visible harness authorship — transcript forensics was half the reason the
 * template approach was abandoned). n > CWK_TOOL_ERROR_VOICE_MAX_STREAK →
 * null (stateless anti-nag cap, see constant comment). Delivery is
 * per-request and ephemeral: the warning disappears the moment the streak
 * breaks, unlike the template's permanent re-rendered warnings.
 * Known coverage limit (documented, not hidden): this slot is evaluated once
 * per USER turn in buildMessages; an error streak that begins and dies
 * entirely INSIDE one SDK run is invisible to it (the template fork used to
 * catch those because it re-rendered every internal step). The covered case
 * is the expensive one: a run that ENDED leaving unresolved consecutive
 * errors, which is exactly the state the next turn inherits.
 * @param {Array<object>|undefined|null} payloadMessages
 * @returns {string|null}
 */
function getToolErrorVoice(payloadMessages) {
  if (!SUPERVISOR_ENABLED || !TOOL_ERROR_VOICE_ENABLED) return null;
  if (!Array.isArray(payloadMessages) || payloadMessages.length === 0) {
    return null;
  }
  const trailing = collectTrailingToolResults(payloadMessages);
  let n = 0;
  for (const body of trailing) {
    if (isToolErrorContent(body)) n++;
    else break;
  }
  if (n === 0 || n > TOOL_ERROR_VOICE_MAX_STREAK) return null;
  if (n === 1) {
    return (
      '# Harness note — tool error\n\n' +
      'The previous tool call returned an error. Diagnose the failure and ' +
      'retry with completely corrected arguments. Do not repeat the exact ' +
      'call that failed.'
    );
  }
  return (
    '# Harness note — repeated tool errors\n\n' +
    `${n} consecutive tool errors detected. Your previous approach is ` +
    'incorrect. You MUST switch to a fundamentally different approach or ' +
    'materially corrected arguments — inspect the error text, verify the ' +
    'preconditions (path, arguments, tool choice) instead of guessing again. ' +
    'If nothing fits, say plainly what is blocking you rather than ' +
    'continuing to fail silently.'
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
    /**
     * @librechat/agents >= 4 owns the continuation loop and THROWS if a Stop
     * hook injects after its own `maxStopContinuations` budget (default 8) is
     * spent. Our per-turn ceiling is tracked separately in Mongo, so defer to
     * the SDK's number here rather than relying on the two staying in step.
     */
    if (typeof input?.continuationBudgetRemaining === 'number' && input.continuationBudgetRemaining <= 0) {
      logger.info(
        `[harnessSupervisor] conv=${convLabel} stall-recovery Stop hook: SDK continuation budget exhausted — not continuing`,
      );
      return {};
    }
    if (!shouldFireStallProbe(input?.messages)) {
      /** [cowork] Component 7 (S5): the probe declining is exactly where a
       *  fabricated completion claim hides — its closing message looks
       *  complete (often DONE_TAG'd or a question, both probe exemptions).
       *  Checked only on the probe's "no", so when both qualify the probe
       *  wins (more general nudge). Rides the same slot machinery: caps are
       *  shared, so a fabricating run cannot spend more continuations than a
       *  stalling one. */
      if (shouldReconcileClaims(input?.messages)) {
        const reserved = await tryReserveStallProbeSlot(conversationId, {
          toolCount: 0,
          responseText: extractMessageText(input.messages[input.messages.length - 1]),
        });
        if (reserved) {
          continuationNumber += 1;
          logger.info(
            `[harnessSupervisor] conv=${convLabel} claim-vs-call mismatch — blocking Stop to reconcile ` +
              `(continuation=${continuationNumber})`,
          );
          /** Same visibility marker as the stall probe (docs/42): a
           *  continuation the chat cannot see looks like a glitch. */
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
          return {
            decision: 'block',
            reason: CLAIM_RECONCILE_MESSAGE,
            additionalContext: CLAIM_RECONCILE_MESSAGE,
          };
        }
        logger.info(
          `[harnessSupervisor] conv=${convLabel} claim-vs-call mismatch detected but not continuing (cap reached)`,
        );
      }
      /** Debug-level either way: a silent non-firing decision was previously
       *  diagnosable only by querying Mongo directly (docs/39 §10). Reason
       *  recomputed here (cheap string checks) purely for this log line —
       *  shouldFireStallProbe stays a plain boolean so its contract/tests
       *  don't change. "awaiting user input" vs "done/no stall" is worth
       *  telling apart on its own after the 2026-08-26 question-nudge bug. */
      const lastMsg = (input?.messages ?? []).at(-1);
      const lastText = getMessageType(lastMsg) === 'ai' ? extractMessageText(lastMsg) : '';
      const reason = containsDoneTag(lastText)
        ? 'DONE_TAG present'
        : endsWithQuestionToUser(lastText)
          ? 'ends in a question — awaiting user input'
          : 'no stall condition (mid tool-call or no prior tool call)';
      logger.debug(
        `[harnessSupervisor] conv=${convLabel} stall-recovery Stop hook: no continuation needed (${reason})`,
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
    const runMessages = input?.messages ?? [];
    const toolCount = runMessages.filter((m) => getMessageType(m) === 'tool').length;
    /** The stalled round's own final text — same message shouldFireStallProbe
     *  just read to call this a stall. Lets tryReserveStallProbeSlot catch a
     *  round that repeats its own conclusion verbatim while still issuing
     *  fresh-but-unproductive tool calls (e.g. retried searches that never
     *  find anything) — a real pattern tool-count alone missed. */
    const responseText = extractMessageText(runMessages[runMessages.length - 1]);
    /** Cap is authoritative and fails closed; see tryReserveStallProbeSlot. */
    const reserved = await tryReserveStallProbeSlot(conversationId, { toolCount, responseText });
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
    /** SDK >= 4 only continues when the hook supplies a message to inject
     *  (`additionalContext` / `injectedMessages`); `reason` alone is ignored. */
    return { decision: 'block', reason: STALL_PROBE_MESSAGE, additionalContext: STALL_PROBE_MESSAGE };
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
  getToolVoiceReminder,
  decideToolVoice,
  assistantHasToolCall,
  hasCompletionClaimContent,
  shouldReconcileClaims,
  CLAIM_RECONCILE_MESSAGE,
  getToolErrorVoice,
  STALL_PROBE_MESSAGE,
  DONE_TAG,
  STALL_PROBE_REENTRY_ENABLED,
};
