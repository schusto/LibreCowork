// [cowork] Scoped task-list store — /api/tasks
//
// Conversation- and (intended) actor-scoped persistence for todo_list MCP server v2.
// Design: docs/changes/2026-11-scoped-tasklist.md (§3.2).
// Executor notes: docs/handover/scoped-tasklist-implementation.md.
//
// Storage: raw mongoose collection "tasklists" (precedent: harnessSupervisor.js —
// plain collection, lazy TTL index, no schema registration).
//   _id:     `${conversationId}:${messageId}`
//   { userId, conversationId, messageId, todos: [...], updatedAt }
//
// Auth mirrors the regen_title pattern (convos.js):
//   - POST: X-Internal-Secret header == CWK_TASKLIST_INTERNAL_SECRET (.env), only.
//   - GET/DELETE: internal secret (MCP forwarder) OR user JWT (browser).
//     JWT callers must own conversationId (db.getConvo); else 404 — no existence leak.
//
// KNOWN RISK (accepted 2026-11 by Erik; revisit if usage shows clobbering):
// The actor dimension (messageId separating parent from subagent threads) depends
// on per-actor requestBody propagation. Static reading of @librechat/agents
// SubagentExecutor.mjs (~L187–227) says the child inherits the parent's
// requestBody AND thread_id verbatim (spread order puts inheritedConfigurable
// last), so a subagent's todo_update likely lands under the PARENT's messageId —
// i.e. documents are TURN-scoped, not actor-scoped, within one user turn.
// No stable per-actor id survives to the MCP tool boundary under the current SDK
// (run_id propagated too; __pregel_* keys are per-superstep volatile). Fix path
// if needed: SDK patch stamping childRunId into the child requestBody.messageId
// (see design §3.2 fix options). Rendering (P2) is unaffected: nested lists come
// from each thread's own content args either way.

const express = require('express');
const mongoose = require('mongoose');
const { logger } = require('@librechat/data-schemas');
const requireJwtAuth = require('~/server/middleware/requireJwtAuth');
const db = require('~/models');

const COLLECTION_NAME = 'tasklists';
const TTL_SECONDS = 7 * 24 * 60 * 60; // 7 days (design §3.2)
const MAX_TODOS = 50;
const MAX_FIELD_LEN = 500;
const VALID_STATUSES = new Set(['pending', 'in_progress', 'completed']);
// conversationId/messageId arrive via injected env; keep _id composition safe.
const SCOPE_ID_RE = /^[A-Za-z0-9_.:-]{1,128}$/;

const INTERNAL_SECRET = process.env.CWK_TASKLIST_INTERNAL_SECRET || '';

let indexEnsured = false;
function getCollection() {
  const collection = mongoose.connection.collection(COLLECTION_NAME);
  if (!indexEnsured) {
    indexEnsured = true;
    // Fire-and-forget, idempotent (harnessSupervisor precedent). Failure only
    // means docs never expire, not that reads/writes fail.
    collection
      .createIndex({ updatedAt: 1 }, { expireAfterSeconds: TTL_SECONDS })
      .catch((e) => logger.warn(`[tasks] failed to ensure TTL index: ${e.message}`));
  }
  return collection;
}

/** Write gate: internal secret only. Never calls next() without identity. */
function internalSecretAuth(req, res, next) {
  const secret = req.headers['x-internal-secret'];
  if (INTERNAL_SECRET && secret === INTERNAL_SECRET) {
    req.user = { id: 'internal-service', role: 'SERVICE' };
    return next();
  }
  return res.status(401).json({ error: 'Valid X-Internal-Secret header required' });
}

/** Read/delete gate: internal secret (MCP forwarder) OR user JWT (browser).
 * Exactly the regenTitleAuth shape from convos.js. */
function internalSecretOrJwt(req, res, next) {
  const secret = req.headers['x-internal-secret'];
  if (INTERNAL_SECRET && secret === INTERNAL_SECRET) {
    req.user = { id: 'internal-service', role: 'SERVICE' };
    return next();
  }
  return requireJwtAuth(req, res, next);
}

// ── Validation ───────────────────────────────────────────────────────────────

function validateScope(conversationId, messageId) {
  if (typeof conversationId !== 'string' || !SCOPE_ID_RE.test(conversationId)) {
    return 'conversationId missing or invalid';
  }
  if (typeof messageId !== 'string' || !SCOPE_ID_RE.test(messageId)) {
    return 'messageId missing or invalid';
  }
  return null;
}

/** Returns { todos } cleaned, or { error } — statuses/labels constrained hard. */
function validateTodos(raw) {
  if (!Array.isArray(raw)) return { error: "'todos' must be an array" };
  if (raw.length > MAX_TODOS) return { error: `too many todos (max ${MAX_TODOS})` };
  const todos = [];
  for (const t of raw) {
    if (
      t == null ||
      typeof t.content !== 'string' ||
      typeof t.activeForm !== 'string' ||
      typeof t.status !== 'string' ||
      !VALID_STATUSES.has(t.status)
    ) {
      return { error: "each todo needs content, activeForm and status in pending|in_progress|completed" };
    }
    todos.push({
      content: t.content.slice(0, MAX_FIELD_LEN),
      activeForm: t.activeForm.slice(0, MAX_FIELD_LEN),
      status: t.status,
    });
  }
  return { todos };
}

// ── Routes ───────────────────────────────────────────────────────────────────

const router = express.Router();

/**
 * POST /api/tasks — upsert the caller's scoped doc.
 * Body: { conversationId, messageId, userId?, todos: [...] }
 * Writes come from the MCP forwarder with X-Internal-Secret only.
 */
router.post('/', internalSecretAuth, async (req, res) => {
  const { conversationId, messageId, userId, todos: rawTodos } = req.body ?? {};
  const scopeErr = validateScope(conversationId, messageId);
  if (scopeErr) return res.status(400).json({ error: scopeErr });
  const { todos, error } = validateTodos(rawTodos);
  if (error) return res.status(400).json({ error });

  const _id = `${conversationId}:${messageId}`;
  const updatedAt = new Date();
  try {
    await getCollection().updateOne(
      { _id },
      {
        $set: {
          userId: typeof userId === 'string' ? userId : null,
          conversationId,
          messageId,
          todos,
          updatedAt,
        },
      },
      { upsert: true },
    );
    return res.status(200).json({ ok: true, id: _id, count: todos.length, updatedAt });
  } catch (e) {
    logger.error(`[tasks] POST failed for ${_id}:`, e);
    return res.status(500).json({ error: 'write failed' });
  }
});

/**
 * GET /api/tasks?conversationId=<id>[&messageId=<id>]
 * JWT + ownership. With messageId: that exact doc. Without: newest parent-scope
 * doc for the conversation (continuity fallback, design §3.4 first bullet —
 * "newest" = highest updatedAt; actor docs share the same key space today per
 * the KNOWN RISK note, so this doubles as the turn-scoped read).
 */
router.get('/', internalSecretOrJwt, async (req, res) => {
  const conversationId = req.query?.conversationId;
  const messageId = req.query?.messageId;
  if (typeof conversationId !== 'string' || !SCOPE_ID_RE.test(conversationId)) {
    return res.status(400).json({ error: 'conversationId query param required' });
  }
  if (messageId != null && (typeof messageId !== 'string' || !SCOPE_ID_RE.test(messageId))) {
    return res.status(400).json({ error: 'messageId query param invalid' });
  }

  // Ownership gate: reject ids the caller doesn't own without leaking existence.
  // Internal service identity is exempt (MCP forwarder; mirrors regen_title).
  if (req.user.id !== 'internal-service') {
    try {
      const convo = await db.getConvo(req.user.id, conversationId);
      if (!convo) return res.status(404).json({ error: 'not found' });
    } catch (e) {
      logger.error('[tasks] ownership check failed:', e);
      return res.status(500).json({ error: 'ownership check failed' });
    }
  }

  try {
    if (messageId) {
      const doc = await getCollection().findOne({ _id: `${conversationId}:${messageId}` });
      return res.status(200).json({ todos: doc?.todos ?? [], updatedAt: doc?.updatedAt ?? null });
    }
    const doc = await getCollection()
      .find({ conversationId })
      .sort({ updatedAt: -1 })
      .limit(1)
      .next();
    return res.status(200).json({ todos: doc?.todos ?? [], updatedAt: doc?.updatedAt ?? null });
  } catch (e) {
    logger.error('[tasks] GET failed:', e);
    return res.status(500).json({ error: 'read failed' });
  }
});

/**
 * DELETE /api/tasks — clear the scoped doc.
 * Body: { conversationId, messageId }. Internal secret (MCP todo_clear) or JWT owner.
 */
router.delete('/', (req, res, next) => {
  const secret = req.headers['x-internal-secret'];
  if (INTERNAL_SECRET && secret === INTERNAL_SECRET) {
    req.user = { id: 'internal-service', role: 'SERVICE' };
    return next();
  }
  return requireJwtAuth(req, res, next);
}, async (req, res) => {
  const { conversationId, messageId } = req.body ?? {};
  const scopeErr = validateScope(conversationId, messageId);
  if (scopeErr) return res.status(400).json({ error: scopeErr });

  // JWT callers must own the conversation; internal service skips the check
  // (same asymmetry as regen_title resolving the real owner from data).
  if (req.user.id !== 'internal-service') {
    try {
      const convo = await db.getConvo(req.user.id, conversationId);
      if (!convo) return res.status(404).json({ error: 'not found' });
    } catch (e) {
      logger.error('[tasks] ownership check failed:', e);
      return res.status(500).json({ error: 'ownership check failed' });
    }
  }

  try {
    const r = await getCollection().deleteOne({ _id: `${conversationId}:${messageId}` });
    return res.status(200).json({ ok: true, deleted: r.deletedCount });
  } catch (e) {
    logger.error('[tasks] DELETE failed:', e);
    return res.status(500).json({ error: 'delete failed' });
  }
});

module.exports = router;
