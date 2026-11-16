const express = require('express');
const request = require('supertest');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const { createModels } = require('@librechat/data-schemas');

/**
 * Multi-tenant hardening specs for /api/tasks (docs/handover/
 * 2026-11-tasklist-multitenant-hardening.md). Integration wiring follows
 * grants.spec.js: real MongoDB (memory server), auth injected per request.
 */

process.env.CWK_TASKLIST_INTERNAL_SECRET = 'test-internal-secret';

// JWT stand-in: identity comes from the x-test-jwt-user header so tests can
// switch callers freely; absence means unauthenticated.
jest.mock('~/server/middleware/requireJwtAuth', () => {
  return function requireJwtAuth(req, res, next) {
    const uid = req.headers['x-test-jwt-user'];
    if (!uid) return res.status(401).json({ error: 'no jwt' });
    req.user = { id: uid, role: 'USER' };
    return next();
  };
});

// Mirror of data-schemas getConvo against the real registered model.
jest.mock('~/models', () => {
  const mg = require('mongoose');
  return {
    getConvo: jest.fn(async (user, conversationId) =>
      mg.models.Conversation.findOne({ user, conversationId }).lean(),
    ),
  };
});

const SECRET = { 'x-internal-secret': 'test-internal-secret' };
const objId = () => new mongoose.Types.ObjectId().toHexString();

let mongoServer;
let app;
let collection;

beforeAll(async () => {
  mongoServer = await MongoMemoryServer.create();
  await mongoose.connect(mongoServer.getUri());
  createModels(mongoose);
  const router = require('../tasks');
  app = express();
  app.use(express.json());
  app.use('/api/tasks', router);
  collection = mongoose.connection.collection('tasklists');
});

afterAll(async () => {
  await mongoose.disconnect();
  await mongoServer.stop();
});

afterEach(async () => {
  await collection.deleteMany({});
  await mongoose.models.Conversation.deleteMany({});
});

async function seedConvo(userId, conversationId) {
  await mongoose.models.Conversation.create({
    user: userId,
    conversationId,
    endpoint: 'free_chat',
  });
}

const todo = (label = 't') => [
  { content: label, activeForm: `doing ${label}`, status: 'pending' },
];

describe('Fix 8 — per-conversation document cap', () => {
  it('writing past the cap trims oldest so total stays <= 200', async () => {
    const convo = 'convo-f8';
    const uid = objId();
    const base = Date.now() - 1000 * 60 * 60;
    // Seed 204 older docs directly (cap is enforced on POST).
    const seeds = Array.from({ length: 204 }, (_, i) => ({
      _id: `${convo}:seed-${String(i).padStart(4, '0')}`,
      userId: uid,
      conversationId: convo,
      messageId: `seed-${String(i).padStart(4, '0')}`,
      todos: todo(`seed-${i}`),
      updatedAt: new Date(base + i * 1000),
    }));
    await collection.insertMany(seeds);

    const res = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'fresh', userId: uid, todos: todo('fresh') });
    expect(res.status).toBe(200);

    const remaining = await collection.countDocuments({ conversationId: convo });
    expect(remaining).toBeLessThanOrEqual(200);
    // Newest survivors kept, oldest trimmed: seed-0 gone, seed-203 kept, fresh present.
    expect(await collection.findOne({ _id: `${convo}:seed-0000` })).toBeNull();
    expect(await collection.findOne({ _id: `${convo}:seed-0203` })).not.toBeNull();
    expect(await collection.findOne({ _id: `${convo}:fresh` })).not.toBeNull();
  });
});

describe('Fix 7 — compound index for the fallback query', () => {
  it('tasklists carries TTL and conversationId+updatedAt indexes', async () => {
    await request(app) // ensure getCollection() ran at least once
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: 'c-f7', messageId: 'm1', userId: objId(), todos: todo() });
    // Indexes are ensured fire-and-forget; give them a moment to land.
    let names = [];
    for (let i = 0; i < 25; i++) {
      const idx = await collection.indexes();
      names = idx.map((x) => Object.keys(x.key).join('+'));
      if (names.includes('conversationId+updatedAt') && names.includes('updatedAt')) break;
      await new Promise((r) => setTimeout(r, 40));
    }
    expect(names).toContain('updatedAt');
    expect(names).toContain('conversationId+updatedAt');
  });
});

describe('Fix 6 — DELETE keeps secret-or-JWT semantics after dedupe', () => {
  it('internal secret deletes; non-owner JWT gets 404; owner JWT deletes', async () => {
    const owner = objId();
    const convo = 'convo-f6';
    await seedConvo(owner, convo);
    await collection.insertOne({
      _id: `${convo}:m1`, userId: owner, conversationId: convo,
      messageId: 'm1', todos: todo(), updatedAt: new Date(),
    });
    // Non-owner JWT: ownership gate 404s before touching the doc.
    const bad = await request(app)
      .delete('/api/tasks')
      .set({ 'x-test-jwt-user': objId() })
      .send({ conversationId: convo, messageId: 'm1' });
    expect(bad.status).toBe(404);
    expect(await collection.countDocuments({})).toBe(1);
    // Owner JWT deletes.
    const own = await request(app)
      .delete('/api/tasks')
      .set({ 'x-test-jwt-user': owner })
      .send({ conversationId: convo, messageId: 'm1' });
    expect(own.status).toBe(200);
    expect(own.body.deleted).toBe(1);
    // Internal secret deletes too.
    await collection.insertOne({
      _id: `${convo}:m2`, userId: owner, conversationId: convo,
      messageId: 'm2', todos: todo(), updatedAt: new Date(),
    });
    const svc = await request(app)
      .delete('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'm2' });
    expect(svc.status).toBe(200);
  });
});

describe('Fix 4 — fallback sort has a deterministic tiebreaker', () => {
  it('equal updatedAt resolves to the higher _id', async () => {
    const convo = 'convo-f4';
    const now = new Date();
    await collection.insertMany([
      {
        _id: `${convo}:msg-a`, userId: objId(), conversationId: convo,
        messageId: 'msg-a', todos: todo('a'), updatedAt: now,
      },
      {
        _id: `${convo}:msg-b`, userId: objId(), conversationId: convo,
        messageId: 'msg-b', todos: todo('b'), updatedAt: now,
      },
    ]);
    const res = await request(app).get('/api/tasks').set(SECRET).query({ conversationId: convo });
    expect(res.body.todos[0].content).toBe('b'); // 'msg-b' > 'msg-a' lexicographically
  });
});

describe('Fix 3 — GET fallback filters by the JWT owner', () => {
  it('JWT caller never receives a doc whose stored userId differs', async () => {
    const owner = objId();
    const convo = 'convo-f3';
    await seedConvo(owner, convo);
    // A doc in the owned conversation stamped with someone else's userId
    // (possible pre-F1): must not surface on the messageId-less fallback.
    await collection.insertOne({
      _id: `${convo}:msg-other`,
      userId: objId(),
      conversationId: convo,
      messageId: 'msg-other',
      todos: todo('foreign'),
      updatedAt: new Date(Date.now() + 60_000), // newest, so fallback would pick it
    });
    await collection.insertOne({
      _id: `${convo}:msg-mine`,
      userId: owner,
      conversationId: convo,
      messageId: 'msg-mine',
      todos: todo('mine'),
      updatedAt: new Date(),
    });

    const res = await request(app)
      .get('/api/tasks')
      .set({ 'x-test-jwt-user': owner })
      .query({ conversationId: convo });
    expect(res.status).toBe(200);
    expect(res.body.todos[0].content).toBe('mine');

    // Internal service still sees the newest doc regardless of userId.
    const svc = await request(app).get('/api/tasks').set(SECRET).query({ conversationId: convo });
    expect(svc.body.todos[0].content).toBe('foreign');
  });
});

describe('Fix 2 — scope ids may not contain the ":" separator', () => {
  it('POST rejects conversationId or messageId containing ":"', async () => {
    const r1 = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: 'a:b', messageId: 'c', userId: objId(), todos: todo() });
    expect(r1.status).toBe(400);
    const r2 = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: 'ab', messageId: 'c:d', userId: objId(), todos: todo() });
    expect(r2.status).toBe(400);
    expect(await collection.countDocuments({})).toBe(0);
  });

  it('GET rejects conversationId containing ":"', async () => {
    const res = await request(app)
      .get('/api/tasks')
      .set(SECRET)
      .query({ conversationId: 'a:b' });
    expect(res.status).toBe(400);
  });
});

describe('Fix 1 — POST must not trust scope ids from the body', () => {
  it('rejects victim-conversation + attacker-userId with 400 and leaves the victim doc untouched', async () => {
    const victim = objId();
    const attacker = objId();
    const convo = 'convo-victim';
    await seedConvo(victim, convo);

    // Victim's legitimate write lands first.
    const okRes = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'msg-v', userId: victim, todos: todo('mine') });
    expect(okRes.status).toBe(200);

    // Attacker (service secret assumed held) tries to clobber under victim's conv.
    const bad = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'msg-v', userId: attacker, todos: todo('evil') });
    expect(bad.status).toBe(400);

    const doc = await collection.findOne({ _id: `${convo}:msg-v` });
    expect(doc.todos[0].content).toBe('mine');
    expect(doc.userId).toBe(victim);
  });

  it('accepts a first-turn write (conversation not yet persisted) with well-formed userId', async () => {
    const res = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({
        conversationId: 'brand-new-convo',
        messageId: 'm1',
        userId: objId(),
        todos: todo(),
      });
    expect(res.status).toBe(200);
  });

  it('rejects unknown conversation combined with malformed userId', async () => {
    const res = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({
        conversationId: 'ghost-convo',
        messageId: 'm1',
        userId: 'not-an-object-id',
        todos: todo(),
      });
    expect(res.status).toBe(400);
    expect(await collection.countDocuments({})).toBe(0);
  });

  // Finding 2 lock-in (docs/handover/2026-11-tasklist-finding2-null-userid.md):
  // userId null/absent must NEVER reach $set — not even for a conversation that
  // exists and could otherwise pass ownership checks. This closes the one gap
  // the malformed-userId test left open (that case exercised the ghost branch).
  it('rejects absent/null userId on an existing conversation (invariant: no null-stamped docs)', async () => {
    const uid = objId();
    const convo = 'convo-f2';
    await seedConvo(uid, convo);

    const absent = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'm-absent', todos: todo() });
    expect(absent.status).toBe(400);

    const explicitNull = await request(app)
      .post('/api/tasks')
      .set(SECRET)
      .send({ conversationId: convo, messageId: 'm-null', userId: null, todos: todo() });
    expect(explicitNull.status).toBe(400);

    // Invariant itself: nothing was persisted under a null/missing stamp.
    expect(await collection.countDocuments({ userId: null })).toBe(0);
    expect(await collection.countDocuments({ userId: { $exists: false } })).toBe(0);
    expect(await collection.countDocuments({})).toBe(0);
  });
});
