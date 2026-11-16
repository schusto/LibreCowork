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
});
