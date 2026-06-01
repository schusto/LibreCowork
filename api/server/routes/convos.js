const fs = require('fs');
const http = require('http');
const https = require('https');
const multer = require('multer');
const express = require('express');
const yaml = require('js-yaml');
const { sleep } = require('@librechat/agents');
const {
  isEnabled,
  resolveImportMaxFileSize,
  sanitizeTitle,
  restoreTenantContextFromReq,
} = require('@librechat/api');
const { logger } = require('@librechat/data-schemas');
const { CacheKeys, EModelEndpoint } = require('librechat-data-provider');
const {
  createImportLimiters,
  validateConvoAccess,
  createForkLimiters,
  configMiddleware,
} = require('~/server/middleware');
const { forkConversation, duplicateConversation } = require('~/server/utils/import/fork');
const { storage, importFileFilter } = require('~/server/routes/files/multer');
const requireJwtAuth = require('~/server/middleware/requireJwtAuth');
const { importConversations } = require('~/server/utils/import');
const getLogStores = require('~/cache/getLogStores');
const db = require('~/models');

const assistantClients = {
  [EModelEndpoint.azureAssistants]: require('~/server/services/Endpoints/azureAssistants'),
  [EModelEndpoint.assistants]: require('~/server/services/Endpoints/assistants'),
};

const router = express.Router();
router.use(requireJwtAuth);

router.get('/', async (req, res) => {
  const limit = parseInt(req.query.limit, 10) || 25;
  const cursor = req.query.cursor;
  const isArchived = isEnabled(req.query.isArchived);
  const search = req.query.search ? decodeURIComponent(req.query.search) : undefined;
  const sortBy = req.query.sortBy || 'updatedAt';
  const sortDirection = req.query.sortDirection || 'desc';

  let tags;
  if (req.query.tags) {
    tags = Array.isArray(req.query.tags) ? req.query.tags : [req.query.tags];
  }

  try {
    const result = await db.getConvosByCursor(req.user.id, {
      cursor,
      limit,
      isArchived,
      tags,
      search,
      sortBy,
      sortDirection,
    });
    res.status(200).json(result);
  } catch (error) {
    logger.error('Error fetching conversations', error);
    res.status(500).json({ error: 'Error fetching conversations' });
  }
});

router.get('/:conversationId', async (req, res) => {
  const { conversationId } = req.params;
  const convo = await db.getConvo(req.user.id, conversationId);

  if (convo) {
    res.status(200).json(convo);
  } else {
    res.status(404).end();
  }
});

router.get('/gen_title/:conversationId', async (req, res) => {
  const { conversationId } = req.params;
  const titleCache = getLogStores(CacheKeys.GEN_TITLE);
  const key = `${req.user.id}-${conversationId}`;
  let title = await titleCache.get(key);

  if (!title) {
    // Exponential backoff: 500ms, 1s, 2s, 4s, 8s (total ~15.5s max wait)
    const delays = [500, 1000, 2000, 4000, 8000];
    for (const delay of delays) {
      await sleep(delay);
      title = await titleCache.get(key);
      if (title) {
        break;
      }
    }
  }

  if (title) {
    await titleCache.delete(key);
    res.status(200).json({ title });
  } else {
    res.status(404).json({
      message: "Title not found or method not implemented for the conversation's endpoint",
    });
  }
});

router.delete('/', async (req, res) => {
  let filter = {};
  const { conversationId, source, thread_id, endpoint } = req.body?.arg ?? {};

  // Prevent deletion of all conversations
  if (!conversationId && !source && !thread_id && !endpoint) {
    return res.status(400).json({
      error: 'no parameters provided',
    });
  }

  if (conversationId) {
    filter = { conversationId };
  } else if (source === 'button') {
    return res.status(200).send('No conversationId provided');
  }

  if (
    typeof endpoint !== 'undefined' &&
    Object.prototype.propertyIsEnumerable.call(assistantClients, endpoint)
  ) {
    /** @type {{ openai: OpenAI }} */
    const { openai } = await assistantClients[endpoint].initializeClient({ req, res });
    try {
      const response = await openai.beta.threads.delete(thread_id);
      logger.debug('Deleted OpenAI thread:', response);
    } catch (error) {
      logger.error('Error deleting OpenAI thread:', error);
    }
  }

  try {
    const dbResponse = await db.deleteConvos(req.user.id, filter);
    if (filter.conversationId) {
      await db.deleteToolCalls(req.user.id, filter.conversationId);
      await db.deleteConvoSharedLink(req.user.id, filter.conversationId);
    }
    res.status(201).json(dbResponse);
  } catch (error) {
    logger.error('Error clearing conversations', error);
    res.status(500).send('Error clearing conversations');
  }
});

router.delete('/all', async (req, res) => {
  try {
    const dbResponse = await db.deleteConvos(req.user.id, {});
    await db.deleteToolCalls(req.user.id);
    await db.deleteAllSharedLinks(req.user.id);
    res.status(201).json(dbResponse);
  } catch (error) {
    logger.error('Error clearing conversations', error);
    res.status(500).send('Error clearing conversations');
  }
});

/**
 * Archives or unarchives a conversation.
 * @route POST /archive
 * @param {string} req.body.arg.conversationId - The conversation ID to archive/unarchive.
 * @param {boolean} req.body.arg.isArchived - Whether to archive (true) or unarchive (false).
 * @returns {object} 200 - The updated conversation object.
 */
router.post('/archive', validateConvoAccess, async (req, res) => {
  const { conversationId, isArchived } = req.body?.arg ?? {};

  if (!conversationId) {
    return res.status(400).json({ error: 'conversationId is required' });
  }

  if (typeof isArchived !== 'boolean') {
    return res.status(400).json({ error: 'isArchived must be a boolean' });
  }

  try {
    const dbResponse = await db.saveConvo(
      {
        userId: req?.user?.id,
        isTemporary: req?.body?.isTemporary,
        interfaceConfig: req?.config?.interfaceConfig,
      },
      { conversationId, isArchived },
      { context: `POST /api/convos/archive ${conversationId}` },
    );
    res.status(200).json(dbResponse);
  } catch (error) {
    logger.error('Error archiving conversation', error);
    res.status(500).send('Error archiving conversation');
  }
});

/** Maximum allowed length for conversation titles */
const MAX_CONVO_TITLE_LENGTH = 1024;

/**
 * Updates a conversation's title.
 * @route POST /update
 * @param {string} req.body.arg.conversationId - The conversation ID to update.
 * @param {string} req.body.arg.title - The new title for the conversation.
 * @returns {object} 201 - The updated conversation object.
 */
router.post('/update', validateConvoAccess, async (req, res) => {
  const { conversationId, title } = req.body?.arg ?? {};

  if (!conversationId) {
    return res.status(400).json({ error: 'conversationId is required' });
  }

  if (title === undefined) {
    return res.status(400).json({ error: 'title is required' });
  }

  if (typeof title !== 'string') {
    return res.status(400).json({ error: 'title must be a string' });
  }

  const sanitizedTitle = title.trim().slice(0, MAX_CONVO_TITLE_LENGTH);

  try {
    const dbResponse = await db.saveConvo(
      {
        userId: req?.user?.id,
        isTemporary: req?.body?.isTemporary,
        interfaceConfig: req?.config?.interfaceConfig,
      },
      { conversationId, title: sanitizedTitle },
      { context: `POST /api/convos/update ${conversationId}` },
    );
    res.status(201).json(dbResponse);
  } catch (error) {
    logger.error('Error updating conversation', error);
    res.status(500).send('Error updating conversation');
  }
});

const { importIpLimiter, importUserLimiter } = createImportLimiters();
/** Fork and duplicate share one rate-limit budget (same "clone" operation class) */
const { forkIpLimiter, forkUserLimiter } = createForkLimiters();
const importMaxFileSize = resolveImportMaxFileSize();
const upload = multer({
  storage,
  fileFilter: importFileFilter,
  limits: { fileSize: importMaxFileSize },
});
const uploadSingle = upload.single('file');

function handleUpload(req, res, next) {
  uploadSingle(req, res, (err) => {
    if (err && err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ message: 'File exceeds the maximum allowed size' });
    }
    if (err) {
      return next(err);
    }
    next();
  });
}

/**
 * Imports a conversation from a JSON file and saves it to the database.
 * @route POST /import
 * @param {Express.Multer.File} req.file - The JSON file to import.
 * @returns {object} 201 - success response - application/json
 */
router.post(
  '/import',
  importIpLimiter,
  importUserLimiter,
  configMiddleware,
  handleUpload,
  restoreTenantContextFromReq,
  async (req, res) => {
    try {
      /* TODO: optimize to return imported conversations and add manually */
      await importConversations({
        filepath: req.file.path,
        requestUserId: req.user.id,
        userRole: req.user.role,
        interfaceConfig: req.config?.interfaceConfig,
      });
      res.status(201).json({ message: 'Conversation(s) imported successfully' });
    } catch (error) {
      logger.error('Error processing file', error);
      res.status(500).send('Error processing file');
    }
  },
);

/**
 * POST /fork
 * This route handles forking a conversation based on the TForkConvoRequest and responds with TForkConvoResponse.
 * @route POST /fork
 * @param {express.Request<{}, TForkConvoResponse, TForkConvoRequest>} req - Express request object.
 * @param {express.Response<TForkConvoResponse>} res - Express response object.
 * @returns {Promise<void>} - The response after forking the conversation.
 */
router.post('/fork', forkIpLimiter, forkUserLimiter, async (req, res) => {
  try {
    /** @type {TForkConvoRequest} */
    const { conversationId, messageId, option, splitAtTarget, latestMessageId } = req.body;
    const result = await forkConversation({
      requestUserId: req.user.id,
      originalConvoId: conversationId,
      targetMessageId: messageId,
      latestMessageId,
      records: true,
      splitAtTarget,
      option,
    });

    res.json(result);
  } catch (error) {
    logger.error('Error forking conversation:', error);
    res.status(500).send('Error forking conversation');
  }
});

router.post('/duplicate', forkIpLimiter, forkUserLimiter, async (req, res) => {
  const { conversationId, title } = req.body;

  try {
    const result = await duplicateConversation({
      userId: req.user.id,
      conversationId,
      title,
    });
    res.status(201).json(result);
  } catch (error) {
    logger.error('Error duplicating conversation:', error);
    res.status(500).send('Error duplicating conversation');
  }
});

// ── Title regeneration ────────────────────────────────────────────────────────

/**
 * Read titlePrompt and titleModel from librechat.yaml for a named custom endpoint.
 * Falls back to the TITLE_PROMPT env var, then a hardcoded default.
 *
 * @param {string} endpointName - e.g. "MLX"
 * @returns {{ titlePrompt: string, titleModel: string|null, baseURL: string }}
 */
function loadTitleConfig(endpointName = 'MLX') {
  const FALLBACK_PROMPT =
    'Summarize the purpose of this conversation and provide a concise title ' +
    'in the detected language ({convo})';
  const FALLBACK_BASE_URL = 'http://host.docker.internal:1235/v1';

  // Candidate paths for librechat.yaml (host path + symlink target)
  const candidates = [
    process.env.LIBRECHAT_CONFIG_PATH,
    '/app/librechat.yaml',
    '/data/librechat.yaml',
  ].filter(Boolean);

  for (const p of candidates) {
    try {
      const raw = fs.readFileSync(p, 'utf8');
      const cfg = yaml.load(raw);
      const customs = cfg?.endpoints?.custom ?? [];
      const ep = customs.find((e) => e.name === endpointName) ?? customs[0];
      if (ep) {
        return {
          titlePrompt: ep.titlePrompt ?? FALLBACK_PROMPT,
          titleModel: ep.titleModel === 'current_model' ? null : (ep.titleModel ?? null),
          baseURL: ep.baseURL ?? FALLBACK_BASE_URL,
        };
      }
    } catch (_) {
      // file not found or parse error — try next candidate
    }
  }

  return { titlePrompt: FALLBACK_PROMPT, titleModel: null, baseURL: FALLBACK_BASE_URL };
}

/**
 * Call the mlx-proxy (or any OpenAI-compatible endpoint) for a title completion.
 * Returns the raw response string, or null on failure.
 *
 * @param {{ baseURL: string, model: string, prompt: string }} opts
 * @returns {Promise<string|null>}
 */
function fetchTitle({ baseURL, model, prompt }) {
  return new Promise((resolve) => {
    const url = new URL('/chat/completions', baseURL.replace(/\/v1\/?$/, '') + '/v1');
    const payload = JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 256,
      temperature: 0.3,
      stream: false,
    });

    const lib = url.protocol === 'https:' ? https : http;
    const options = {
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
        Authorization: `Bearer ${process.env.MLX_API_KEY || 'mlx-proxy'}`,
      },
    };

    const req = lib.request(options, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => {
        try {
          const body = JSON.parse(data);
          resolve(body?.choices?.[0]?.message?.content ?? null);
        } catch (_) {
          resolve(null);
        }
      });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(45_000, () => { req.destroy(); resolve(null); });
    req.write(payload);
    req.end();
  });
}

/**
 * Discover the currently-loaded model from the mlx-proxy /health endpoint.
 * Returns null if unreachable.
 *
 * @param {string} baseURL
 * @returns {Promise<string|null>}
 */
function discoverCurrentModel(baseURL) {
  return new Promise((resolve) => {
    const url = new URL('/health', baseURL.replace(/\/v1\/?$/, ''));
    const lib = url.protocol === 'https:' ? https : http;
    const req = lib.get(url.toString(), (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => {
        try { resolve(JSON.parse(data)?.current_model ?? null); }
        catch (_) { resolve(null); }
      });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(5_000, () => { req.destroy(); resolve(null); });
  });
}

/**
 * Auth middleware for /regen_title.
 *
 * Two accepted paths — exactly one must succeed, otherwise 401:
 *
 *   1. X-Internal-Secret header matches LIBRECHAT_INTERNAL_SECRET (set in .env).
 *      Synthesises a minimal req.user and calls next() immediately — no JWT needed.
 *      Disabled (falls through to path 2) if LIBRECHAT_INTERNAL_SECRET is not set.
 *
 *   2. Valid user JWT cookie/header — delegated to requireJwtAuth, which either
 *      populates req.user and calls next(), or sends a 401 itself.
 *
 * Security note: the route MUST NOT call next() without a valid identity.
 * Path 1 authenticates the MCP server process; path 2 authenticates a browser session.
 */
const INTERNAL_SECRET = process.env.LIBRECHAT_INTERNAL_SECRET || '';

function regenTitleAuth(req, res, next) {
  const secret = req.headers['x-internal-secret'];
  if (INTERNAL_SECRET && secret === INTERNAL_SECRET) {
    // Valid internal secret — synthesise a service identity and proceed.
    req.user = { id: 'internal-service', role: 'SERVICE' };
    return next();
  }
  // No valid secret (or secret not configured) — require a real user JWT.
  return requireJwtAuth(req, res, next);
}

/**
 * Regenerate the title for an existing conversation using the same titlePrompt
 * and model configured in librechat.yaml for the MLX (or named) custom endpoint.
 *
 * The first user message of the conversation is used as the input — consistent
 * with how LibreChat generates titles on new conversations.
 *
 * Auth: standard JWT (user session) OR X-Internal-Secret header (MCP server).
 *
 * @route POST /regen_title
 * @param {string} req.body.conversationId - ID of the conversation to retitle.
 * @param {string} [req.body.endpoint]     - Custom endpoint name to read config from (default: "MLX").
 * @param {string} [req.body.proposedTitle] - Fallback title if LLM call fails.
 * @returns {object} 200 - { conversationId, title }
 */
// regenTitleAuth is a complete auth gate (secret OR JWT) — no unauthenticated path.
const regenTitleRouter = express.Router();
regenTitleRouter.post('/regen_title', regenTitleAuth, configMiddleware, async (req, res) => {
  const { conversationId, endpoint: endpointName = 'MLX', proposedTitle } = req.body ?? {};

  if (!conversationId) {
    return res.status(400).json({ error: 'conversationId is required' });
  }

  // 1. Fetch the first user message for this conversation
  let firstUserText = null;
  let convoUserId = req.user.id;
  try {
    // When called via internal secret, req.user.id is the sentinel 'internal-service';
    // pass user only when it's a real user ID so getMessages doesn't filter nothing out.
    const msgQuery = { conversationId };
    if (req.user.id !== 'internal-service') msgQuery.user = req.user.id;
    const messages = await db.getMessages(msgQuery);
    const userMessages = (messages ?? [])
      .filter((m) => m.isCreatedByUser && m.text)
      .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));

    if (userMessages.length === 0) {
      return res.status(404).json({ error: 'No user messages found for this conversation' });
    }

    firstUserText = userMessages[0].text.slice(0, 1200);

    // Resolve real userId — when called via X-Internal-Secret, req.user.id is the
    // sentinel 'internal-service' which doesn't match any conversation's user field.
    // Extract the real owner from the first user message instead.
    convoUserId =
      req.user.id === 'internal-service'
        ? (userMessages[0]?.user?.toString?.() ?? req.user.id)
        : req.user.id;

    logger.info(`[regen_title] conversationId=${conversationId} convoUserId=${convoUserId} (req.user.id=${req.user.id})`);
  } catch (err) {
    logger.error('[regen_title] Error fetching messages:', err);
    return res.status(500).json({ error: 'Failed to fetch conversation messages' });
  }

  // 2. Load title config from librechat.yaml
  const { titlePrompt, titleModel, baseURL } = loadTitleConfig(endpointName);

  // 3. Resolve model: use configured titleModel, or discover currently-loaded model
  let model = titleModel;
  if (!model) {
    model = await discoverCurrentModel(baseURL);
  }
  if (!model) {
    return res.status(503).json({ error: 'Could not determine model — is mlx-proxy running?' });
  }

  // 4. Build the prompt and call the LLM
  const prompt = titlePrompt.replace('{convo}', firstUserText);
  const rawTitle = await fetchTitle({ baseURL, model, prompt });

  // 5. Sanitize; fall back to proposedTitle if the LLM returned nothing
  const generatedTitle = rawTitle ? (sanitizeTitle(rawTitle) ?? rawTitle.trim().slice(0, 60)) : null;
  const title = generatedTitle ?? (proposedTitle ? proposedTitle.trim().slice(0, 60) : null);

  if (!title) {
    return res.status(502).json({ error: 'Title generation failed — no response from model and no proposed_title fallback' });
  }

  let savedConvo;
  try {
    savedConvo = await db.saveConvo(
      {
        userId: convoUserId,
        isTemporary: false,
        interfaceConfig: req.config?.interfaceConfig,
      },
      { conversationId, title },
      { context: 'api/server/routes/convos.js /regen_title', noUpsert: true },
    );
  } catch (err) {
    logger.error('[regen_title] Error saving title:', err);
    return res.status(500).json({ error: 'Title generated but failed to save' });
  }

  // saveConvo returns null when noUpsert:true and no document matched.
  // It returns { message: '...' } on internal error.
  if (!savedConvo) {
    logger.error(`[regen_title] saveConvo returned null — no conversation matched { conversationId: ${conversationId}, user: ${convoUserId} }. Title "${title}" was NOT saved.`);
    return res.status(404).json({ error: `Conversation not found for userId=${convoUserId}`, title });
  }
  if (savedConvo.message) {
    logger.error(`[regen_title] saveConvo returned error: ${savedConvo.message}`);
    return res.status(500).json({ error: 'Title generated but saveConvo reported an error', title });
  }

  const source = generatedTitle ? 'llm' : 'proposed_title_fallback';
  logger.info(`[regen_title] ${conversationId} → "${title}" (source: ${source}) — saved OK for user ${convoUserId}`);
  return res.status(200).json({ conversationId, title });
});

// Parent router: regenTitleRouter is mounted first (its own auth gate),
// then the JWT-gated router for everything else.
const rootRouter = express.Router();
rootRouter.use(regenTitleRouter);
rootRouter.use(router);
module.exports = rootRouter;
