import { resolveContextProjection } from './projection';
import { QUOTE_MAX_COUNT } from '~/utils/quotes';
import { countFormattedMessageTokens } from '~/agents/client';

jest.mock('@librechat/agents', () => ({
  Providers: { OPENAI: 'openai' },
  createTokenCounter: jest.fn(async () => jest.fn(() => 1)),
  projectAgentContextUsage: jest.fn(() => ({ tokenCount: 1, maxContextTokens: 1000 })),
}));

const GRAPH_SELECT = 'messageId parentMessageId metadata.summaryUsedTokens';
// [cowork] `content` was added so the gauge can run the same deterministic
// context reduction the live path applies — see docs/18_context_reduction.md.
const BODY_SELECT = 'messageId parentMessageId tokenCount isCreatedByUser text quotes content';

function textStats(messageId: string, textBytes = 5) {
  return {
    messageId,
    textBytes,
    quoteCount: 0,
    quoteBytes: 0,
    quoteLineCount: 0,
    nonStringQuoteCount: 0,
  };
}

describe('resolveContextProjection', () => {
  const baseParams = {
    conversationId: 'conversation-1',
    messageId: 'message-1',
    endpoint: 'openai',
    maxContextTokens: 1000,
    model: 'gpt-4o',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null before tokenization when the conversation is too large', async () => {
    const { createTokenCounter } = jest.requireMock('@librechat/agents');
    const messages = Array.from({ length: 513 }, (_, index) => ({
      messageId: `message-${index}`,
      parentMessageId: index === 0 ? null : `message-${index - 1}`,
      isCreatedByUser: true,
      text: 'hello',
    }));
    const getMessages = jest.fn(async () => messages);
    const getMessageTextStats = jest.fn();

    const result = await resolveContextProjection(
      { userId: 'user-1', getMessages, getMessageTextStats },
      { ...baseParams, messageId: 'message-512' },
    );

    expect(result).toBeNull();
    expect(getMessages).toHaveBeenCalledTimes(1);
    expect(getMessages).toHaveBeenCalledWith(
      { conversationId: 'conversation-1', user: 'user-1' },
      GRAPH_SELECT,
      { limit: 513, sort: false },
    );
    expect(getMessageTextStats).not.toHaveBeenCalled();
    expect(createTokenCounter).not.toHaveBeenCalled();
  });

  it('returns null before tokenization when the branch is too long', async () => {
    const { createTokenCounter } = jest.requireMock('@librechat/agents');
    const messages = Array.from({ length: 257 }, (_, index) => ({
      messageId: `message-${index}`,
      parentMessageId: index === 0 ? null : `message-${index - 1}`,
      isCreatedByUser: true,
      text: 'hello',
    }));
    const getMessages = jest.fn(async () => messages);
    const getMessageTextStats = jest.fn();

    const result = await resolveContextProjection(
      { userId: 'user-1', getMessages, getMessageTextStats },
      { ...baseParams, messageId: 'message-256' },
    );

    expect(result).toBeNull();
    expect(getMessages).toHaveBeenCalledTimes(1);
    expect(getMessageTextStats).not.toHaveBeenCalled();
    expect(createTokenCounter).not.toHaveBeenCalled();
  });

  it('returns null before loading bodies when the branch text is too large', async () => {
    const { createTokenCounter } = jest.requireMock('@librechat/agents');
    const getMessages = jest.fn(async () => [
      {
        messageId: 'message-1',
        parentMessageId: null,
      },
    ]);
    const getMessageTextStats = jest.fn(async () => [textStats('message-1', 512 * 1024 + 1)]);
    const result = await resolveContextProjection(
      {
        userId: 'user-1',
        getMessages,
        getMessageTextStats,
      },
      baseParams,
    );

    expect(result).toBeNull();
    expect(getMessages).toHaveBeenCalledTimes(1);
    expect(getMessageTextStats).toHaveBeenCalledWith(
      {
        conversationId: 'conversation-1',
        user: 'user-1',
        messageId: { $in: ['message-1'] },
      },
      { limit: 1 },
    );
    expect(createTokenCounter).not.toHaveBeenCalled();
  });

  it('loads only branch message bodies after resolving the graph', async () => {
    const graph = [
      { messageId: 'message-1', parentMessageId: null },
      { messageId: 'message-2', parentMessageId: 'message-1' },
      { messageId: 'off-branch', parentMessageId: null },
    ];
    const bodies = [
      {
        messageId: 'message-1',
        parentMessageId: null,
        isCreatedByUser: true,
        text: 'first',
        tokenCount: 5,
      },
      {
        messageId: 'message-2',
        parentMessageId: 'message-1',
        isCreatedByUser: false,
        text: 'second',
        tokenCount: 6,
      },
    ];
    const getMessages = jest.fn(async (_filter: object, select?: string) =>
      select === GRAPH_SELECT ? graph : bodies,
    );
    const getMessageTextStats = jest.fn(async () => [
      textStats('message-1', 5),
      textStats('message-2', 6),
    ]);

    const result = await resolveContextProjection(
      { userId: 'user-1', getMessages, getMessageTextStats },
      { ...baseParams, messageId: 'message-2' },
    );

    expect(result).toEqual({ tokenCount: 1, maxContextTokens: 1000 });
    expect(getMessages).toHaveBeenNthCalledWith(
      1,
      { conversationId: 'conversation-1', user: 'user-1' },
      GRAPH_SELECT,
      { limit: 513, sort: false },
    );
    expect(getMessageTextStats).toHaveBeenCalledWith(
      {
        conversationId: 'conversation-1',
        user: 'user-1',
        messageId: { $in: ['message-1', 'message-2'] },
      },
      { limit: 2 },
    );
    expect(getMessages).toHaveBeenNthCalledWith(
      2,
      {
        conversationId: 'conversation-1',
        user: 'user-1',
        messageId: { $in: ['message-1', 'message-2'] },
      },
      BODY_SELECT,
      { limit: 2, sort: false },
    );
  });

  it('returns null before loading bodies when a branch message has too many quotes', async () => {
    const { createTokenCounter } = jest.requireMock('@librechat/agents');
    const getMessages = jest.fn(async () => [
      {
        messageId: 'message-1',
        parentMessageId: null,
      },
    ]);
    const getMessageTextStats = jest.fn(async () => [
      {
        ...textStats('message-1'),
        quoteCount: QUOTE_MAX_COUNT + 1,
        quoteBytes: 10,
        quoteLineCount: QUOTE_MAX_COUNT + 1,
      },
    ]);

    const result = await resolveContextProjection(
      { userId: 'user-1', getMessages, getMessageTextStats },
      baseParams,
    );

    expect(result).toBeNull();
    expect(getMessages).toHaveBeenCalledTimes(1);
    expect(getMessageTextStats).toHaveBeenCalledTimes(1);
    expect(createTokenCounter).not.toHaveBeenCalled();
  });

  // [cowork] Regression test for the gauge-vs-live-path drift fixed alongside
  // this endpoint's `content`/reduction wiring — see docs/18_context_reduction.md.
  it('recounts a message the reducer truncates instead of trusting its stale stored tokenCount', async () => {
    const { projectAgentContextUsage } = jest.requireMock('@librechat/agents');

    // 5-message branch, tail is message-5. Only message-2 (assistant, with a
    // large tool_call output) needs a `content` array; the rest are plain text
    // turns. userIndices among branch positions = [0,2,3,4] (message-1,3,4,5)
    // -> length 4 > TOOL_TRUNCATE_TURNS default (3) -> boundary = index 2, so
    // message-2 (position 1) falls in the strip zone and gets truncated.
    const bigOutput = 'A'.repeat(5000);
    const graph = [
      { messageId: 'message-1', parentMessageId: null },
      { messageId: 'message-2', parentMessageId: 'message-1' },
      { messageId: 'message-3', parentMessageId: 'message-2' },
      { messageId: 'message-4', parentMessageId: 'message-3' },
      { messageId: 'message-5', parentMessageId: 'message-4' },
    ];
    const bodies = [
      { messageId: 'message-1', isCreatedByUser: true, text: 'read the file please', tokenCount: 5 },
      {
        messageId: 'message-2',
        isCreatedByUser: false,
        text: 'reading it now',
        tokenCount: 6, // stale — reducer will truncate the actual content below
        content: [
          {
            type: 'tool_call',
            tool_call: {
              id: 'call_1',
              name: 'read_file',
              args: JSON.stringify({ path: '/big.txt' }),
              output: bigOutput,
            },
          },
        ],
      },
      { messageId: 'message-3', isCreatedByUser: true, text: 'thanks, next?', tokenCount: 5 },
      { messageId: 'message-4', isCreatedByUser: true, text: 'and one more', tokenCount: 5 },
      { messageId: 'message-5', isCreatedByUser: true, text: 'go', tokenCount: 3 },
    ];
    const getMessages = jest.fn(async (_filter: object, select?: string) =>
      select === GRAPH_SELECT ? graph : bodies,
    );
    const getMessageTextStats = jest.fn(async () =>
      bodies.map((b) => textStats(b.messageId, b.text.length)),
    );

    await resolveContextProjection(
      { userId: 'user-1', getMessages, getMessageTextStats },
      { ...baseParams, messageId: 'message-5' },
    );

    expect(projectAgentContextUsage).toHaveBeenCalledTimes(1);
    const call = projectAgentContextUsage.mock.calls[0][0];
    const indexTokenCountMap = call.indexTokenCountMap as Record<string, number>;

    // message-2 sits at array index 1; its truncated count must be materially
    // smaller than what the untruncated 5000-char output would count as, and
    // must NOT be the stale stored `tokenCount: 6`.
    const fullOutputCount = countFormattedMessageTokens(
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_call',
            tool_call: {
              name: 'read_file',
              args: JSON.stringify({ path: '/big.txt' }),
              output: bigOutput,
            },
          },
        ],
      },
      'o200k_base',
    );
    expect(indexTokenCountMap['1']).not.toBe(6);
    expect(indexTokenCountMap['1']).toBeLessThan(fullOutputCount);

    // Untouched, plain-text messages with a valid stored tokenCount keep it.
    expect(indexTokenCountMap['0']).toBe(5);
    expect(indexTokenCountMap['2']).toBe(5);
  });
});
