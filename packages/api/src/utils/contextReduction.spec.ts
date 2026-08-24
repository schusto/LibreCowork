import { applyContextReductionCore } from './contextReduction';
import type { ReducibleMessage } from './contextReduction';

/**
 * These tests exercise the shared reduction core directly (bypassing the
 * `CWK_*` env vars entirely, since `applyContextReductionCore` reads no env
 * vars of its own — those are read once at module import for the *boundary
 * window sizes* (THINK_KEEP_TURNS etc.), which default to 2 / 3 / 5 / 2 user
 * turns respectively; see contextReduction.ts). Fixtures below are sized
 * against those defaults.
 */

function userMsg(text = 'hi'): ReducibleMessage {
  return { role: 'user', content: [{ type: 'text', text }] };
}

function toolCallMsg(name: string, output: string, path = '/foo.txt'): ReducibleMessage {
  return {
    role: 'assistant',
    content: [
      {
        type: 'tool_call',
        tool_call: { id: 'call_1', name, args: JSON.stringify({ path }), output },
      },
    ],
  };
}

function thinkAndTextMsg(think: string, text: string): ReducibleMessage {
  return {
    role: 'assistant',
    content: [
      { type: 'think', think },
      { type: 'text', text },
    ],
  };
}

describe('applyContextReductionCore', () => {
  it('is a no-op on an empty array', () => {
    const result = applyContextReductionCore([]);
    expect(result.messages).toEqual([]);
    expect(result.stats).toEqual({
      thinkStripped: 0,
      toolDeduped: 0,
      toolTruncated: 0,
      savedChars: 0,
    });
  });

  it('keeps every message reference-identical when there are too few user turns to strip anything', () => {
    const messages: ReducibleMessage[] = [
      userMsg(),
      thinkAndTextMsg('reasoning...', 'answer'),
      toolCallMsg('read_file', 'A'.repeat(5000)),
    ];
    const result = applyContextReductionCore(messages);
    expect(result.messages).toHaveLength(3);
    for (let i = 0; i < messages.length; i++) {
      expect(result.messages[i]).toBe(messages[i]);
    }
    expect(result.stats).toEqual({
      thinkStripped: 0,
      toolDeduped: 0,
      toolTruncated: 0,
      savedChars: 0,
    });
  });

  it('strips thinking blocks outside the keep window but leaves recent ones intact', () => {
    // THINK_KEEP_TURNS defaults to 2 — need > 2 user turns for a strip zone to exist.
    const oldThinking = thinkAndTextMsg('old reasoning', 'old answer');
    const recentThinking = thinkAndTextMsg('recent reasoning', 'recent answer');
    const messages: ReducibleMessage[] = [
      userMsg('turn 1'),
      oldThinking,
      userMsg('turn 2'),
      userMsg('turn 3'),
      recentThinking,
      userMsg('turn 4'),
    ];

    const result = applyContextReductionCore(messages);

    const reducedOld = result.messages[1];
    expect(reducedOld).not.toBe(oldThinking);
    expect(Array.isArray(reducedOld.content) ? reducedOld.content : []).not.toContainEqual(
      expect.objectContaining({ type: 'think' }),
    );

    const reducedRecent = result.messages[4];
    expect(reducedRecent).toBe(recentThinking);
    expect(Array.isArray(reducedRecent.content) ? reducedRecent.content : []).toContainEqual(
      expect.objectContaining({ type: 'think', think: 'recent reasoning' }),
    );

    expect(result.stats.thinkStripped).toBe(1);
  });

  it('truncates a filesystem tool result outside the keep window', () => {
    // TOOL_TRUNCATE_TURNS defaults to 3 — need > 3 user turns for a strip zone.
    const bigRead = toolCallMsg('read_file', 'X'.repeat(5000));
    const messages: ReducibleMessage[] = [
      userMsg('turn 1'),
      bigRead,
      userMsg('turn 2'),
      userMsg('turn 3'),
      userMsg('turn 4'),
    ];

    const result = applyContextReductionCore(messages);
    const reduced = result.messages[1];

    expect(reduced).not.toBe(bigRead);
    expect(result.stats.toolTruncated).toBe(1);
    const content = reduced.content;
    expect(Array.isArray(content)).toBe(true);
    if (Array.isArray(content)) {
      const output = (content[0] as { tool_call?: { output?: string } }).tool_call?.output ?? '';
      expect(output.length).toBeLessThan(5000);
      expect(output).toContain('truncated by context reducer');
    }
  });

  it('dedupes an identical repeated filesystem read when both reads fall outside the keep window', () => {
    const output = 'Y'.repeat(400); // >= TOOL_DEDUP_MIN_CHARS default (300)
    const firstRead = toolCallMsg('read_file', output, '/same.txt');
    const secondRead = toolCallMsg('read_file', output, '/same.txt');
    // Need > TOOL_TRUNCATE_TURNS (3) user turns AFTER both reads for the keep
    // boundary to land past index 3, putting both reads in the strip zone.
    // userIndices = [0,2,4,5,6] -> boundary = userIndices[5-3] = userIndices[2] = 4.
    const messages: ReducibleMessage[] = [
      userMsg('turn 1'),
      firstRead,
      userMsg('turn 2'),
      secondRead,
      userMsg('turn 3'),
      userMsg('turn 4'),
      userMsg('turn 5'),
    ];

    const result = applyContextReductionCore(messages);

    // First occurrence seeds the dedup map but isn't itself rewritten.
    expect(result.messages[1]).toBe(firstRead);

    const secondReduced = result.messages[3];
    expect(secondReduced).not.toBe(secondRead);
    expect(result.stats.toolDeduped).toBe(1);
    const content = secondReduced.content;
    if (Array.isArray(content)) {
      const dedupedOutput =
        (content[0] as { tool_call?: { output?: string } }).tool_call?.output ?? '';
      expect(dedupedOutput).toContain('identical to earlier call in this session');
    }
  });

  it('leaves RAG and web search results outside the tool-truncate window alone within their own longer/shorter windows', () => {
    const ragMsg: ReducibleMessage = {
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          tool_call: { id: 'c1', name: 'search_knowledge_base', args: '{}', output: 'Z'.repeat(500) },
        },
      ],
    };
    // Only 2 user turns after ragMsg — RAG_TRUNCATE_TURNS defaults to 5, so this
    // stays inside the RAG keep window (boundary is -1: "keep everything").
    const messages: ReducibleMessage[] = [userMsg('turn 1'), ragMsg, userMsg('turn 2')];

    const result = applyContextReductionCore(messages);
    expect(result.messages[1]).toBe(ragMsg);
    expect(result.stats.toolTruncated).toBe(0);
  });
});
