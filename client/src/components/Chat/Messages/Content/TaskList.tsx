import { useState, useEffect, useRef, useMemo, createContext, useContext, useCallback } from 'react';
import { CheckCircle2, Circle, Loader2, ListChecks, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '~/utils';

// ── Types ──────────────────────────────────────────────────────────────────

type TodoStatus = 'pending' | 'in_progress' | 'completed';

interface Todo {
  content: string;
  activeForm: string;
  status: TodoStatus;
}

interface TaskListProps {
  /** Raw JSON string from the tool call args */
  args: string | Record<string, unknown>;
  /** True while the assistant is still streaming */
  isSubmitting: boolean;
  /** Conversation scope for the collapse-state key (design §3.4). Optional:
   *  inline/nested instances without a known conversation keep local-only state. */
  conversationId?: string | null;
  /** Enable the scoped store channel (/api/tasks). Dormant by default: the
   *  store is TURN-scoped today (parent and subagents share one doc — design
   *  §0b gate 2), so polling it into the parent widget would violate D2. Flip
   *  on when per-actor keys land. */
  enableStore?: boolean;
  /** Message owning this widget (required with enableStore). */
  messageId?: string;
}

/** How todo tool parts should render inside this subtree (see Part.tsx).
 *  'suppressed' (root chat): the sticky bottom widget owns the display, so
 *  inline cards would duplicate it. 'inline' (subagent thread panels, activity
 *  phases): render each thread's OWN list from its own content args — D2. */
export const TodoRenderModeContext = createContext<'suppressed' | 'inline'>('suppressed');
export const useTodoRenderMode = () => useContext(TodoRenderModeContext);

// ── Helpers ────────────────────────────────────────────────────────────────

function parseTodos(args: string | Record<string, unknown>): Todo[] {
  try {
    const parsed = typeof args === 'string' ? JSON.parse(args) : args;
    const todos = parsed?.todos;
    if (!Array.isArray(todos)) {
      return [];
    }
    return todos.filter(
      (t): t is Todo =>
        typeof t?.content === 'string' &&
        typeof t?.status === 'string' &&
        ['pending', 'in_progress', 'completed'].includes(t.status),
    );
  } catch {
    return [];
  }
}

const COLLAPSE_STORAGE_PREFIX = 'tasklist:';

function loadCollapsedPref(conversationId?: string | null): boolean {
  // Absent key ⇒ collapsed (new-chat default, D3).
  if (!conversationId) return true;
  try {
    return localStorage.getItem(COLLAPSE_STORAGE_PREFIX + conversationId) !== 'expanded';
  } catch {
    return true;
  }
}

function saveCollapsedPref(conversationId: string | null | undefined, collapsed: boolean) {
  if (!conversationId) return;
  try {
    localStorage.setItem(COLLAPSE_STORAGE_PREFIX + conversationId, collapsed ? 'collapsed' : 'expanded');
  } catch {
    /* private mode etc. — preference simply won't persist */
  }
}

// ── Sub-components ─────────────────────────────────────────────────────────

function StatusIcon({ status }: { status: TodoStatus }) {
  if (status === 'completed') {
    return (
      <CheckCircle2
        className="mt-0.5 size-4 shrink-0 text-green-500 dark:text-green-400"
        aria-hidden="true"
      />
    );
  }
  if (status === 'in_progress') {
    return (
      <Loader2
        className="mt-0.5 size-4 shrink-0 animate-spin text-blue-500 dark:text-blue-400"
        aria-hidden="true"
      />
    );
  }
  // pending
  return (
    <Circle
      className="mt-0.5 size-4 shrink-0 text-gray-400 dark:text-gray-500"
      aria-hidden="true"
    />
  );
}

function TodoRow({ todo }: { todo: Todo }) {
  const label = todo.status === 'in_progress' ? todo.activeForm : todo.content;

  return (
    <li
      className={cn(
        'flex items-start gap-2.5 text-sm',
        todo.status === 'completed' && 'text-gray-400 line-through dark:text-gray-500',
        todo.status === 'in_progress' && 'font-medium text-text-primary',
        todo.status === 'pending' && 'text-text-secondary',
      )}
    >
      <StatusIcon status={todo.status} />
      <span>{label}</span>
    </li>
  );
}

// ── Scoped store hook (design §3.2 widget read path) ─────────────────────

/** Poll /api/tasks while submitting; fetch once on mount; stop when settled.
 *  Returns null whenever disabled or nothing fetched yet (⇒ caller falls back
 *  to content state). Errors keep last known state (failure posture §3.3). */
function useScopedTasks(opts: {
  enableStore?: boolean;
  conversationId?: string | null;
  messageId?: string;
  isSubmitting: boolean;
}): Todo[] | null {
  const { enableStore, conversationId, messageId, isSubmitting } = opts;
  const [todos, setTodos] = useState<Todo[] | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enableStore || !conversationId) {
      return;
    }
    let cancelled = false;

    const fetchOnce = async () => {
      try {
        const params = new URLSearchParams({ conversationId });
        if (messageId) params.set('messageId', messageId);
        const res = await fetch(`/api/tasks?${params.toString()}`, { cache: 'no-store' });
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (!cancelled && Array.isArray(data?.todos)) {
          setTodos(data.todos as Todo[]);
        }
      } catch {
        /* keep last known state */
      }
    };

    fetchOnce();
    return () => {
      cancelled = true;
    };
  }, [enableStore, conversationId, messageId]);

  useEffect(() => {
    if (!enableStore || !conversationId || !isSubmitting) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    const poll = async () => {
      try {
        const params = new URLSearchParams({ conversationId });
        if (messageId) params.set('messageId', messageId);
        const res = await fetch(`/api/tasks?${params.toString()}`, { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data?.todos)) setTodos(data.todos as Todo[]);
      } catch {
        /* ignore — keep last known state */
      }
    };
    poll();
    timerRef.current = setInterval(poll, 1000);
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [enableStore, conversationId, messageId, isSubmitting]);

  return enableStore ? todos : null;
}

// ── Main component ────────────────────────────────────────────────────────

/** Visible-row budget before the list scrolls (design §3.4: ≈5 rows). */
const MAX_VISIBLE_ROWS = 5;
/** Auto-collapse delay once every task is done (§3.4). */
const ALL_DONE_COLLAPSE_MS = 5000;

/**
 * TaskList renders a `todo_update` tool call as a structured task progress widget.
 *
 * Data sources, in precedence order:
 *   1. Scoped store (`enableStore`, default off — see props): /api/tasks under
 *      the injected scope. Off because the store is turn-scoped today; flipping
 *      it on is one prop once per-actor keys land (design §0b gate 2).
 *   2. Content args (always): the latest todo_update snapshot in this subtree.
 *      Nested subagent threads render exclusively from their own args (D2),
 *      which is why inline instances never fetch.
 *
 * Behaviour (design §3.4, D3): default collapsed; expanded state remembered per
 * conversation in localStorage; long lists cap at ~5 visible rows with scrolling
 * anchored so the newest completed row sits on top (user touch suspends
 * re-anchoring until submission ends); all-done auto-collapses after 5 s.
 */
export default function TaskList({
  args,
  isSubmitting,
  conversationId,
  messageId,
  enableStore,
}: TaskListProps) {
  const contentTodos = useMemo(() => parseTodos(args), [args]);
  const storedTodos = useScopedTasks({ enableStore, conversationId, messageId, isSubmitting });
  const todos = storedTodos && storedTodos.length > 0 ? storedTodos : contentTodos;

  const [collapsed, setCollapsed] = useState<boolean>(() => loadCollapsedPref(conversationId));
  const toggle = useCallback(() => {
    setCollapsed((current) => {
      const next = !current;
      saveCollapsedPref(conversationId, next);
      return next;
    });
  }, [conversationId]);

  // Auto-collapse shortly after everything is done (§3.4). Only expands-state
  // cards are affected; a collapsed card stays collapsed.
  const allDone = todos.length > 0 && todos.every((t) => t.status === 'completed');
  useEffect(() => {
    if (!allDone || collapsed) return;
    const timer = setTimeout(() => setCollapsed(true), ALL_DONE_COLLAPSE_MS);
    return () => clearTimeout(timer);
  }, [allDone, collapsed]);

  // Anchor scroll: keep the most-recently-completed row at the top of the
  // scrolled viewport. User interaction with the scroller suspends re-anchoring
  // until submission ends (then it resumes for the next snapshot).
  const listRef = useRef<HTMLUListElement | null>(null);
  const userTouchedRef = useRef(false);
  const lastAnchorKey = useRef<string | null>(null);

  const markUserTouch = useCallback(() => {
    userTouchedRef.current = true;
  }, []);
  useEffect(() => {
    if (!isSubmitting) userTouchedRef.current = false;
  }, [isSubmitting]);

  const anchorIndex = useMemo(() => {
    for (let i = todos.length - 1; i >= 0; i--) {
      if (todos[i].status === 'completed') return i;
    }
    return -1;
  }, [todos]);

  useEffect(() => {
    if (collapsed || anchorIndex < 0 || userTouchedRef.current) return;
    const list = listRef.current;
    if (!list) return;
    const anchorKey = `${conversationId ?? ''}:${messageId ?? ''}:${anchorIndex}`;
    if (lastAnchorKey.current === anchorKey) return;
    lastAnchorKey.current = anchorKey;
    const row = list.children[anchorIndex] as HTMLElement | undefined;
    if (row) list.scrollTop = Math.max(0, row.offsetTop - list.offsetTop);
  }, [collapsed, anchorIndex, todos, conversationId, messageId]);

  if (todos.length === 0) {
    return null;
  }

  const completedCount = todos.filter((t) => t.status === 'completed').length;
  const total = todos.length;
  const hasActive = todos.some((t) => t.status === 'in_progress');
  const active = todos.find((t) => t.status === 'in_progress');
  const headerLabel = allDone ? 'Tasks complete' : hasActive || isSubmitting ? 'Working…' : 'Tasks';

  // Collapsed: icon + counter + activeForm of the running task only (§3.4).
  if (collapsed) {
    return (
      <div className="my-2 rounded-lg border bg-surface-secondary px-3 py-1.5 border-border-light dark:border-border-medium">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={false}
          aria-label={`Expand task list (${completedCount}/${total})`}
          className="flex w-full items-center gap-2 text-left"
        >
          <ChevronRight className="size-4 shrink-0 text-text-secondary" aria-hidden="true" />
          <ListChecks className="size-4 shrink-0 text-text-secondary" aria-hidden="true" />
          <span className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
            {headerLabel}
          </span>
          {active != null && (
            <span className="min-w-0 flex-1 truncate text-sm font-medium text-text-primary">
              {active.activeForm}
            </span>
          )}
          <span className="ml-auto text-xs text-text-secondary">
            {completedCount}/{total}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'my-2 rounded-lg border bg-surface-secondary px-4 py-3',
        'border-border-light dark:border-border-medium',
        'max-h-[35vh] md:max-h-none overflow-hidden flex flex-col',
      )}
      role="status"
      aria-label="Task progress"
    >
      {/* Header */}
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={toggle}
          aria-expanded
          aria-label="Collapse task list"
          className="flex min-w-0 items-center gap-1.5"
        >
          <ChevronDown className="size-4 shrink-0 text-text-secondary" aria-hidden="true" />
          <span className="truncate text-xs font-semibold uppercase tracking-wide text-text-secondary">
            {headerLabel}
          </span>
        </button>
        <span className="text-xs text-text-secondary">
          {completedCount}/{total}
        </span>
      </div>

      {/* Task list: capped at ~5 rows, scrollable with anchor behaviour */}
      <ul
        ref={listRef}
        onWheel={markUserTouch}
        onTouchStart={markUserTouch}
        className="space-y-1.5 overflow-y-auto"
        style={{ maxHeight: `calc(${MAX_VISIBLE_ROWS} * 1.75rem)` }}
      >
        {todos.map((todo, i) => (
          <TodoRow key={`${todo.content}-${i}`} todo={todo} />
        ))}
      </ul>
    </div>
  );
}
