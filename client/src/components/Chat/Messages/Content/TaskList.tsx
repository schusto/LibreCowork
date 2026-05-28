import { useState, useEffect, useRef, useMemo } from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { cn } from '~/utils';

const TODO_POLL_INTERVAL_MS = 1000;
const TODO_STATUS_URL = '/api/todo_status';

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
}

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

// ── Main component ─────────────────────────────────────────────────────────

/**
 * TaskList renders a `todo_update` tool call as a structured task progress widget.
 *
 * The model calls `todo_update` with a `todos` array containing the full list
 * on each update. This component parses the latest snapshot and renders it
 * as a compact card, replacing the generic ToolCall UI.
 *
 * To enable this widget, configure the model/agent with a tool named
 * `todo_update` that accepts:
 *   { todos: Array<{ content: string, activeForm: string, status: 'pending'|'in_progress'|'completed' }> }
 */
export default function TaskList({ args, isSubmitting }: TaskListProps) {
  // Todos parsed from the parent message's content (last todo_update the parent
  // agent called directly — may be stale if sub-agents have since updated).
  const contentTodos = useMemo(() => parseTodos(args), [args]);

  // Polled todos from mlx-proxy → todos.json (reflects ALL agents, including
  // sub-agents whose tool calls never appear in this conversation's content).
  const [polledTodos, setPolledTodos] = useState<Todo[] | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!isSubmitting) {
      // Stop polling when the assistant has finished; keep last polled state
      // so the widget remains accurate after the final sub-agent turn.
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const res = await fetch(TODO_STATUS_URL, { cache: 'no-store' });
        const data = await res.json();
        if (Array.isArray(data?.todos) && data.todos.length > 0) {
          setPolledTodos(data.todos as Todo[]);
        }
      } catch {
        // ignore — keep last known state
      }
    };

    poll();
    pollTimerRef.current = setInterval(poll, TODO_POLL_INTERVAL_MS);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [isSubmitting]);

  // Prefer the polled state (always current) over the content-derived state.
  const todos = (polledTodos && polledTodos.length > 0) ? polledTodos : contentTodos;

  if (todos.length === 0) {
    return null;
  }

  const completedCount = todos.filter((t) => t.status === 'completed').length;
  const total = todos.length;
  const allDone = completedCount === total;
  const hasActive = todos.some((t) => t.status === 'in_progress');

  return (
    <div
      className={cn(
        'my-2 rounded-lg border bg-surface-secondary px-4 py-3',
        'border-border-light dark:border-border-medium',
      )}
      role="status"
      aria-label="Task progress"
    >
      {/* Header */}
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
          {allDone ? 'Tasks complete' : hasActive || isSubmitting ? 'Working…' : 'Tasks'}
        </span>
        <span className="text-xs text-text-secondary">
          {completedCount}/{total}
        </span>
      </div>

      {/* Task list */}
      <ul className="space-y-1.5">
        {todos.map((todo, i) => (
          <TodoRow key={`${todo.content}-${i}`} todo={todo} />
        ))}
      </ul>
    </div>
  );
}
