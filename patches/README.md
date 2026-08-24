# Local dependency patches

Patches applied to `node_modules` after install. Named with the
[patch-package](https://github.com/ds300/patch-package) convention
(`<package>+<version>.patch`) so they can be handed to that tool later, but
they are applied with plain `git apply` — the image has `git` and no `patch`
binary, and this avoids adding a dependency plus a lockfile change.

The Dockerfile applies every file here after `COPY . .` (the source tree,
including this directory, does not exist yet at `npm ci` time, which is why
this is a separate step rather than a `postinstall` script).

**Working outside Docker** (running `npm ci` on a host) — apply them by hand:

```bash
for p in patches/*.patch; do git apply -p1 "$p"; done
```

`git apply` fails loudly on an already-applied or non-matching patch, which is
the intended behaviour: a silently skipped patch would mean silently losing the
behaviour it adds. Add `--check` first if you want to test without applying.

## `@librechat+agents+3.4.7.patch`

Makes the SDK act on the `Stop` hook decision it already computes.

`@librechat/agents` fully implements Claude-Code-style stop hooks — its own
types declare `StopDecision = 'continue' | 'block'` with `block` meaning "do
not stop, run another turn", and `executeHooks.cjs` folds every registered
hook's decision into `stopDecision` correctly. But `run.cjs` calls the `Stop`
hook as `executeHooks({...}).catch(() => {})` and never reads the result, so
`block` is computed and thrown away. Verified present in 3.2.54, the pinned
3.4.7, and the latest published 3.7.1 — this is not fixed by upgrading.

The patch wraps the stream consumption inside `processStream`'s `consumeStream`
in a loop: on `stopDecision === 'block'` it appends the hook's `reason` as a
nudge message and re-enters `streamEvents` in place, bounded by a runaway
backstop of 10 rounds. Continuing *inside* `consumeStream` is what makes it
correct — no `resetValues()` and no `clearHeavyState()` has run at that point,
so the Graph's content indices keep appending and its handler registry is still
live, and the continuation streams to the client like an ordinary extra turn.

Used by the harness supervisor's stall recovery
(`api/server/utils/harnessSupervisor.js`, `createStallRecoveryStopHook`). See
`docs/41_stall_recovery_stop_hook_report.md` for the full analysis, including
why the previous host-side `processStream()` re-entry could not work.

**Without the patch the app still runs correctly.** The hook fires and logs,
its decision is discarded, and the run finalizes exactly as stock — the feature
is inert rather than broken. Worth reporting upstream rather than carrying
locally forever.
