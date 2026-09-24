const { logger } = require('@librechat/data-schemas');
const {
  getMissingCustomUserVars,
  requiresEphemeralUserConnection,
  getMissingRuntimeBodyPlaceholderFields,
} = require('@librechat/api');
const { CacheKeys, Constants } = require('librechat-data-provider');
const { getMCPManager, getMCPServersRegistry, getFlowStateManager } = require('~/config');
const { findToken, createToken, updateToken, deleteTokens } = require('~/models');
const { getGraphApiToken } = require('~/server/services/GraphTokenService');
const { exchangeOboToken } = require('~/server/services/OboTokenService');
const { createOboTrustChecker } = require('~/server/services/OboPolicyService');
const {
  getMCPToolsCacheGeneration,
  updateMCPServerTools,
  getMCPServerTools,
} = require('~/server/services/Config');
const { reserveMCPToolsChangedRevision } = require('@librechat/api');
const { getLogStores } = require('~/cache');

const MCP_REINITIALIZE_FAILURE_REASONS = {
  UNREACHABLE: 'unreachable',
  MISSING_CUSTOM_USER_VARS: 'missing_custom_user_vars',
  OAUTH_REQUIRED: 'oauth_required',
  INITIALIZATION_FAILED: 'initialization_failed',
};

/**
 * Reinitializes an MCP server connection and discovers available tools.
 * When OAuth is required, uses discovery mode to list tools without full authentication
 * (per MCP spec, tool listing should be possible without auth).
 * @param {Object} params
 * @param {IUser} params.user - The user from the request object.
 * @param {string} params.serverName - The name of the MCP server
 * @param {boolean} params.returnOnOAuth - Whether to initiate OAuth and return, or wait for OAuth flow to finish
 * @param {AbortSignal} [params.signal] - The abort signal to handle cancellation.
 * @param {boolean} [params.forceNew]
 * @param {number} [params.connectionTimeout]
 * @param {FlowStateManager<any>} [params.flowManager]
 * @param {(authURL: string, options?: { expiresAt?: number }) => Promise<void>} [params.oauthStart]
 * @param {() => Promise<void>} [params.oauthEnd]
 * @param {import('@librechat/api').RequestBody} [params.requestBody]
 * @param {import('@librechat/api').RequestScopedMCPConnectionStore} [params.requestScopedConnections]
 * @param {Record<string, Record<string, string>>} [params.userMCPAuthMap]
 */
async function reinitMCPServer({
  user,
  signal,
  forceNew,
  serverName,
  configServers,
  userMCPAuthMap,
  connectionTimeout,
  returnOnOAuth = true,
  oauthStart: _oauthStart,
  flowManager: _flowManager,
  serverConfig: providedConfig,
  requestBody,
  requestScopedConnections,
  oauthEnd,
}) {
  /** @type {MCPConnection | null} */
  let connection = null;
  let serverConfig = providedConfig;
  /** @type {LCAvailableTools | null} */
  let availableTools = null;
  /** @type {ReturnType<MCPConnection['fetchTools']> | null} */
  let tools = null;
  let oauthRequired = false;
  let oauthUrl = null;
  let oauthExpiresAt;
  let ephemeralServer = false;
  let publicationGeneration;
  let publicationRevision;

  try {
    const registry = getMCPServersRegistry();
    serverConfig =
      serverConfig ?? (await registry.getServerConfig(serverName, user?.id, configServers));
    ephemeralServer = serverConfig ? requiresEphemeralUserConnection(serverConfig) : false;
    if (serverConfig?.inspectionFailed) {
      if (serverConfig.source === 'config') {
        logger.info(
          `[MCP Reinitialize] Config-source server ${serverName} has inspectionFailed — retry handled by config cache`,
        );
        return {
          availableTools: null,
          success: false,
          message: `MCP server '${serverName}' is still unreachable`,
          failureReason: MCP_REINITIALIZE_FAILURE_REASONS.UNREACHABLE,
          oauthRequired: false,
          serverName,
          oauthUrl: null,
          tools: null,
        };
      } else {
        logger.info(
          `[MCP Reinitialize] Server ${serverName} had failed inspection, attempting reinspection`,
        );
        try {
          const storageLocation = serverConfig.source === 'user' ? 'DB' : 'CACHE';
          await registry.reinspectServer(serverName, storageLocation, user?.id);
          logger.info(`[MCP Reinitialize] Reinspection succeeded for server: ${serverName}`);
        } catch (reinspectError) {
          logger.error(
            `[MCP Reinitialize] Reinspection failed for server ${serverName}:`,
            reinspectError,
          );
          return {
            availableTools: null,
            success: false,
            message: `MCP server '${serverName}' is still unreachable`,
            failureReason: MCP_REINITIALIZE_FAILURE_REASONS.UNREACHABLE,
            oauthRequired: false,
            serverName,
            oauthUrl: null,
            tools: null,
          };
        }
      }
    }

    const customUserVars = userMCPAuthMap?.[`${Constants.mcp_prefix}${serverName}`];

    const missingUserVars = getMissingCustomUserVars(serverConfig ?? {}, customUserVars);
    if (missingUserVars.length > 0) {
      logger.warn(
        `[MCP Reinitialize] Skipping server '${serverName}': required user-provided variable(s) not set: ${missingUserVars.join(
          ', ',
        )}. Tools will not be exposed until the user configures them.`,
      );
      return {
        availableTools: null,
        success: false,
        message: `MCP server '${serverName}' requires user-provided variable(s) [${missingUserVars.join(
          ', ',
        )}] which are not set`,
        failureReason: MCP_REINITIALIZE_FAILURE_REASONS.MISSING_CUSTOM_USER_VARS,
        missingUserVars,
        oauthRequired: false,
        serverName,
        oauthUrl: null,
        tools: null,
      };
    }

    /** `{{LIBRECHAT_BODY_*}}` placeholders only resolve during a chat turn; connecting
     *  without them would fail, so defer the connection instead of reporting a failure. */
    const missingBodyFields = serverConfig
      ? getMissingRuntimeBodyPlaceholderFields(serverConfig, requestBody)
      : [];
    if (missingBodyFields.length > 0) {
      logger.info(
        `[MCP Reinitialize] Server '${serverName}' requires request body field(s) [${missingBodyFields.join(
          ', ',
        )}] for runtime placeholders; connection deferred to first use in a chat turn`,
      );
      return {
        availableTools: null,
        success: true,
        /** Lets clients distinguish "connection deferred to a chat turn" from a
         *  plain success with no tools, e.g. to attach the server at the server
         *  level instead of waiting for a tool list that never arrives. */
        connectionDeferred: true,
        message: `MCP server '${serverName}' uses request-scoped placeholders; connection will be established on first use in a chat turn`,
        oauthRequired: false,
        serverName,
        oauthUrl: null,
        tools: null,
      };
    }

    const flowManager = _flowManager ?? getFlowStateManager(getLogStores(CacheKeys.FLOWS));
    const mcpManager = getMCPManager();
    const tokenMethods = { findToken, updateToken, createToken, deleteTokens };

    if (!ephemeralServer) {
      publicationGeneration = await getMCPToolsCacheGeneration({
        userId: user.id,
        serverName,
      });
    }

    const oauthStart =
      _oauthStart ??
      (async (authURL, options) => {
        logger.info(`[MCP Reinitialize] OAuth URL received for ${serverName}`);
        if (authURL !== oauthUrl) {
          oauthExpiresAt = undefined;
        }
        oauthUrl = authURL;
        if (typeof options?.expiresAt === 'number' && Number.isFinite(options.expiresAt)) {
          oauthExpiresAt = options.expiresAt;
        }
        oauthRequired = true;
      });

    try {
      connection = await mcpManager.getConnection({
        user,
        signal,
        forceNew,
        oauthStart,
        serverName,
        flowManager,
        tokenMethods,
        returnOnOAuth,
        oauthEnd,
        customUserVars,
        requestBody,
        requestScopedConnections,
        connectionTimeout,
        serverConfig,
        graphTokenResolver: getGraphApiToken,
        oboTokenResolver: exchangeOboToken,
        oboTrustChecker: createOboTrustChecker(),
      });

      logger.info(`[MCP Reinitialize] Successfully established connection for ${serverName}`);
    } catch (err) {
      logger.info(`[MCP Reinitialize] getConnection threw error: ${err.message}`);
      logger.info(
        `[MCP Reinitialize] OAuth state - oauthRequired: ${oauthRequired}, oauthUrl: ${oauthUrl ? 'present' : 'null'}`,
      );

      const isOAuthError =
        err.message?.includes('OAuth') ||
        err.message?.includes('authentication') ||
        err.message?.includes('401');

      const isOAuthFlowInitiated = err.message === 'OAuth flow initiated - return early';

      if (isOAuthError || oauthRequired || isOAuthFlowInitiated) {
        logger.info(
          `[MCP Reinitialize] OAuth required for ${serverName}, attempting tool discovery without auth`,
        );
        oauthRequired = true;

        try {
          const discoveryResult = await mcpManager.discoverServerTools({
            user,
            signal,
            serverName,
            flowManager,
            tokenMethods,
            oauthStart,
            customUserVars,
            requestBody,
            connectionTimeout,
            configServers,
            graphTokenResolver: getGraphApiToken,
            oboTokenResolver: exchangeOboToken,
            oboTrustChecker: createOboTrustChecker(),
          });

          if (discoveryResult.tools && discoveryResult.tools.length > 0) {
            tools = discoveryResult.tools;
            logger.info(
              `[MCP Reinitialize] Discovered ${tools.length} tools for ${serverName} without full auth`,
            );
          }
        } catch (discoveryErr) {
          logger.debug(
            `[MCP Reinitialize] Tool discovery failed for ${serverName}: ${discoveryErr?.message ?? String(discoveryErr)}`,
          );
        }
      } else {
        logger.error(
          `[MCP Reinitialize] Error initializing MCP server ${serverName} for user:`,
          err,
        );
      }
    }

    if (connection && !oauthRequired) {
      publicationGeneration =
        mcpManager.getToolPublicationGeneration(connection) ?? publicationGeneration;
      // ─────────────────────────────────────────────────────────────────────
      // [cowork] Reserve app-level publish ordering BEFORE the tools/list call
      // (mirrors connection.ts's refreshChangedTools). Without this,
      // updateMCPServerTools() -> replaceAppServerTools() silently no-ops
      // (logger.debug, no warning) for every APP-level (YAML) server
      // reinitialized through this per-message path — tools.ts:375-378
      // requires a non-empty `publicationRevision` and just `return false`s
      // otherwise. The upstream rc1 reinit path here never supplied one.
      //
      // Symptom: the MCP server connects fine, tools/list returns real tools,
      // yet the agent still fails with "configured to use MCP tools, but none
      // are available" — with ZERO error/warn log lines, because the cache
      // write was skipped, not failed. Manually clicking "reinitialize" in the
      // UI works around it (api/server/routes/mcp.js's reinitialize route
      // calls invalidateCachedTools() first, which forces the connection's
      // OWN toolsChanged event — a separate path that DOES reserve a revision
      // via connection.ts — to repopulate the cache on the next read).
      // Found live 2026-09-11 on the `rag` server. See
      // docs/22_librecowork_fork_changes.md §32. `reserveMCPToolsChangedRevision`
      // itself no-ops (returns undefined) for genuinely user-scoped servers,
      // so calling it unconditionally here is safe.
      publicationRevision = await reserveMCPToolsChangedRevision({ serverName, serverConfig });
      // ─────────────────────────────────────────────────────────────────────
      let snapshot;
      if (typeof connection.fetchOrderedToolsSnapshot === 'function') {
        snapshot = await connection.fetchOrderedToolsSnapshot();
      } else if (typeof connection.fetchToolsSnapshot === 'function') {
        snapshot = await connection.fetchToolsSnapshot();
      } else {
        snapshot = { tools: await connection.fetchTools(), complete: true };
      }
      if (snapshot.complete) {
        tools = snapshot.tools;
      } else {
        logger.warn(
          `[MCP Reinitialize] Preserving cached tools for ${serverName} because tools/list returned an incomplete snapshot`,
        );
      }

      // ─────────────────────────────────────────────────────────────────────────
      // [cowork] BEGIN v0.8.8-rc1 MCP cold-catalog regression guard.
      // ‼️  REMOVE THIS BLOCK when merging upstream LibreChat v0.8.8 (rc2 or final).
      //     rc2 fixes this properly via packages/api/src/mcp/catalog/recovery.ts
      //     (#15323 "Recover Missing MCP Catalogs" + #15429 "Shared Gate"), which
      //     rediscovers via a fresh ephemeral connection instead of the pooled one.
      //     See docs/22_librecowork_fork_changes.md §28.
      //
      // Symptom: after a long idle (overnight), the first message fails with
      //   "[ResumableAgentController] ... configured to use MCP tools, but none are
      //    available". Every server logs "Successfully established connection"
      //    (twice — getOrFetch + refresh), retrying never recovers it, only a
      //    container restart does.
      //
      // Root cause: for APP-level (YAML) servers, MCPManager.getConnection returns
      //   the pooled MCPConnection verbatim and ignores `forceNew` (only
      //   getUserConnection honors it). ConnectionsRepository only rebuilds on
      //   isStale()/!isConnected(). So once a pooled app connection reports
      //   isConnected()===true but its published tool catalog is [] (an idle
      //   reconnect published an empty list, or tools/list briefly returned none),
      //   nothing in the request path — not `forceNew`, not refreshToolList(), not
      //   the tool cache (also empty) — can rebuild it. Only disconnectAppServers()
      //   (process shutdown) clears it.
      //
      // Fix: when an app server hands back an empty catalog, force a real transport
      //   rebuild on the connection object itself (connect() disconnects + rebuilds
      //   + re-runs initialize) and read tools/list directly. Always log the
      //   outcome at info level so a persistent failure is fully explained.
      const gotNoUsableTools = !Array.isArray(tools) || tools.length === 0;
      if (connection && !oauthRequired && !ephemeralServer && gotNoUsableTools) {
        const snapCount = Array.isArray(snapshot?.tools) ? snapshot.tools.length : 'n/a';
        let rebuiltCount = 'n/a';
        let rebuiltComplete = 'n/a';
        let connectedAfter = 'n/a';
        try {
          if (typeof connection.connect === 'function') {
            await connection.connect();
          } else if (typeof connection.refreshToolList === 'function') {
            await connection.refreshToolList();
          }
          if (typeof connection.isConnected === 'function') {
            connectedAfter = String(await connection.isConnected());
          }
          let rebuilt;
          if (typeof connection.fetchToolsSnapshot === 'function') {
            rebuilt = await connection.fetchToolsSnapshot();
          } else {
            rebuilt = { tools: await connection.fetchTools(), complete: true };
          }
          rebuiltComplete = String(rebuilt?.complete);
          rebuiltCount = Array.isArray(rebuilt?.tools) ? String(rebuilt.tools.length) : 'n/a';
          if (Array.isArray(rebuilt?.tools) && rebuilt.tools.length > 0) {
            tools = rebuilt.tools;
          }
        } catch (rebuildErr) {
          logger.warn(
            `[MCP Reinitialize] [cowork] cold-catalog guard: forced reconnect for ${serverName} threw: ${
              rebuildErr?.message ?? String(rebuildErr)
            }`,
          );
        }

        if (!Array.isArray(tools) || tools.length === 0) {
          tools = null;
          const cachedAvailableTools = await getMCPServerTools(user.id, serverName, serverConfig);
          if (cachedAvailableTools && Object.keys(cachedAvailableTools).length > 0) {
            availableTools = cachedAvailableTools;
          }
          logger.warn(
            `[MCP Reinitialize] [cowork] cold-catalog guard EXHAUSTED for ${serverName}: ` +
              `snapshot.complete=${snapshot?.complete} snapshot.tools=${snapCount} ` +
              `rebuilt.complete=${rebuiltComplete} rebuilt.tools=${rebuiltCount} ` +
              `connectedAfterRebuild=${connectedAfter} ` +
              `cacheFallback=${availableTools ? Object.keys(availableTools).length : 0} tool(s)`,
          );
        } else {
          logger.info(
            `[MCP Reinitialize] [cowork] cold-catalog guard RECOVERED ${tools.length} tool(s) for ${serverName} ` +
              `via forced reconnect (snapshot.tools was ${snapCount}, connectedAfter=${connectedAfter})`,
          );
        }
      }
      // [cowork] END v0.8.8-rc1 MCP cold-catalog regression guard.
      // ─────────────────────────────────────────────────────────────────────────
    }

    if (tools && !ephemeralServer && publicationGeneration) {
      const currentGeneration = await getMCPToolsCacheGeneration({
        userId: user.id,
        serverName,
      });
      if (currentGeneration !== publicationGeneration) {
        logger.warn(
          `[MCP Reinitialize] Discarding stale tools for ${serverName} because its publication generation changed during discovery`,
        );
        tools = null;
      }
    }

    if (tools) {
      availableTools = await updateMCPServerTools({
        userId: user.id,
        serverName,
        tools,
        serverConfig,
        ...(publicationGeneration && { publicationGeneration }),
        // [cowork] see the reservation comment above — required for
        // replaceAppServerTools() to actually persist app-level catalogs.
        ...(publicationRevision && { publicationRevision }),
      });
      if (availableTools == null) {
        tools = null;
      }
    }

    logger.debug(
      `[MCP Reinitialize] Sending response for ${serverName} - oauthRequired: ${oauthRequired}, oauthUrl: ${oauthUrl ? 'present' : 'null'}`,
    );

    const getResponseMessage = () => {
      if (oauthRequired && tools && tools.length > 0) {
        return `MCP server '${serverName}' tools discovered, OAuth required for execution`;
      }
      if (oauthRequired) {
        return `MCP server '${serverName}' ready for OAuth authentication`;
      }
      if (connection) {
        return `MCP server '${serverName}' reinitialized successfully`;
      }
      return `Failed to reinitialize MCP server '${serverName}'`;
    };

    const success = Boolean(
      (connection && !oauthRequired) || (oauthRequired && oauthUrl) || (tools && tools.length > 0),
    );
    let failureReason;
    if (!success) {
      failureReason = oauthRequired
        ? MCP_REINITIALIZE_FAILURE_REASONS.OAUTH_REQUIRED
        : MCP_REINITIALIZE_FAILURE_REASONS.INITIALIZATION_FAILED;
    }
    const result = {
      availableTools,
      success,
      message: getResponseMessage(),
      failureReason,
      oauthRequired,
      serverName,
      oauthUrl,
      oauthExpiresAt,
      tools,
    };

    logger.debug(`[MCP Reinitialize] Response for ${serverName}:`, {
      success: result.success,
      oauthRequired: result.oauthRequired,
      oauthUrl: result.oauthUrl ? 'present' : null,
      toolsCount: tools?.length ?? 0,
    });

    return result;
  } catch (error) {
    logger.error(
      '[MCP Reinitialize] Error loading MCP Tools, servers may still be initializing:',
      error,
    );
  } finally {
    if (connection && ephemeralServer && !requestScopedConnections) {
      try {
        await connection.dispose();
      } catch (error) {
        logger.warn(`[MCP Reinitialize] Failed to dispose ephemeral server ${serverName}`, error);
      }
    }
  }
}

module.exports = {
  reinitMCPServer,
};
