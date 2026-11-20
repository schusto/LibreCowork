import { useRef, useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { useStore } from 'jotai';
import { useNavigate } from 'react-router-dom';
import { QueryKeys } from 'librechat-data-provider';
import { useQueryClient } from '@tanstack/react-query';
import { launchedConversationFamily } from '@librechat/chat/react';
import type { ChatTurnsParams, TurnServices } from '@librechat/chat/react';
import type { TEndpointsConfig } from 'librechat-data-provider';
import {
  logger,
  requestChatFocus,
  markPasteSubmitted,
  bindComposerLaunch,
  getRouteChatProjectId,
} from '~/utils';
import useGetConversation from '~/hooks/Conversations/useGetConversation';
import useDiscardPastedFiles from '~/hooks/Files/useDiscardPastedFiles';
import useCodeApprovalMode from '~/hooks/Agents/useCodeApprovalMode';
import useSetFilesToDelete from '~/hooks/Files/useSetFilesToDelete';
import { useAgentsMapContext } from '~/Providers/AgentsMapContext';
import { useChatSettings } from '~/Providers/ChatSettingsContext';
// [cowork] Context-compaction toggle atom (docs/changes/2026-11-context-reduction*).
// Aliased: this module's local `store` is the Jotai store for composer-launch subs.
import recoilStore from '~/store';
import useCodeWorkspace from '~/hooks/Agents/useCodeWorkspace';
import { cancelUploads } from '~/hooks/Files/useFileHandling';
import useUserKey from '~/hooks/Input/useUserKey';
import { isTargetReady } from './ask/guards';

/** The app's side of a pane's turns: its catalogs, router, composer files, code workspace and
 *  logger, for the package's `useTurns`. */
export default function useTurnServices({
  index,
  conversation,
}: Pick<ChatTurnsParams, 'index' | 'conversation'>): TurnServices {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const discardFiles = useDiscardPastedFiles(index, conversation?.conversationId);
  const setFilesToDelete = useSetFilesToDelete();
  const agentsMap = useAgentsMapContext();
  const { isTemporary, config, auth, addedConversation } = useChatSettings();
  const { getEphemeralAgent: getBaseEphemeralAgent } = useChatSettings();
  // [cowork] Per-chat context-reduction setting; rides each send as
  // ephemeralAgent.contextReduction (reader: api/server/utils/contextReduction.js).
  const contextReductionSetting = useRecoilValue(recoilStore.contextReduction);
  const { getExpiry } = useUserKey(conversation?.endpoint ?? '');
  const getConversation = useGetConversation(index);
  const codeWorkspaceState = useCodeWorkspace(conversation, addedConversation);
  const codeApprovalState = useCodeApprovalMode(
    conversation,
    addedConversation,
    codeWorkspaceState.mode,
  );

  /** A send reads the newest render's values, even through a `readCodeInputs` it captured from an
   *  earlier render. */
  const codeReaders = useRef({
    workspace: codeWorkspaceState.readInputs,
    approval: codeApprovalState.readInputs,
  });
  codeReaders.current = {
    workspace: codeWorkspaceState.readInputs,
    approval: codeApprovalState.readInputs,
  };

  const readiness = useRef({ config, addedConversation });
  readiness.current = { config, addedConversation };

  const store = useStore();
  /** The composer draft follows the conversation a submission launches; the package writes the
   *  launch just before it publishes the submission, and the subscription binds it inside that
   *  write. */
  useEffect(() => {
    const launched = launchedConversationFamily(index);
    return store.sub(launched, () => bindComposerLaunch(store.get(launched), index));
  }, [store, index]);

  return {
    logger,
    isTemporary,
    user: auth.user,
    modelSpecs: config.modelSpecs,
    getEndpointsConfig: () => queryClient.getQueryData<TEndpointsConfig>([QueryKeys.endpoints]),
    agentsMap,
    /** [cowork] Fork seam: upstream moved turn assembly into @librechat/chat/react, whose
     *  `services.getEphemeralAgent(convoKey)` is now the ONLY place an agent submit attaches
     *  the ephemeral agent (formerly built inline in ask(), hooks/Chat/useChatFunctions.ts). */
    getEphemeralAgent: (conversationId: string) => ({
      ...(getBaseEphemeralAgent(conversationId) ?? {}),
      contextReduction: contextReductionSetting,
    }),
    getUserKey: getExpiry,
    getRouteProjectId: getRouteChatProjectId,
    canSubmit: (submitted) => {
      const { config, addedConversation } = readiness.current;
      return (
        config.loaded !== false &&
        isTargetReady(queryClient, submitted) &&
        (addedConversation == null || isTargetReady(queryClient, addedConversation))
      );
    },
    readCodeInputs: () => ({
      workspace: codeReaders.current.workspace(),
      approval: codeReaders.current.approval(),
      latest: getConversation(),
    }),
    openNewChat: (path) => {
      requestChatFocus();
      navigate(path);
    },
    cancelUploads,
    discardFiles,
    releaseFiles: (fileIds) => markPasteSubmitted(...fileIds),
    clearFilesToDelete: () => setFilesToDelete({}),
  };
}
