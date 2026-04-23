import { useCallback } from 'react';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import {
  respondToElicitation as apiRespondToElicitation,
  ElicitationResponse,
  ElicitationState,
} from 'librechat-data-provider';
import { activeElicitationsState, elicitationDataState } from '~/store';

export function useElicitation(toolCallId?: string) {
  const activeElicitations = useRecoilValue(activeElicitationsState);
  const elicitationData = useRecoilValue(elicitationDataState);
  const setActiveElicitations = useSetRecoilState(activeElicitationsState);
  const setElicitationData = useSetRecoilState(elicitationDataState);

  const getElicitationForToolCall = useCallback(
    (tcId?: string) => {
      if (tcId) {
        const activeElicitation = activeElicitations[tcId];
        if (activeElicitation) {
          return elicitationData[activeElicitation.id] || null;
        }
      }
      return null;
    },
    [activeElicitations, elicitationData],
  );

  const respondToElicitation = useCallback(
    async (elicitationId: string, response: ElicitationResponse) => {
      try {
        await apiRespondToElicitation(elicitationId, response);

        setElicitationData((prev) => {
          const newState = { ...prev };
          delete newState[elicitationId];
          return newState;
        });

        setActiveElicitations((prev) => {
          const newState = { ...prev };
          Object.keys(newState).forEach((key) => {
            if (newState[key].id === elicitationId) {
              delete newState[key];
            }
          });
          return newState;
        });
      } catch (error) {
        console.error('Failed to respond to elicitation:', error);
        throw error;
      }
    },
    [setElicitationData, setActiveElicitations],
  );

  const currentElicitation = getElicitationForToolCall(toolCallId) as ElicitationState | null;

  return {
    activeElicitation: currentElicitation,
    hasActiveElicitation: !!currentElicitation,
    respondToElicitation,
    getElicitationForToolCall,
  };
}
