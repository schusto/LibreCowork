export interface ElicitationPropertySchema {
  type: 'string' | 'number' | 'integer' | 'boolean';
  title?: string;
  description?: string;
  minLength?: number;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  format?: 'email' | 'uri' | 'date' | 'date-time';
  default?: string | number | boolean;
  enum?: string[];
  enumNames?: string[];
}

export interface ElicitationRequestSchema {
  type: 'object';
  properties: Record<string, ElicitationPropertySchema>;
  required?: string[];
}

export interface ElicitationRequest {
  message: string;
  requestedSchema: ElicitationRequestSchema;
}

export type ElicitationAction = 'accept' | 'decline' | 'cancel';

export interface ElicitationResponse {
  action: ElicitationAction;
  content?: Record<string, unknown>;
}

export interface ElicitationState {
  id: string;
  serverName: string;
  userId: string;
  request: ElicitationRequest;
  tool_call_id?: string;
  timestamp: number;
}
