/** 引用消息 */
export interface QuotedMessage {
  id: string
  content: string
}

/** 消息来源 */
export interface Source {
  title: string
  content: string
  url?: string
  metadata?: Record<string, unknown>
}

/** 单条消息 */
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  images?: string[]
  audio?: string
  sources?: Source[]
  route?: string
  suggestions?: string[]
  error?: string
  quotedMessage?: QuotedMessage
  /** HITL 高危操作确认信息 */
  hitl?: HitlInfo
}

/** 会话元数据（列表用） */
export interface SessionMeta {
  thread_id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

/** 完整会话 */
export interface ChatSession {
  id: string
  threadId: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

/** 发送消息参数 */
export interface SendMessageParams {
  content: string
  images?: string[]
  audio?: string
  asrLanguage?: string
  asrItn?: boolean
  quotedMessage?: QuotedMessage
}

/** HITL 高危操作确认信息 */
export interface HitlInfo {
  thread_id: string
  operations: string[]
  order_id: string
  requires_confirmation: boolean
}

/** 聊天响应 */
export interface ChatResponse {
  route: string
  answer: string
  sources?: Source[]
  suggestions?: string[]
  error?: string
  /** 高危操作待确认信息（route="hitl_confirm" 时存在） */
  hitl?: HitlInfo
}
