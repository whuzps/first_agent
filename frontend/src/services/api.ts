import axios from 'axios'
import type {
  ApiResponse,
  GreetingResponse,
  HealthStatus,
  KbTask,
  MilvusCollection,
  ModelInfo,
  OrderInfo,
  SuggestionEvent,
} from '../types/api'
import type { AuthPayload, LoginResponse, User } from '../types/auth'
import type { ChatResponse, HitlInfo, SessionMeta, Source } from '../types/chat'

const http = axios.create({ baseURL: '', timeout: 120_000 })

/* ── 请求拦截：注入认证头 ── */
http.interceptors.request.use((cfg) => {
  const token = localStorage.getItem('auth_token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  const tenant = localStorage.getItem('tenantId')
  if (tenant) cfg.headers['X-Tenant-ID'] = tenant
  const apiKey = localStorage.getItem('apiKey')
  if (apiKey) cfg.headers['X-API-Key'] = apiKey
  const userId = localStorage.getItem('userId')
  if (userId) cfg.headers['X-User-ID'] = userId
  return cfg
})

/* ── 响应拦截：401 跳转登录 ── */
http.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('auth_user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  },
)

/* ============ 认证 ============ */
/** 注册 */
export async function signup(payload: AuthPayload) {
  const { data } = await http.post<ApiResponse<LoginResponse>>('/auth/signup', payload)
  return data
}

/** 登录 */
export async function login(payload: AuthPayload) {
  const { data } = await http.post<ApiResponse<LoginResponse>>('/auth/login', payload)
  return data
}

/** 获取当前用户 */
export async function getMe() {
  const { data } = await http.get<ApiResponse<User>>('/auth/me')
  return data
}

/* ============ 聊天 ============ */
/** 发送聊天消息 */
export async function sendMessage(body: {
  query?: string
  user_id?: string
  thread_id?: string
  images?: string[]
  audio?: string
  asr_language?: string
  asr_itn?: boolean
  quoted_message?: string
}) {
  const { data } = await http.post<ChatResponse>('/chat', body)
  return data
}

/** HITL 高危操作确认 */
export async function confirmHitl(body: { thread_id: string; decision: 'approved' | 'rejected' }) {
  const { data } = await http.post<ChatResponse>('/chat/confirm', body)
  return data
}

/** SSE 流式完成事件 */
export interface StreamDoneEvent {
  type: 'done'
  route: string
  answer: string
  sources?: Source[]
  trace_id?: string
}

/** SSE 流式 HITL 确认事件 */
export interface StreamHitlEvent {
  type: 'hitl_confirm'
  message: string
  operations: string[]
  order_id: string
  thread_id: string
  trace_id?: string
}

/** 流式发送聊天消息（POST SSE），逐 token 回调 */
export async function sendMessageStream(
  body: Parameters<typeof sendMessage>[0],
  onToken: (content: string) => void,
  onDone: (data: StreamDoneEvent) => void,
  onError?: (message: string) => void,
  onHitl?: (data: StreamHitlEvent) => void,
): Promise<void> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = localStorage.getItem('auth_token')
  if (token) headers['Authorization'] = `Bearer ${token}`
  const tenant = localStorage.getItem('tenantId')
  if (tenant) headers['X-Tenant-ID'] = tenant
  const apiKey = localStorage.getItem('apiKey')
  if (apiKey) headers['X-API-Key'] = apiKey
  const userId = localStorage.getItem('userId')
  if (userId) headers['X-User-ID'] = userId

  const resp = await fetch('/chat/stream', {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(text || `HTTP ${resp.status}`)
  }
  if (!resp.body) throw new Error('响应体为空')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const evt = JSON.parse(line.slice(6))
        if (evt.type === 'token') onToken(evt.content)
        else if (evt.type === 'done') onDone(evt as StreamDoneEvent)
        else if (evt.type === 'hitl_confirm') onHitl?.(evt as StreamHitlEvent)
        else if (evt.type === 'error') onError?.(evt.message)
      } catch { /* 忽略解析错误 */ }
    }
  }
}

/* ============ 建议流 ============ */
/** 订阅 SSE 建议流 */
export function subscribeSuggestions(
  threadId: string,
  onEvent: (evt: SuggestionEvent) => void,
  onDone?: () => void,
): EventSource {
  const es = new EventSource(`/suggest/${threadId}`)
  const timeout = setTimeout(() => es.close(), 15_000)

  const handle = (e: MessageEvent) => {
    try {
      const parsed: SuggestionEvent = JSON.parse(e.data)
      onEvent(parsed)
      if (parsed.final) {
        clearTimeout(timeout)
        es.close()
        onDone?.()
      }
    } catch { /* 忽略非法JSON */ }
  }

  es.addEventListener('react_start', handle)
  es.addEventListener('suggest', handle)
  es.addEventListener('error', handle)
  es.onerror = () => { clearTimeout(timeout); es.close(); onDone?.() }
  return es
}

/* ============ 会话管理 ============ */
/** 获取会话列表 */
export async function getSessions() {
  const { data } = await http.get<ApiResponse<{ sessions: SessionMeta[] }>>('/sessions')
  return data
}

/** 获取会话消息 */
export async function getSessionMessages(threadId: string) {
  const { data } = await http.get<ApiResponse<{ thread_id: string; title: string; messages: unknown[] }>>(
    `/sessions/${threadId}/messages`,
  )
  return data
}

/** 删除会话 */
export async function deleteSession(threadId: string) {
  const { data } = await http.delete<ApiResponse<{ deleted: string }>>(`/sessions/${threadId}`)
  return data
}

/** 重命名会话 */
export async function renameSession(threadId: string, title: string) {
  const { data } = await http.patch<ApiResponse<{ thread_id: string; title: string }>>(
    `/sessions/${threadId}`,
    { title },
  )
  return data
}

/* ============ 欢迎语 ============ */
export async function getGreeting() {
  const { data } = await http.get<GreetingResponse>('/greet')
  return data
}

/* ============ 健康检查 ============ */
export async function getHealth() {
  const { data } = await http.get<HealthStatus>('/health')
  return data
}

/* ============ 模型 ============ */
export async function getModels() {
  const { data } = await http.get<ApiResponse<ModelInfo>>('/models/list')
  return data
}

export async function switchModel(name: string) {
  const { data } = await http.post<ApiResponse<ModelInfo>>('/models/switch', { name })
  return data
}

/* ============ 订单 ============ */
export async function getOrder(orderId: string) {
  const { data } = await http.get<OrderInfo>(`/api/orders/${orderId}`)
  return data
}

/* ============ 知识库 ============ */
/** 批量上传知识库文档，支持上传进度回调 */
export async function uploadKbFiles(files: File[], onProgress?: (pct: number) => void) {
  const fd = new FormData()
  files.forEach((f) => fd.append('files', f))
  const { data } = await http.post<ApiResponse<{ tasks: KbTask[]; collection_name: string }>>(
    '/api/v1/kb/upload',
    fd,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (e.total && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
      },
    },
  )
  return data
}

/** 获取上传任务列表 */
export async function getKbTasks() {
  const { data } = await http.get<ApiResponse<{ tasks: KbTask[] }>>('/api/v1/kb/tasks')
  return data
}

/** 删除任务 */
export async function deleteKbTask(taskId: string) {
  const { data } = await http.delete<ApiResponse<{ deleted: string }>>(`/api/v1/kb/tasks/${taskId}`)
  return data
}

/** 获取Milvus集合列表（含统计信息） */
export async function listCollections() {
  const { data } = await http.get<ApiResponse<{ collections: MilvusCollection[] }>>('/api/v1/milvus/collections')
  return data
}

/** 清空并重建指定 Milvus 集合 */
export async function deleteCollection(name: string) {
  const { data } = await http.post<ApiResponse<{ message: string; collection_name: string }>>(
    '/api/v1/milvus/collections/delete',
    { collection_name: name },
  )
  return data
}

/** 批量添加向量文本 */
export async function addVectors(items: Array<{ text: string; metadata?: Record<string, unknown>; id?: string }>) {
  const { data } = await http.post<ApiResponse<{ added: number; ids: string[]; skipped: string[] }>>(
    '/api/v1/vectors/items',
    { items },
  )
  return data
}

/** 按 ID 删除向量 */
export async function deleteVectors(ids: string[]) {
  const { data } = await http.delete<ApiResponse<{ deleted: number; ids: string[] }>>(
    '/api/v1/vectors/items',
    { data: { ids } },
  )
  return data
}
