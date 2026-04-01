import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { v4 as uuid } from 'uuid'
import type { ChatSession, Message, SessionMeta } from '../types/chat'
import * as api from '../services/api'

interface ChatState {
  sessions: ChatSession[]
  currentSession: ChatSession | null
  threadId: string | null
  isLoading: boolean
  error: string | null
  /** 是否启用流式输出 */
  streamEnabled: boolean

  /** 创建新会话 */
  createSession: (title?: string) => void
  /** 切换会话（从元数据加载） */
  switchSession: (meta: SessionMeta) => Promise<void>
  /** 添加消息 */
  addMessage: (msg: Message) => void
  /** 更新消息 */
  updateMessage: (id: string, patch: Partial<Message>) => void
  /** 删除会话 */
  deleteSession: (threadId: string) => Promise<void>
  /** 重命名会话 */
  renameSession: (threadId: string, title: string) => Promise<void>
  /** 设置加载状态 */
  setLoading: (v: boolean) => void
  /** 设置错误 */
  setError: (e: string | null) => void
  /** 设置 threadId */
  setThreadId: (id: string) => void
  /** 切换流式输出 */
  toggleStream: () => void
}

/** 聊天状态管理（持久化到 localStorage） */
export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSession: null,
      threadId: null,
      isLoading: false,
      error: null,
      streamEnabled: false,

      createSession: (title) => {
        const tid = uuid()
        const session: ChatSession = {
          id: uuid(),
          threadId: tid,
          title: title || '新对话',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        set((s) => ({
          sessions: [session, ...s.sessions],
          currentSession: session,
          threadId: tid,
          error: null,
        }))
      },

      switchSession: async (meta) => {
        const existing = get().sessions.find((s) => s.threadId === meta.thread_id)
        if (existing) {
          set({ currentSession: existing, threadId: meta.thread_id })
          return
        }
        try {
          const resp = await api.getSessionMessages(meta.thread_id)
          if (resp.code !== 0) return
          const messages: Message[] = (resp.data.messages as Record<string, unknown>[]).map((m) => ({
            id: uuid(),
            role: (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
            content: (m.content as string) || '',
            timestamp: new Date(),
          }))
          const session: ChatSession = {
            id: uuid(),
            threadId: meta.thread_id,
            title: resp.data.title || meta.title,
            messages,
            createdAt: new Date(meta.created_at * 1000),
            updatedAt: new Date(meta.updated_at * 1000),
          }
          set((s) => ({
            sessions: [session, ...s.sessions.filter((x) => x.threadId !== meta.thread_id)],
            currentSession: session,
            threadId: meta.thread_id,
          }))
        } catch { /* ignore */ }
      },

      addMessage: (msg) => {
        set((s) => {
          if (!s.currentSession) return s
          const updated = {
            ...s.currentSession,
            messages: [...s.currentSession.messages, msg],
            updatedAt: new Date(),
          }
          return {
            currentSession: updated,
            sessions: s.sessions.map((x) => (x.threadId === updated.threadId ? updated : x)),
          }
        })
      },

      updateMessage: (id, patch) => {
        set((s) => {
          if (!s.currentSession) return s
          const updated = {
            ...s.currentSession,
            messages: s.currentSession.messages.map((m) =>
              m.id === id ? { ...m, ...patch } : m,
            ),
          }
          return {
            currentSession: updated,
            sessions: s.sessions.map((x) => (x.threadId === updated.threadId ? updated : x)),
          }
        })
      },

      deleteSession: async (threadId) => {
        try {
          await api.deleteSession(threadId)
        } catch { /* ignore */ }
        set((s) => {
          const sessions = s.sessions.filter((x) => x.threadId !== threadId)
          return {
            sessions,
            currentSession: s.currentSession?.threadId === threadId ? null : s.currentSession,
            threadId: s.threadId === threadId ? null : s.threadId,
          }
        })
      },

      renameSession: async (threadId, title) => {
        try {
          await api.renameSession(threadId, title)
        } catch { /* ignore */ }
        set((s) => ({
          sessions: s.sessions.map((x) => (x.threadId === threadId ? { ...x, title } : x)),
          currentSession:
            s.currentSession?.threadId === threadId
              ? { ...s.currentSession, title }
              : s.currentSession,
        }))
      },

      setLoading: (v) => set({ isLoading: v }),
      setError: (e) => set({ error: e }),
      setThreadId: (id) => set({ threadId: id }),
      toggleStream: () => set((s) => ({ streamEnabled: !s.streamEnabled })),
    }),
    {
      name: 'agent-chat-v2',
      partialize: (s) => ({
        sessions: s.sessions.slice(0, 50),
        currentSession: s.currentSession,
        threadId: s.threadId,
        streamEnabled: s.streamEnabled,
      }),
    },
  ),
)
