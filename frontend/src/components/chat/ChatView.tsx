import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { v4 as uuid } from 'uuid'
import { Sparkles, MessageSquarePlus } from 'lucide-react'
import toast from 'react-hot-toast'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import { useChatStore } from '../../stores/chatStore'
import { chatService } from '../../services'
import { subscribeSuggestions, getGreeting } from '../../services/api'
import type { Message, QuotedMessage, HitlInfo } from '../../types/chat'
import type { GreetingResponse } from '../../types/api'

/** 聊天主视图 */
export default function ChatView() {
  const {
    currentSession, threadId, isLoading,
    createSession, addMessage, updateMessage, setLoading, setThreadId,
  } = useChatStore()
  const [quotedMessage, setQuotedMessage] = useState<QuotedMessage | null>(null)
  const [greeting, setGreeting] = useState<GreetingResponse | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  /* ── 获取欢迎语 ── */
  useEffect(() => {
    getGreeting().then(setGreeting).catch(() => {})
  }, [])

  /* ── 自动滚动 ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentSession?.messages])

  /* ── 发送消息 ── */
  const handleSend = useCallback(
    async (text: string, images?: string[], audio?: string, quoted?: QuotedMessage) => {
      if (!currentSession) createSession()
      const tid = threadId || useChatStore.getState().threadId

      const userMsg: Message = {
        id: uuid(),
        role: 'user',
        content: text || (audio ? '🎤 语音消息' : ''),
        timestamp: new Date(),
        images,
        audio,
        quotedMessage: quoted,
      }
      addMessage(userMsg)
      setLoading(true)

      const assistantId = uuid()
      addMessage({
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      })

      const isStream = useChatStore.getState().streamEnabled

      try {
        if (isStream) {
          /* ── 流式输出路径 ── */
          let accumulated = ''
          await chatService.sendMessageStream(
            { content: text, images, audio, quotedMessage: quoted },
            tid || undefined,
            (token) => {
              accumulated += token
              updateMessage(assistantId, { content: accumulated })
            },
            (resp) => {
              if (resp.route && !tid) {
                const newTid = useChatStore.getState().threadId
                if (newTid) setThreadId(newTid)
              }
              updateMessage(assistantId, {
                content: resp.answer || accumulated,
                sources: resp.sources,
                route: resp.route,
              })
            },
            (errMsg) => {
              updateMessage(assistantId, { content: accumulated || '', error: errMsg })
              toast.error(errMsg)
            },
            /* ── 流式 HITL 确认回调 ── */
            (hitlInfo, hitlMessage) => {
              updateMessage(assistantId, {
                content: hitlMessage,
                route: 'hitl_confirm',
                hitl: hitlInfo,
              })
            },
          )
        } else {
          /* ── 非流式输出路径 ── */
          const resp = await chatService.sendMessage(
            { content: text, images, audio, quotedMessage: quoted },
            tid || undefined,
          )

          if (resp.route && !tid) {
            const newTid = useChatStore.getState().threadId
            if (newTid) setThreadId(newTid)
          }

          /* ── 非流式 HITL 确认响应 ── */
          if (resp.route === 'hitl_confirm' && resp.hitl) {
            updateMessage(assistantId, {
              content: resp.answer,
              route: resp.route,
              hitl: resp.hitl,
            })
          } else {
            updateMessage(assistantId, {
              content: resp.answer,
              sources: resp.sources,
              route: resp.route,
              suggestions: resp.suggestions,
            })
          }
        }

        /* ── 建议问题（两个路径共享）── */
        const currentTid = useChatStore.getState().threadId
        if (currentTid) {
          setTimeout(() => {
            subscribeSuggestions(
              currentTid,
              (evt) => {
                if (evt.suggestions?.length) {
                  updateMessage(assistantId, { suggestions: evt.suggestions })
                }
              },
            )
          }, 500)
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '发送失败'
        updateMessage(assistantId, { content: '', error: msg })
        toast.error(msg)
      } finally {
        setLoading(false)
      }
    },
    [currentSession, threadId, createSession, addMessage, updateMessage, setLoading, setThreadId],
  )

  /* ── HITL 高危操作确认 ── */
  const handleHitlConfirm = useCallback(
    async (msgId: string, hitlInfo: HitlInfo, decision: 'approved' | 'rejected') => {
      // 禁用确认按钮：清除 hitl 标记
      updateMessage(msgId, {
        hitl: { ...hitlInfo, requires_confirmation: false },
      })
      setLoading(true)

      const resultId = uuid()
      addMessage({
        id: resultId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      })

      try {
        const resp = await chatService.confirmHitl(hitlInfo.thread_id, decision)
        updateMessage(resultId, {
          content: resp.answer,
          sources: resp.sources,
          route: resp.route,
        })
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '操作处理失败'
        updateMessage(resultId, { content: '', error: msg })
        toast.error(msg)
      } finally {
        setLoading(false)
      }
    },
    [addMessage, updateMessage, setLoading],
  )

  /* ── 快捷问题 ── */
  const handleQuickQuestion = (q: string) => {
    if (!currentSession) createSession()
    handleSend(q)
  }

  const messages = currentSession?.messages || []
  const isEmpty = messages.length === 0

  return (
    <div className="flex h-full flex-col">
      {/* 消息区域 */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {isEmpty ? (
          /* 欢迎区 */
          <div className="flex h-full flex-col items-center justify-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-3xl gradient-brand shadow-2xl shadow-purple-500/30">
                <Sparkles className="h-10 w-10 text-white" />
              </div>
              <h2 className="mb-2 text-2xl font-bold text-white">
                {greeting?.message || '您好，请问有什么可以帮您？'}
              </h2>
              <p className="mb-8 text-slate-500">选择下方话题快速开始，或直接输入您的问题</p>

              {greeting?.options && (
                <div className="flex flex-wrap justify-center gap-3">
                  {greeting.options.map((opt) => (
                    <motion.button
                      key={opt.key}
                      whileHover={{ scale: 1.03, y: -2 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={() => handleQuickQuestion(opt.desc)}
                      className="glass-card flex flex-col items-start gap-1 px-5 py-4 text-left hover:border-violet-500/30 transition-colors w-52"
                    >
                      <span className="text-sm font-medium text-white">{opt.title}</span>
                      <span className="text-xs text-slate-500 line-clamp-2">{opt.desc}</span>
                    </motion.button>
                  ))}
                </div>
              )}
            </motion.div>
          </div>
        ) : (
          /* 消息列表 */
          <div className="mx-auto max-w-3xl space-y-6">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onQuote={(m) => setQuotedMessage({ id: m.id, content: m.content })}
                onHitlConfirm={
                  msg.hitl?.requires_confirmation
                    ? (decision) => handleHitlConfirm(msg.id, msg.hitl!, decision)
                    : undefined
                }
              />
            ))}

            {/* 打字中 */}
            <AnimatePresence>
              {isLoading && messages[messages.length - 1]?.role === 'assistant' && !messages[messages.length - 1]?.content && !messages[messages.length - 1]?.error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center gap-3"
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white">
                    <Sparkles className="h-4 w-4" />
                  </div>
                  <div className="glass-strong rounded-2xl px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* 输入区 */}
      <div className="mx-auto w-full max-w-3xl">
        <ChatInput
          onSend={handleSend}
          disabled={isLoading}
          quotedMessage={quotedMessage}
          onClearQuote={() => setQuotedMessage(null)}
        />
      </div>
    </div>
  )
}
