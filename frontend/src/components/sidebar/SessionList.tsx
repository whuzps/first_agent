import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageSquare, Trash2, Pencil, Check, X, MoreHorizontal } from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { getSessions } from '../../services/api'
import type { SessionMeta } from '../../types/chat'
import { formatSessionTime, truncate } from '../../utils/helpers'

/** 会话列表组件 */
export default function SessionList() {
  const [metas, setMetas] = useState<SessionMeta[]>([])
  const [editing, setEditing] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const [menuOpen, setMenuOpen] = useState<string | null>(null)
  const { currentSession, switchSession, deleteSession, renameSession } = useChatStore()

  /* ── 加载会话列表 ── */
  useEffect(() => {
    const load = async () => {
      try {
        const resp = await getSessions()
        if (resp.code === 0) setMetas(resp.data.sessions)
      } catch { /* ignore */ }
    }
    load()
    const timer = setInterval(load, 30_000)
    return () => clearInterval(timer)
  }, [])

  const handleRename = async (threadId: string) => {
    if (editTitle.trim()) {
      await renameSession(threadId, editTitle.trim())
      setMetas((prev) =>
        prev.map((m) => (m.thread_id === threadId ? { ...m, title: editTitle.trim() } : m)),
      )
    }
    setEditing(null)
  }

  const handleDelete = async (threadId: string) => {
    await deleteSession(threadId)
    setMetas((prev) => prev.filter((m) => m.thread_id !== threadId))
    setMenuOpen(null)
  }

  return (
    <div className="space-y-1">
      <AnimatePresence>
        {metas.map((meta) => {
          const active = currentSession?.threadId === meta.thread_id
          return (
            <motion.div
              key={meta.thread_id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className={`group relative flex items-center gap-2 rounded-xl px-3 py-2.5 cursor-pointer transition-all ${
                active
                  ? 'bg-violet-500/15 border border-violet-500/20 text-white'
                  : 'hover:bg-white/5 text-slate-400 hover:text-slate-200'
              }`}
              onClick={() => {
                if (editing !== meta.thread_id) switchSession(meta)
              }}
            >
              <MessageSquare className={`h-4 w-4 shrink-0 ${active ? 'text-violet-400' : ''}`} />

              {editing === meta.thread_id ? (
                <div className="flex flex-1 items-center gap-1">
                  <input
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRename(meta.thread_id)
                      if (e.key === 'Escape') setEditing(null)
                    }}
                    className="flex-1 bg-white/10 rounded-md px-2 py-0.5 text-sm text-white focus:outline-none"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                  <button onClick={(e) => { e.stopPropagation(); handleRename(meta.thread_id) }} className="p-0.5 text-emerald-400 hover:text-emerald-300">
                    <Check className="h-3.5 w-3.5" />
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setEditing(null) }} className="p-0.5 text-slate-400 hover:text-slate-300">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm">{truncate(meta.title, 20)}</p>
                    <p className="text-[10px] text-slate-600">{formatSessionTime(meta.updated_at)}</p>
                  </div>

                  <div className="relative shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setMenuOpen(menuOpen === meta.thread_id ? null : meta.thread_id)
                      }}
                      className="rounded-md p-1 hover:bg-white/10 text-slate-500 hover:text-white transition-colors"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </button>

                    <AnimatePresence>
                      {menuOpen === meta.thread_id && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.95 }}
                          className="absolute right-0 top-8 z-50 w-28 rounded-xl glass-card py-1 shadow-xl"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={() => {
                              setEditing(meta.thread_id)
                              setEditTitle(meta.title)
                              setMenuOpen(null)
                            }}
                            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-slate-300 hover:bg-white/10 transition-colors"
                          >
                            <Pencil className="h-3 w-3" /> 重命名
                          </button>
                          <button
                            onClick={() => handleDelete(meta.thread_id)}
                            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                          >
                            <Trash2 className="h-3 w-3" /> 删除
                          </button>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </>
              )}
            </motion.div>
          )
        })}
      </AnimatePresence>

      {metas.length === 0 && (
        <p className="py-8 text-center text-xs text-slate-600">暂无历史会话</p>
      )}
    </div>
  )
}
