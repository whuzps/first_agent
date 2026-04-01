import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send, Mic, MicOff, Image as ImageIcon, X, Quote,
  Paperclip, StopCircle, Zap,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useAudioRecorder } from '../../hooks/useAudioRecorder'
import { imageToBase64 } from '../../utils/helpers'
import { useChatStore } from '../../stores/chatStore'
import type { QuotedMessage } from '../../types/chat'

interface Props {
  onSend: (text: string, images?: string[], audio?: string, quoted?: QuotedMessage) => void
  disabled?: boolean
  quotedMessage?: QuotedMessage | null
  onClearQuote?: () => void
}

/** 聊天输入框组件 */
export default function ChatInput({ onSend, disabled, quotedMessage, onClearQuote }: Props) {
  const { streamEnabled, toggleStream } = useChatStore()
  const [text, setText] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [showImageUpload, setShowImageUpload] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isRecording, duration, start, stop, cancel } = useAudioRecorder()

  /* ── 发送 ── */
  const handleSend = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && images.length === 0) return
    onSend(trimmed, images.length > 0 ? images : undefined, undefined, quotedMessage || undefined)
    setText('')
    setImages([])
    onClearQuote?.()
  }, [text, images, quotedMessage, onSend, onClearQuote])

  /* ── 按键 ── */
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  /* ── 图片选择 ── */
  const handleFiles = async (files: FileList | null) => {
    if (!files) return
    const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    const arr = Array.from(files).filter((f) => allowed.includes(f.type))
    if (arr.length === 0) {
      toast.error('仅支持 JPG/PNG/GIF/WEBP')
      return
    }
    try {
      const b64 = await Promise.all(arr.map(imageToBase64))
      setImages((prev) => [...prev, ...b64].slice(0, 5))
      setShowImageUpload(false)
    } catch {
      toast.error('图片读取失败')
    }
  }

  /* ── 拖拽 ── */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  /* ── 语音 ── */
  const handleVoiceToggle = async () => {
    if (isRecording) {
      const audio = await stop()
      if (audio) {
        onSend('', undefined, audio, quotedMessage || undefined)
        onClearQuote?.()
      }
    } else {
      try {
        await start()
      } catch {
        toast.error('无法访问麦克风，请检查权限')
      }
    }
  }

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="border-t border-white/5 bg-slate-950/80 backdrop-blur-xl px-4 py-3">
      {/* 引用预览 */}
      <AnimatePresence>
        {quotedMessage && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mb-2 flex items-start gap-2 rounded-xl bg-violet-500/10 border border-violet-500/20 px-3 py-2"
          >
            <Quote className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
            <p className="flex-1 text-sm text-slate-300 line-clamp-2">{quotedMessage.content}</p>
            <button onClick={onClearQuote} className="shrink-0 text-slate-500 hover:text-white transition-colors">
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 图片预览 */}
      <AnimatePresence>
        {images.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mb-2 flex flex-wrap gap-2"
          >
            {images.map((img, i) => (
              <div key={i} className="group/img relative">
                <img src={img} alt="" className="h-16 w-16 rounded-lg object-cover border border-white/10" />
                <button
                  onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
                  className="absolute -right-1.5 -top-1.5 rounded-full bg-red-500 p-0.5 opacity-0 group-hover/img:opacity-100 transition-opacity"
                >
                  <X className="h-3 w-3 text-white" />
                </button>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* 录音状态 */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mb-2 flex items-center justify-between rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-2.5"
          >
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 animate-pulse rounded-full bg-red-500" />
              <span className="text-sm text-red-300">录音中 {formatDuration(duration)}</span>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={cancel} className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white transition-colors">
                <X className="h-4 w-4" />
              </button>
              <button onClick={handleVoiceToggle} className="rounded-lg bg-red-500 p-1.5 text-white hover:bg-red-600 transition-colors">
                <StopCircle className="h-4 w-4" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 输入区域 */}
      <div
        className="flex items-end gap-2 rounded-2xl bg-white/5 border border-white/10 px-3 py-2 focus-within:border-violet-500/40 focus-within:ring-1 focus-within:ring-violet-500/20 transition-all"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {/* 附件按钮 */}
        <div className="flex items-center gap-1 pb-0.5">
          <button
            onClick={() => fileRef.current?.click()}
            className="rounded-lg p-2 text-slate-500 hover:text-violet-400 hover:bg-white/5 transition-colors"
            title="上传图片"
          >
            <ImageIcon className="h-5 w-5" />
          </button>
          <button
            onClick={handleVoiceToggle}
            disabled={isRecording}
            className={`rounded-lg p-2 transition-colors ${
              isRecording
                ? 'text-red-400 bg-red-500/10'
                : 'text-slate-500 hover:text-violet-400 hover:bg-white/5'
            }`}
            title="语音输入"
          >
            {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </button>
          <button
            onClick={toggleStream}
            className={`rounded-lg p-2 transition-colors ${
              streamEnabled
                ? 'text-amber-400 bg-amber-500/15'
                : 'text-slate-500 hover:text-violet-400 hover:bg-white/5'
            }`}
            title={streamEnabled ? '流式输出已开启' : '点击开启流式输出'}
          >
            <Zap className={`h-5 w-5 ${streamEnabled ? 'fill-amber-400/30' : ''}`} />
          </button>
        </div>

        {/* 文本输入 */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            const el = e.target
            el.style.height = 'auto'
            el.style.height = Math.min(el.scrollHeight, 150) + 'px'
          }}
          onKeyDown={handleKeyDown}
          placeholder="输入消息... (Shift+Enter 换行)"
          className="max-h-[150px] min-h-[24px] flex-1 resize-none bg-transparent py-1 text-sm text-white placeholder-slate-500 focus:outline-none"
          rows={1}
          disabled={disabled || isRecording}
        />

        {/* 发送按钮 */}
        <button
          onClick={handleSend}
          disabled={disabled || isRecording || (!text.trim() && images.length === 0)}
          className={`mb-0.5 shrink-0 rounded-xl p-2 transition-all ${
            text.trim() || images.length > 0
              ? 'gradient-brand text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40'
              : 'bg-white/5 text-slate-600'
          } disabled:opacity-40`}
        >
          <Send className="h-5 w-5" />
        </button>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
