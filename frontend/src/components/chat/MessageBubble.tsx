import { useState } from 'react'
import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  Copy, Check, Quote, ChevronDown, ChevronUp,
  Bot, User as UserIcon, AlertCircle,
} from 'lucide-react'
import type { Message } from '../../types/chat'
import { copyToClipboard, formatMessageTime } from '../../utils/helpers'
import toast from 'react-hot-toast'

interface Props {
  message: Message
  onQuote?: (msg: Message) => void
}

/** 消息气泡组件 */
export default function MessageBubble({ message, onQuote }: Props) {
  const [copied, setCopied] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  const handleCopy = async () => {
    const ok = await copyToClipboard(message.content)
    if (ok) {
      setCopied(true)
      toast.success('已复制')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`group flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* 头像 */}
      <div
        className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-white ${
          isUser
            ? 'bg-gradient-to-br from-violet-500 to-indigo-600'
            : 'bg-gradient-to-br from-emerald-500 to-teal-600'
        }`}
      >
        {isUser ? <UserIcon className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* 内容 */}
      <div className={`flex max-w-[75%] flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        {/* 引用 */}
        {message.quotedMessage && (
          <div className="mb-1 flex items-start gap-1.5 rounded-lg bg-white/5 px-3 py-1.5 text-xs text-slate-400 border-l-2 border-violet-500/50">
            <Quote className="mt-0.5 h-3 w-3 shrink-0 text-violet-400" />
            <span className="line-clamp-2">{message.quotedMessage.content}</span>
          </div>
        )}

        {/* 图片 */}
        {message.images && message.images.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {message.images.map((img, i) => (
              <img
                key={i}
                src={img}
                alt=""
                className="max-h-48 max-w-48 rounded-xl border border-white/10 object-cover cursor-pointer hover:opacity-90 transition-opacity"
                onClick={() => window.open(img, '_blank')}
              />
            ))}
          </div>
        )}

        {/* 消息体 */}
        <div
          className={`relative rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-gradient-to-br from-violet-600/90 to-indigo-600/90 text-white'
              : 'glass-strong text-slate-200'
          } ${message.error ? 'border border-red-500/30' : ''}`}
        >
          {message.error ? (
            <div className="flex items-center gap-2 text-red-400">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span className="text-sm">{message.error}</span>
            </div>
          ) : (
            <div className="markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ className, children, ...rest }) {
                    const match = /language-(\w+)/.exec(className || '')
                    const codeStr = String(children).replace(/\n$/, '')
                    if (match) {
                      return (
                        <div className="relative group/code">
                          <div className="flex items-center justify-between bg-slate-800 px-4 py-1.5 text-xs text-slate-400 rounded-t-xl">
                            <span>{match[1]}</span>
                            <button
                              onClick={() => copyToClipboard(codeStr).then(() => toast.success('代码已复制'))}
                              className="hover:text-white transition-colors"
                            >
                              <Copy className="h-3.5 w-3.5" />
                            </button>
                          </div>
                          <SyntaxHighlighter
                            style={oneDark}
                            language={match[1]}
                            PreTag="div"
                            customStyle={{
                              margin: 0,
                              borderTopLeftRadius: 0,
                              borderTopRightRadius: 0,
                              borderBottomLeftRadius: 12,
                              borderBottomRightRadius: 12,
                              fontSize: '0.85rem',
                            }}
                          >
                            {codeStr}
                          </SyntaxHighlighter>
                        </div>
                      )
                    }
                    return <code className={className} {...rest}>{children}</code>
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* 来源 */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-1.5 w-full">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              <span>{message.sources.length} 个参考来源</span>
            </button>
            {showSources && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="mt-1 space-y-1"
              >
                {message.sources.map((s, i) => (
                  <div key={i} className="rounded-lg bg-white/5 px-3 py-2 text-xs text-slate-400">
                    <span className="font-medium text-slate-300">{s.title}</span>
                    <p className="mt-0.5 line-clamp-2">{s.content}</p>
                  </div>
                ))}
              </motion.div>
            )}
          </div>
        )}

        {/* 操作栏 + 时间 */}
        <div className="mt-1 flex items-center gap-2 text-xs text-slate-600">
          <span>{formatMessageTime(message.timestamp)}</span>
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleCopy}
              className="rounded-md p-1 hover:bg-white/10 text-slate-500 hover:text-slate-300 transition-colors"
              title="复制"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
            {onQuote && (
              <button
                onClick={() => onQuote(message)}
                className="rounded-md p-1 hover:bg-white/10 text-slate-500 hover:text-slate-300 transition-colors"
                title="引用"
              >
                <Quote className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          {message.route && (
            <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-violet-400">
              {message.route}
            </span>
          )}
        </div>

        {/* 建议 */}
        {message.suggestions && message.suggestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.suggestions.map((s, i) => (
              <button
                key={i}
                className="rounded-full border border-violet-500/30 bg-violet-500/10 px-3 py-1 text-xs text-violet-300 hover:bg-violet-500/20 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}
