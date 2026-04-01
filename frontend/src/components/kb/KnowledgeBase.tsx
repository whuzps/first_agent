import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileText, Trash2, RefreshCw, Database,
  CheckCircle2, AlertCircle, Clock, Loader2,
  HardDrive, X, FileUp, Filter, BarChart3, Info,
  Search, Send, BookOpen, Settings, Plus, FolderOpen,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { v4 as uuidv4 } from 'uuid'
import {
  uploadKbFiles, getKbTasks, deleteKbTask,
  listCollections, deleteCollection,
  sendMessage, addVectors, deleteVectors,
} from '../../services/api'
import type { KbTask, MilvusCollection } from '../../types/api'
import type { ChatResponse, Source } from '../../types/chat'
import { formatFileSize, relativeTime } from '../../utils/helpers'

/* ── 常量 ── */
type TaskStatus = KbTask['status'] | 'all'

const STATUS_META: Record<string, { icon: typeof CheckCircle2; color: string; label: string; bg: string }> = {
  done:       { icon: CheckCircle2, color: 'text-emerald-400', label: '完成',   bg: 'bg-emerald-500/10' },
  processing: { icon: Loader2,     color: 'text-amber-400',   label: '处理中', bg: 'bg-amber-500/10' },
  pending:    { icon: Clock,       color: 'text-slate-400',   label: '等待中', bg: 'bg-slate-500/10' },
  failed:     { icon: AlertCircle, color: 'text-red-400',     label: '失败',   bg: 'bg-red-500/10' },
}

const FILTER_TABS: { key: TaskStatus; label: string }[] = [
  { key: 'all',        label: '全部' },
  { key: 'pending',    label: '等待中' },
  { key: 'processing', label: '处理中' },
  { key: 'done',       label: '已完成' },
  { key: 'failed',     label: '失败' },
]

const ALLOWED_EXT = ['.txt', '.md', '.pdf', '.docx']

const TAB_ITEMS = [
  { key: 'upload',  label: '文档上传', icon: FileUp },
  { key: 'search',  label: '知识库查询', icon: Search },
  { key: 'milvus',  label: '向量数据库', icon: Database },
  { key: 'manage',  label: '数据管理', icon: Settings },
] as const

type TabKey = typeof TAB_ITEMS[number]['key']

/* ================================================================
   文档上传 Tab
   ================================================================ */
function UploadTab() {
  const [tasks, setTasks] = useState<KbTask[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [stagedFiles, setStagedFiles] = useState<File[]>([])
  const [filter, setFilter] = useState<TaskStatus>('all')
  const [expandedError, setExpandedError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  /* 拉取任务列表 */
  const fetchTasks = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const resp = await getKbTasks()
      if (resp.code === 0) setTasks(resp.data.tasks)
    } catch { /* ignore */ }
    if (!quiet) setLoading(false)
  }, [])

  useEffect(() => { fetchTasks() }, [fetchTasks])

  /* 轮询：有活跃任务时每 3s 刷新 */
  useEffect(() => {
    const hasActive = tasks.some(t => t.status === 'pending' || t.status === 'processing')
    if (hasActive && !pollingRef.current) {
      pollingRef.current = setInterval(() => fetchTasks(true), 3000)
    } else if (!hasActive && pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
    return () => {
      if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null }
    }
  }, [tasks, fetchTasks])

  /* 统计 */
  const stats = useMemo(() => {
    const total = tasks.length
    const done = tasks.filter(t => t.status === 'done').length
    const processing = tasks.filter(t => t.status === 'processing').length
    const pending = tasks.filter(t => t.status === 'pending').length
    const failed = tasks.filter(t => t.status === 'failed').length
    const doneRate = total ? Math.round((done / total) * 100) : 0
    return { total, done, processing, pending, failed, doneRate }
  }, [tasks])

  const filteredTasks = useMemo(
    () => filter === 'all' ? tasks : tasks.filter(t => t.status === filter),
    [tasks, filter],
  )

  /* 暂存文件 */
  const stageFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    const valid = Array.from(fileList).filter(f =>
      ALLOWED_EXT.some(ext => f.name.toLowerCase().endsWith(ext)),
    )
    if (valid.length === 0) { toast.error('仅支持 .txt .md .pdf .docx 格式'); return }
    const rejected = fileList.length - valid.length
    if (rejected > 0) toast(`已忽略 ${rejected} 个不支持的文件`, { icon: '⚠️' })
    setStagedFiles(prev => {
      const existingNames = new Set(prev.map(f => f.name))
      return [...prev, ...valid.filter(f => !existingNames.has(f.name))]
    })
    if (fileRef.current) fileRef.current.value = ''
  }

  /* 提交上传 */
  const handleUpload = async () => {
    if (stagedFiles.length === 0) return
    setUploading(true)
    setUploadProgress(0)
    try {
      const resp = await uploadKbFiles(stagedFiles, pct => setUploadProgress(pct))
      if (resp.code === 0) {
        toast.success(`已提交 ${resp.data.tasks.length} 个文件，后台解析中…`)
        setStagedFiles([])
        await fetchTasks()
      } else {
        toast.error(resp.message || '上传失败')
      }
    } catch { toast.error('上传失败，请检查网络连接') }
    setTimeout(() => { setUploading(false); setUploadProgress(0) }, 600)
  }

  /* 删除任务 */
  const handleDeleteTask = async (taskId: string) => {
    try {
      await deleteKbTask(taskId)
      setTasks(prev => prev.filter(t => t.task_id !== taskId))
      toast.success('已删除任务记录')
    } catch { toast.error('删除失败') }
  }

  const activePolling = tasks.some(t => t.status === 'pending' || t.status === 'processing')

  return (
    <div className="space-y-6">
      {/* 上传区 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2">
            <Upload className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-white">上传文档</h2>
          </div>
          <span className="text-xs text-slate-500">支持 .txt .md .pdf .docx</span>
        </div>
        <div className="p-5">
          <motion.div
            className={`relative rounded-2xl border-2 border-dashed transition-colors ${
              dragOver
                ? 'border-violet-500/60 bg-violet-500/10'
                : 'border-white/10 hover:border-white/20 bg-white/[0.02]'
            }`}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => { e.preventDefault(); setDragOver(false); stageFiles(e.dataTransfer.files) }}
          >
            <div className="flex flex-col items-center justify-center py-10">
              <div className="mb-3 rounded-2xl bg-violet-500/10 p-3">
                <Upload className="h-7 w-7 text-violet-400" />
              </div>
              <p className="text-sm text-white mb-1">点击或拖拽文件到此区域</p>
              <button
                onClick={() => fileRef.current?.click()}
                disabled={uploading}
                className="text-sm text-violet-400 hover:text-violet-300 transition-colors underline underline-offset-2"
              >
                点击选择文件
              </button>
              <p className="mt-2 text-xs text-slate-600">支持批量上传，文件将在后台多进程并发解析入库</p>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf,.docx"
              multiple
              className="hidden"
              onChange={e => stageFiles(e.target.files)}
            />
          </motion.div>
        </div>
      </div>

      {/* 暂存文件列表 */}
      <AnimatePresence>
        {stagedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="glass-card overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
                <div className="flex items-center gap-2">
                  <FileUp className="h-4 w-4 text-violet-400" />
                  <h2 className="text-sm font-medium text-white">
                    待上传文件 ({stagedFiles.length})
                  </h2>
                  <span className="text-xs text-slate-500">
                    共 {formatFileSize(stagedFiles.reduce((s, f) => s + f.size, 0))}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => setStagedFiles([])} className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
                    清空
                  </button>
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-1.5 text-sm text-white
                               hover:bg-violet-500 disabled:opacity-50 transition-colors"
                  >
                    {uploading
                      ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      : <Upload className="h-3.5 w-3.5" />}
                    {uploading ? '上传中…' : `提交 ${stagedFiles.length} 个文件`}
                  </button>
                </div>
              </div>
              {uploading && (
                <div className="h-1 bg-white/5">
                  <motion.div
                    className="h-full bg-violet-500"
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                  />
                </div>
              )}
              <div className="divide-y divide-white/5 max-h-48 overflow-y-auto">
                {stagedFiles.map(f => (
                  <div key={f.name} className="flex items-center gap-3 px-5 py-2.5 hover:bg-white/[0.02] transition-colors">
                    <FileText className="h-4 w-4 shrink-0 text-slate-500" />
                    <span className="flex-1 truncate text-sm text-white">{f.name}</span>
                    <span className="text-xs text-slate-600">{formatFileSize(f.size)}</span>
                    <button
                      onClick={() => setStagedFiles(prev => prev.filter(pf => pf.name !== f.name))}
                      className="rounded p-1 text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 统计摘要 */}
      {stats.total > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {([
            { label: '总任务', value: stats.total, color: 'text-slate-300', bg: 'bg-slate-500/10' },
            { label: '已完成', value: stats.done, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
            { label: '进行中', value: stats.processing, color: 'text-amber-400', bg: 'bg-amber-500/10' },
            { label: '等待中', value: stats.pending, color: 'text-blue-400', bg: 'bg-blue-500/10' },
            { label: '失败', value: stats.failed, color: 'text-red-400', bg: 'bg-red-500/10' },
          ] as const).map(s => (
            <div key={s.label} className={`rounded-xl ${s.bg} px-4 py-3`}>
              <p className="text-xs text-slate-500">{s.label}</p>
              <p className={`text-lg font-semibold ${s.color}`}>{s.value}</p>
            </div>
          ))}
          {stats.total > 0 && (
            <div className="col-span-2 sm:col-span-5">
              <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    stats.failed > 0 ? 'bg-red-500' : stats.doneRate === 100 ? 'bg-emerald-500' : 'bg-violet-500'
                  }`}
                  style={{ width: `${stats.doneRate}%` }}
                />
              </div>
              <p className="text-xs text-slate-600 mt-1 text-right">{stats.doneRate}% 完成</p>
            </div>
          )}
        </div>
      )}

      {/* 任务列表 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-white">解析任务</h2>
            {activePolling && (
              <span className="flex items-center gap-1 text-[11px] text-amber-400">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
                </span>
                自动刷新中
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => fetchTasks()}
              disabled={loading}
              className="mr-2 rounded-lg p-1.5 text-slate-500 hover:text-white hover:bg-white/5 transition-colors"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <Filter className="h-3.5 w-3.5 text-slate-600 mr-1" />
            {FILTER_TABS.map(tab => {
              const count = tab.key === 'all' ? stats.total : stats[tab.key as keyof typeof stats]
              return (
                <button
                  key={tab.key}
                  onClick={() => setFilter(tab.key)}
                  className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                    filter === tab.key
                      ? 'bg-violet-600/20 text-violet-300'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
                  }`}
                >
                  {tab.label}
                  {typeof count === 'number' && count > 0 && (
                    <span className="ml-1 text-[10px] opacity-70">({count})</span>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {filteredTasks.length === 0 ? (
          <div className="flex flex-col items-center py-12 text-slate-600">
            <FileText className="mb-3 h-10 w-10 opacity-40" />
            <p className="text-sm">
              {stats.total === 0 ? '暂无任务记录，上传文件后自动创建' : '当前筛选条件下无任务'}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-white/5 max-h-[420px] overflow-y-auto">
            {filteredTasks.map(task => {
              const st = STATUS_META[task.status] || STATUS_META.pending
              const Icon = st.icon
              const isExpanded = expandedError === task.task_id
              return (
                <motion.div key={task.task_id} layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="group">
                  <div className="flex items-center gap-4 px-5 py-3 hover:bg-white/[0.02] transition-colors">
                    <div className={`rounded-lg p-1.5 ${st.bg}`}>
                      <Icon className={`h-4 w-4 ${st.color} ${task.status === 'processing' ? 'animate-spin' : ''}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="truncate text-sm text-white">{task.filename}</p>
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-slate-600 mt-0.5">
                        <span>{formatFileSize(task.file_size)}</span>
                        <span className="text-slate-700">·</span>
                        <span>{task.collection_name}</span>
                        {task.chunk_count > 0 && (
                          <>
                            <span className="text-slate-700">·</span>
                            <span className="text-emerald-500/70">{task.chunk_count} 分块</span>
                          </>
                        )}
                        <span className="text-slate-700">·</span>
                        <span>{relativeTime(task.created_at)}</span>
                      </div>
                    </div>
                    <span className={`rounded-md px-2 py-0.5 text-xs ${st.color} ${st.bg}`}>{st.label}</span>
                    {task.error && (
                      <button
                        onClick={() => setExpandedError(isExpanded ? null : task.task_id)}
                        className="rounded-lg p-1.5 text-red-400/60 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                        title="查看错误详情"
                      >
                        <Info className="h-4 w-4" />
                      </button>
                    )}
                    <button
                      onClick={() => handleDeleteTask(task.task_id)}
                      className="rounded-lg p-1.5 text-slate-600 opacity-0 group-hover:opacity-100
                                 hover:text-red-400 hover:bg-red-500/10 transition-all"
                      title="删除任务"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                  <AnimatePresence>
                    {isExpanded && task.error && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="mx-5 mb-3 rounded-lg bg-red-500/5 border border-red-500/10 px-4 py-2.5">
                          <p className="text-xs text-red-300/80 whitespace-pre-wrap break-all">{task.error}</p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

/* ================================================================
   知识库查询 Tab
   ================================================================ */
function SearchTab() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)

  /** 发送查询 */
  const handleSearch = async () => {
    const q = query.trim()
    if (!q) { toast.error('请输入查询内容'); return }
    setLoading(true)
    try {
      const res = await sendMessage({ query: q, thread_id: uuidv4() })
      setAnswer(res)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '查询失败'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* 查询输入 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
          <Search className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-medium text-white">知识库查询</h2>
        </div>
        <div className="p-5">
          <div className="flex gap-3">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !loading && handleSearch()}
              placeholder="输入查询内容，回车或点击查询"
              disabled={loading}
              className="input-field flex-1"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-3 text-sm text-white
                         hover:bg-violet-500 disabled:opacity-50 transition-colors shrink-0"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              查询
            </button>
          </div>
        </div>
      </div>

      {/* 查询结果 */}
      <AnimatePresence>
        {answer && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="space-y-4"
          >
            {/* 答案 */}
            <div className="glass-card overflow-hidden">
              <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
                <BookOpen className="h-4 w-4 text-violet-400" />
                <h2 className="text-sm font-medium text-white">答案</h2>
              </div>
              <div className="p-5">
                <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{answer.answer}</p>
              </div>
            </div>

            {/* 参考来源 */}
            {answer.sources && answer.sources.length > 0 && (
              <div className="glass-card overflow-hidden">
                <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
                  <FileText className="h-4 w-4 text-violet-400" />
                  <h2 className="text-sm font-medium text-white">参考来源</h2>
                  <span className="text-xs text-slate-500">({answer.sources.length})</span>
                </div>
                <div className="divide-y divide-white/5">
                  {answer.sources.map((s: Source, i: number) => (
                    <div key={i} className="px-5 py-3 hover:bg-white/[0.02] transition-colors">
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="rounded-md bg-violet-600/20 px-2 py-0.5 text-xs text-violet-300">
                          {s.title || '知识库'}
                        </span>
                        {s.url && (
                          <a href={s.url} target="_blank" rel="noreferrer"
                             className="text-xs text-violet-400 hover:text-violet-300 underline underline-offset-2">
                            链接
                          </a>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">{s.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ================================================================
   向量数据库管理 Tab
   ================================================================ */
function MilvusTab() {
  const [collections, setCollections] = useState<MilvusCollection[]>([])
  const [loading, setLoading] = useState(false)
  const [clearing, setClearing] = useState<string | null>(null)

  /** 获取集合列表 */
  const fetchCollections = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await listCollections()
      if (resp.code === 0) setCollections(resp.data.collections)
    } catch {
      toast.error('获取集合列表失败')
    }
    setLoading(false)
  }, [])

  useEffect(() => { fetchCollections() }, [fetchCollections])

  /** 清空并重建集合 */
  const handleClear = async (name: string) => {
    if (!window.confirm(`确认清空集合 "${name}" 吗？\n\n集合中所有向量数据将被清空并重建 Schema。\n此操作不可撤销！`)) return
    setClearing(name)
    try {
      const resp = await deleteCollection(name)
      if (resp.code === 0) {
        toast.success(`集合 ${name} 已清空并重建`)
        await fetchCollections()
      } else {
        toast.error(resp.message || '操作失败')
      }
    } catch {
      toast.error('清空集合失败')
    }
    setClearing(null)
  }

  return (
    <div className="space-y-6">
      {/* 集合列表 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-white">Milvus 集合管理</h2>
            {collections.length > 0 && (
              <span className="rounded-md bg-violet-600/20 px-2 py-0.5 text-xs text-violet-300">
                {collections.length} 个集合
              </span>
            )}
          </div>
          <button
            onClick={fetchCollections}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-slate-400
                       hover:text-white hover:bg-white/5 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            刷新
          </button>
        </div>

        {loading && collections.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 text-violet-400 animate-spin" />
          </div>
        ) : collections.length === 0 ? (
          <div className="flex flex-col items-center py-12 text-slate-600">
            <HardDrive className="mb-3 h-10 w-10 opacity-40" />
            <p className="text-sm">暂无集合，上传文档后会自动创建</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-left">
                  <th className="px-5 py-3 text-xs font-medium text-slate-500">集合名称</th>
                  <th className="px-5 py-3 text-xs font-medium text-slate-500 w-32">向量条数</th>
                  <th className="px-5 py-3 text-xs font-medium text-slate-500 w-24">状态</th>
                  <th className="px-5 py-3 text-xs font-medium text-slate-500 w-28">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {collections.map(col => (
                  <tr key={col.collection_name} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <FolderOpen className="h-4 w-4 text-violet-400 shrink-0" />
                        <span className="text-white">{col.collection_name}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3">
                      <span className={`rounded-md px-2 py-0.5 text-xs ${
                        col.row_count > 0 ? 'bg-blue-500/10 text-blue-400' : 'bg-slate-500/10 text-slate-500'
                      }`}>
                        {col.row_count.toLocaleString()} 条
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <span className={`flex items-center gap-1.5 text-xs ${col.exists ? 'text-emerald-400' : 'text-red-400'}`}>
                        <span className={`h-1.5 w-1.5 rounded-full ${col.exists ? 'bg-emerald-400' : 'bg-red-400'}`} />
                        {col.exists ? '可用' : '不可用'}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <button
                        onClick={() => handleClear(col.collection_name)}
                        disabled={clearing === col.collection_name}
                        className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs
                                   text-red-400 hover:bg-red-500/10 disabled:opacity-50 transition-colors"
                      >
                        {clearing === col.collection_name
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <Trash2 className="h-3.5 w-3.5" />}
                        清空重建
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 说明 */}
      <div className="glass-card overflow-hidden">
        <div className="p-5 space-y-2">
          <p className="text-xs text-slate-500 font-medium">说明</p>
          <ul className="space-y-1.5 text-xs text-slate-500">
            <li className="flex gap-2"><span className="text-slate-600">·</span>「清空重建」会删除集合中所有向量数据并重建索引结构，需要重新上传文档入库</li>
            <li className="flex gap-2"><span className="text-slate-600">·</span>向量条数为 Milvus 实时统计值，上传文档后刷新即可看到最新数据</li>
            <li className="flex gap-2"><span className="text-slate-600">·</span>集合在首次上传文档时自动创建，无需手动操作</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   数据管理 Tab
   ================================================================ */
function ManageTab() {
  const [importText, setImportText] = useState('')
  const [importMetadata, setImportMetadata] = useState('')
  const [deleteIds, setDeleteIds] = useState('')
  const [importLoading, setImportLoading] = useState(false)
  const [deleteLoading, setDeleteLoading] = useState(false)

  /** 批量导入文本到向量索引 */
  const handleImport = async () => {
    const lines = importText.split('\n').map(s => s.trim()).filter(Boolean)
    if (lines.length === 0) { toast.error('请输入待导入文本，每行一条'); return }
    let metadata: Record<string, unknown> | undefined
    if (importMetadata.trim()) {
      try { metadata = JSON.parse(importMetadata.trim()) }
      catch { toast.error('Metadata 需为合法 JSON'); return }
    }
    setImportLoading(true)
    try {
      const resp = await addVectors(lines.map(text => ({ text, metadata })))
      if (resp.code === 0) {
        toast.success(`导入成功，新增 ${resp.data.added} 条`)
        setImportText('')
        setImportMetadata('')
      } else {
        toast.error(resp.message || '导入失败')
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '导入失败'
      toast.error(msg)
    }
    setImportLoading(false)
  }

  /** 按 ID 删除向量 */
  const handleDelete = async () => {
    const ids = deleteIds.split(',').map(s => s.trim()).filter(Boolean)
    if (ids.length === 0) { toast.error('请输入待删除的 ID，逗号分隔'); return }
    setDeleteLoading(true)
    try {
      const resp = await deleteVectors(ids)
      if (resp.code === 0) {
        toast.success(`删除成功，共删除 ${resp.data.deleted} 条`)
        setDeleteIds('')
      } else {
        toast.error(resp.message || '删除失败')
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '删除失败'
      toast.error(msg)
    }
    setDeleteLoading(false)
  }

  return (
    <div className="space-y-6">
      {/* 批量导入 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
          <Plus className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-medium text-white">批量导入文本</h2>
        </div>
        <div className="p-5 space-y-4">
          <p className="text-xs text-slate-500">每行一条文本，可选 Metadata JSON</p>
          <textarea
            rows={6}
            value={importText}
            onChange={e => setImportText(e.target.value)}
            placeholder="每行一条文本，例如：&#10;这是第一条知识文本&#10;这是第二条知识文本"
            className="input-field resize-none font-mono text-sm"
          />
          <textarea
            rows={3}
            value={importMetadata}
            onChange={e => setImportMetadata(e.target.value)}
            placeholder='可选 Metadata（JSON），例如：{"source": "manual", "category": "FAQ"}'
            className="input-field resize-none font-mono text-sm"
          />
          <button
            onClick={handleImport}
            disabled={importLoading || !importText.trim()}
            className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm text-white
                       hover:bg-violet-500 disabled:opacity-50 transition-colors"
          >
            {importLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            导入
          </button>
        </div>
      </div>

      {/* 按 ID 删除 */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center gap-2 border-b border-white/5 px-5 py-3">
          <Trash2 className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-medium text-white">按 ID 删除</h2>
        </div>
        <div className="p-5 space-y-4">
          <p className="text-xs text-slate-500">多个 ID 使用逗号分隔</p>
          <div className="flex gap-3">
            <input
              value={deleteIds}
              onChange={e => setDeleteIds(e.target.value)}
              placeholder="id1, id2, id3"
              className="input-field flex-1 font-mono text-sm"
            />
            <button
              onClick={handleDelete}
              disabled={deleteLoading || !deleteIds.trim()}
              className="flex items-center gap-2 rounded-xl bg-red-600/80 px-5 py-2.5 text-sm text-white
                         hover:bg-red-500 disabled:opacity-50 transition-colors shrink-0"
            >
              {deleteLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              删除
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   主组件
   ================================================================ */
export default function KnowledgeBase() {
  const [activeTab, setActiveTab] = useState<TabKey>('upload')

  return (
    <div className="mx-auto max-w-5xl h-full flex flex-col p-6 overflow-hidden">
      {/* 标题栏 */}
      <div className="mb-6 shrink-0">
        <h1 className="text-xl font-bold text-white">知识库管理</h1>
        <p className="text-sm text-slate-500 mt-1">上传文档、查询知识库、管理向量数据库</p>
      </div>

      {/* Tab 导航 */}
      <div className="flex gap-1 mb-6 shrink-0 bg-white/[0.03] rounded-xl p-1">
        {TAB_ITEMS.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex-1 flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm
                          transition-all duration-200 ${
                isActive
                  ? 'bg-violet-600/20 text-violet-300 shadow-sm'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
              }`}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab 内容 */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            {activeTab === 'upload' && <UploadTab />}
            {activeTab === 'search' && <SearchTab />}
            {activeTab === 'milvus' && <MilvusTab />}
            {activeTab === 'manage' && <ManageTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
