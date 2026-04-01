import { format, formatDistanceToNow, isToday, isYesterday } from 'date-fns'
import { zhCN } from 'date-fns/locale'

/** 格式化会话时间（今天/昨天/具体日期） */
export function formatSessionTime(ts: number): string {
  const d = new Date(ts * 1000)
  if (isToday(d)) return format(d, 'HH:mm')
  if (isYesterday(d)) return '昨天'
  return format(d, 'MM/dd')
}

/** 格式化消息时间 */
export function formatMessageTime(date: Date): string {
  return format(date, 'HH:mm')
}

/** 相对时间 */
export function relativeTime(ts: number): string {
  return formatDistanceToNow(new Date(ts * 1000), { addSuffix: true, locale: zhCN })
}

/** 文件大小格式化 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}

/** 截断文本 */
export function truncate(text: string, len: number): string {
  return text.length > len ? text.slice(0, len) + '...' : text
}

/** 复制文本到剪贴板 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  }
}

/** 图片转 Base64 */
export function imageToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/** 状态标签颜色 */
export function statusColor(status: string): string {
  const map: Record<string, string> = {
    done: 'text-emerald-400',
    processing: 'text-amber-400',
    pending: 'text-slate-400',
    failed: 'text-red-400',
    已发货: 'text-blue-400',
    已完成: 'text-emerald-400',
    待发货: 'text-amber-400',
    已取消: 'text-red-400',
    待付款: 'text-orange-400',
  }
  return map[status] || 'text-slate-400'
}
