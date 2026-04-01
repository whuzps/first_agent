import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, Package, Calendar, DollarSign,
  RefreshCw, AlertCircle, ShoppingBag,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { getOrder } from '../../services/api'
import type { OrderInfo } from '../../types/api'
import { statusColor } from '../../utils/helpers'

/** 订单管理页面 */
export default function OrderManager() {
  const [orderId, setOrderId] = useState('')
  const [order, setOrder] = useState<OrderInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [notFound, setNotFound] = useState(false)

  const handleSearch = async () => {
    const id = orderId.trim()
    if (!id) {
      toast.error('请输入订单号')
      return
    }
    setLoading(true)
    setNotFound(false)
    setOrder(null)
    try {
      const data = await getOrder(id)
      if (data && data.order_id) {
        setOrder(data)
      } else {
        setNotFound(true)
      }
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 404) {
        setNotFound(true)
      } else {
        toast.error('查询失败')
      }
    }
    setLoading(false)
  }

  const STATUS_LABELS: Record<string, string> = {
    shipped: '已发货',
    completed: '已完成',
    pending: '待发货',
    cancelled: '已取消',
    paid: '已付款',
    unpaid: '待付款',
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <div>
        <h1 className="text-xl font-bold text-white">订单管理</h1>
        <p className="text-sm text-slate-500">查询订单信息和状态</p>
      </div>

      {/* 搜索栏 */}
      <div className="glass-card p-5">
        <div className="flex gap-3">
          <div className="group relative flex-1">
            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
            <input
              value={orderId}
              onChange={(e) => setOrderId(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="输入订单号，例如 ORD-2024-1001"
              className="input-field pl-11"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className="btn-primary flex items-center gap-2 shrink-0"
          >
            {loading ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            查询
          </button>
        </div>
      </div>

      {/* 查询结果 */}
      <AnimatePresence mode="wait">
        {order && (
          <motion.div
            key="order"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="glass-card overflow-hidden"
          >
            {/* 头部 */}
            <div className="flex items-center justify-between border-b border-white/5 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/10">
                  <Package className="h-5 w-5 text-violet-400" />
                </div>
                <div>
                  <p className="font-medium text-white">{order.order_id}</p>
                  <p className="text-xs text-slate-500">订单编号</p>
                </div>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${statusColor(
                  STATUS_LABELS[order.status] || order.status,
                )} bg-white/5`}
              >
                {STATUS_LABELS[order.status] || order.status}
              </span>
            </div>

            {/* 详情 */}
            <div className="grid grid-cols-2 gap-4 p-6">
              <div className="flex items-center gap-3 rounded-xl bg-white/[0.03] px-4 py-3">
                <DollarSign className="h-5 w-5 text-emerald-400" />
                <div>
                  <p className="text-xs text-slate-500">订单金额</p>
                  <p className="text-lg font-bold text-white">¥{order.amount.toFixed(2)}</p>
                </div>
              </div>
              <div className="flex items-center gap-3 rounded-xl bg-white/[0.03] px-4 py-3">
                <ShoppingBag className="h-5 w-5 text-violet-400" />
                <div>
                  <p className="text-xs text-slate-500">订单状态</p>
                  <p className="text-lg font-bold text-white">
                    {STATUS_LABELS[order.status] || order.status}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3 rounded-xl bg-white/[0.03] px-4 py-3">
                <Calendar className="h-5 w-5 text-blue-400" />
                <div>
                  <p className="text-xs text-slate-500">创建时间</p>
                  <p className="text-sm text-white">{order.create_time}</p>
                </div>
              </div>
              <div className="flex items-center gap-3 rounded-xl bg-white/[0.03] px-4 py-3">
                <RefreshCw className="h-5 w-5 text-amber-400" />
                <div>
                  <p className="text-xs text-slate-500">更新时间</p>
                  <p className="text-sm text-white">{order.update_time}</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {notFound && (
          <motion.div
            key="notfound"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="glass-card flex flex-col items-center py-12"
          >
            <AlertCircle className="mb-3 h-12 w-12 text-slate-600" />
            <p className="text-sm text-slate-400">未找到订单</p>
            <p className="text-xs text-slate-600">请检查订单号是否正确</p>
          </motion.div>
        )}

        {!order && !notFound && !loading && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center py-16 text-slate-600"
          >
            <Package className="mb-3 h-16 w-16" />
            <p className="text-sm">输入订单号查询订单信息</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
