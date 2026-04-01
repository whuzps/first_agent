import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bot, MessageSquarePlus, MessagesSquare, BookOpen,
  ShoppingCart, ChevronLeft, ChevronRight, LogOut, User,
  Settings,
} from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { useAuthStore } from '../../stores/authStore'
import SessionList from './SessionList'

const NAV_ITEMS = [
  { key: '/', icon: MessagesSquare, label: '智能对话', roles: ['admin', 'user'] },
  { key: '/kb', icon: BookOpen, label: '知识库', roles: ['admin'] },
  { key: '/orders', icon: ShoppingCart, label: '订单管理', roles: ['admin', 'user'] },
]

/** 侧边栏组件 */
export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { createSession } = useChatStore()
  const { user, logout } = useAuthStore()

  const handleNewChat = () => {
    createSession()
    navigate('/')
  }

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  const filteredNav = NAV_ITEMS.filter((n) => n.roles.includes(user?.role || 'user'))

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 280 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative flex h-screen flex-col border-r border-white/5 bg-slate-950/50"
    >
      {/* 头部 Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-white/5 px-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl gradient-brand">
          <Bot className="h-5 w-5 text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <h1 className="text-sm font-bold gradient-text">智能客服</h1>
              <p className="text-[10px] text-slate-600">E-commerce Agent</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 新建对话按钮 */}
      <div className="px-3 py-3">
        <button
          onClick={handleNewChat}
          className={`flex items-center gap-2 rounded-xl border border-dashed border-violet-500/30 py-2.5 text-sm text-violet-300 hover:bg-violet-500/10 hover:border-violet-500/50 transition-all w-full ${
            collapsed ? 'justify-center px-2' : 'px-4'
          }`}
        >
          <MessageSquarePlus className="h-4 w-4 shrink-0" />
          {!collapsed && <span>新对话</span>}
        </button>
      </div>

      {/* 会话列表（仅在聊天页展示） */}
      {location.pathname === '/' && !collapsed && (
        <div className="flex-1 overflow-y-auto px-3 pb-2">
          <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
            历史会话
          </p>
          <SessionList />
        </div>
      )}

      {/* 导航菜单 */}
      <div className={`${location.pathname === '/' && !collapsed ? '' : 'flex-1'} px-3 py-2`}>
        {(!collapsed || location.pathname !== '/') && (
          <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
            {collapsed ? '' : '功能'}
          </p>
        )}
        {filteredNav.map(({ key, icon: Icon, label }) => {
          const active = location.pathname === key
          return (
            <button
              key={key}
              onClick={() => navigate(key)}
              className={`mb-0.5 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all ${
                active
                  ? 'bg-violet-500/15 text-violet-300 font-medium'
                  : 'text-slate-500 hover:bg-white/5 hover:text-slate-300'
              } ${collapsed ? 'justify-center' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{label}</span>}
            </button>
          )
        })}
      </div>

      {/* 底部用户区 */}
      <div className="border-t border-white/5 px-3 py-3">
        <div
          className={`flex items-center gap-3 rounded-xl px-3 py-2 ${collapsed ? 'justify-center' : ''}`}
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-xs font-bold text-white">
            {user?.username?.charAt(0).toUpperCase() || 'U'}
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="truncate text-sm text-white">{user?.username}</p>
              <p className="text-[10px] text-slate-600">
                {user?.role === 'admin' ? '管理员' : '用户'}
              </p>
            </div>
          )}
          {!collapsed && (
            <button
              onClick={handleLogout}
              className="rounded-lg p-1.5 text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="退出登录"
            >
              <LogOut className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* 折叠按钮 */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-20 flex h-6 w-6 items-center justify-center rounded-full border border-white/10 bg-slate-900 text-slate-500 hover:text-white hover:border-violet-500/50 transition-colors z-50"
      >
        {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
      </button>
    </motion.aside>
  )
}
