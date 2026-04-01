import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Bot, Eye, EyeOff, User, Lock, ArrowRight, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import { useAuthStore } from '../../stores/authStore'
import { login, signup } from '../../services/api'

/** 登录 / 注册页面 */
export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPwd, setShowPwd] = useState(false)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const authLogin = useAuthStore((s) => s.login)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password.trim()) {
      toast.error('请填写用户名和密码')
      return
    }
    setLoading(true)
    try {
      const fn = isLogin ? login : signup
      const resp = await fn({ username: username.trim(), password })
      if (resp.code !== 0) {
        toast.error(resp.message || '操作失败')
        return
      }
      authLogin(resp.data.user, resp.data.token)
      toast.success(isLogin ? '登录成功' : '注册成功')
      navigate('/', { replace: true })
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message
      toast.error(msg || '网络错误，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950">
      {/* 动态背景 */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-40 -top-40 h-[500px] w-[500px] rounded-full bg-violet-600/20 blur-[120px] animate-pulse-slow" />
        <div className="absolute -right-40 -bottom-40 h-[600px] w-[600px] rounded-full bg-indigo-600/20 blur-[120px] animate-pulse-slow" style={{ animationDelay: '1.5s' }} />
        <div className="absolute left-1/2 top-1/3 h-[300px] w-[300px] -translate-x-1/2 rounded-full bg-purple-600/10 blur-[100px] animate-float" />
      </div>

      {/* 网格背景 */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* 卡片 */}
      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="relative z-10 w-full max-w-md px-4"
      >
        <div className="glass-card p-8 sm:p-10">
          {/* Logo */}
          <div className="mb-8 flex flex-col items-center">
            <motion.div
              className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl gradient-brand shadow-lg shadow-purple-500/30"
              whileHover={{ scale: 1.05, rotate: 5 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <Bot className="h-8 w-8 text-white" />
            </motion.div>
            <h1 className="text-2xl font-bold gradient-text">智能客服助手</h1>
            <p className="mt-1 text-sm text-slate-500">AI 驱动的电商客服解决方案</p>
          </div>

          {/* 选项卡 */}
          <div className="relative mb-8 flex rounded-xl bg-white/5 p-1">
            <motion.div
              className="absolute inset-y-1 rounded-lg gradient-brand"
              animate={{ left: isLogin ? '4px' : '50%', width: 'calc(50% - 4px)' }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            />
            {['登录', '注册'].map((label, i) => (
              <button
                key={label}
                onClick={() => setIsLogin(i === 0)}
                className="relative z-10 flex-1 py-2 text-sm font-medium transition-colors"
                style={{ color: (i === 0) === isLogin ? '#fff' : 'rgb(148,163,184)' }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* 表单 */}
          <AnimatePresence mode="wait">
            <motion.form
              key={isLogin ? 'login' : 'signup'}
              initial={{ opacity: 0, x: isLogin ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: isLogin ? 20 : -20 }}
              transition={{ duration: 0.2 }}
              onSubmit={handleSubmit}
              className="space-y-5"
            >
              {/* 用户名 */}
              <div className="group relative">
                <User className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                <input
                  type="text"
                  placeholder="用户名"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="input-field pl-11"
                  autoComplete="username"
                />
              </div>

              {/* 密码 */}
              <div className="group relative">
                <Lock className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-violet-400" />
                <input
                  type={showPwd ? 'text' : 'password'}
                  placeholder="密码"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-11 pr-11"
                  autoComplete={isLogin ? 'current-password' : 'new-password'}
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(!showPwd)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPwd ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>

              {/* 提交 */}
              <button
                type="submit"
                disabled={loading}
                className="btn-primary flex w-full items-center justify-center gap-2 py-3"
              >
                {loading ? (
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>{isLogin ? '登录' : '注册'}</span>
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>
            </motion.form>
          </AnimatePresence>

          {/* 底部 */}
          <p className="mt-6 text-center text-xs text-slate-600">
            {isLogin ? '还没有账号？' : '已有账号？'}
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="ml-1 text-violet-400 hover:text-violet-300 transition-colors"
            >
              {isLogin ? '立即注册' : '去登录'}
            </button>
          </p>
        </div>
      </motion.div>
    </div>
  )
}
