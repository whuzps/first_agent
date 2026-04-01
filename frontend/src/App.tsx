import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useAuthStore } from './stores/authStore'
import AuthPage from './components/auth/AuthPage'
import AppLayout from './components/layout/AppLayout'
import ChatView from './components/chat/ChatView'
import KnowledgeBase from './components/kb/KnowledgeBase'
import OrderManager from './components/orders/OrderManager'

/** 需要认证的路由守卫 */
function PrivateRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />
}

/** 已登录则跳转首页 */
function PublicRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  return isAuthenticated ? <Navigate to="/" replace /> : <>{children}</>
}

export default function App() {
  const restoreFromStorage = useAuthStore((s) => s.restoreFromStorage)

  useEffect(() => {
    restoreFromStorage()
  }, [restoreFromStorage])

  return (
    <BrowserRouter>
      <Toaster
        position="top-center"
        toastOptions={{
          duration: 3000,
          style: {
            background: 'rgba(30, 30, 50, 0.9)',
            color: '#e2e8f0',
            border: '1px solid rgba(255,255,255,0.1)',
            backdropFilter: 'blur(12px)',
            borderRadius: '12px',
            fontSize: '14px',
          },
        }}
      />
      <Routes>
        <Route
          path="/login"
          element={
            <PublicRoute>
              <AuthPage />
            </PublicRoute>
          }
        />
        <Route
          element={
            <PrivateRoute>
              <AppLayout />
            </PrivateRoute>
          }
        >
          <Route path="/" element={<ChatView />} />
          <Route path="/kb" element={<KnowledgeBase />} />
          <Route path="/orders" element={<OrderManager />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
