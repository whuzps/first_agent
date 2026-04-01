import { create } from 'zustand'
import type { AuthState, User } from '../types/auth'

/** 认证状态管理 */
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,

  login: (user: User, token: string) => {
    localStorage.setItem('auth_token', token)
    localStorage.setItem('auth_user', JSON.stringify(user))
    if (user.tenant_id) localStorage.setItem('tenantId', user.tenant_id)
    set({ user, token, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    localStorage.removeItem('tenantId')
    set({ user: null, token: null, isAuthenticated: false })
  },

  restoreFromStorage: () => {
    const token = localStorage.getItem('auth_token')
    const raw = localStorage.getItem('auth_user')
    if (token && raw) {
      try {
        const user: User = JSON.parse(raw)
        set({ user, token, isAuthenticated: true })
      } catch {
        set({ user: null, token: null, isAuthenticated: false })
      }
    }
  },
}))
