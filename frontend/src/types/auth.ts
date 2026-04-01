/** 用户信息 */
export interface User {
  id: number
  username: string
  role: 'admin' | 'user'
  tenant_id: string
  created_at?: number
}

/** 登录/注册请求体 */
export interface AuthPayload {
  username: string
  password: string
}

/** 登录响应 */
export interface LoginResponse {
  token: string
  user: User
}

/** 认证状态 */
export interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (user: User, token: string) => void
  logout: () => void
  restoreFromStorage: () => void
}
