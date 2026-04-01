import { Outlet } from 'react-router-dom'
import Sidebar from '../sidebar/Sidebar'

/** 主布局：侧边栏 + 内容区 */
export default function AppLayout() {
  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        <Outlet />
      </main>
    </div>
  )
}
