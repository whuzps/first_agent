/** 通用API响应 */
export interface ApiResponse<T = unknown> {
  code: number
  message: string
  data: T
}

/** 健康检查状态 */
export interface HealthStatus {
  model: string
  kb_index: boolean
  orders_db: boolean
  metrics: Record<string, MetricSnapshot>
}

/** 性能指标 */
export interface MetricSnapshot {
  count: number
  min_ms: number
  max_ms: number
  avg_ms: number
  p95_ms: number
}

/** 模型信息 */
export interface ModelInfo {
  current: string
  models: string[]
}

/** 欢迎语 */
export interface GreetingResponse {
  message: string
  options: GreetingOption[]
}

/** 欢迎语选项 */
export interface GreetingOption {
  key: string
  title: string
  desc: string
}

/** 知识库上传任务 */
export interface KbTask {
  task_id: string
  filename: string
  file_size: number
  collection_name: string
  status: 'pending' | 'processing' | 'done' | 'failed'
  error?: string | null
  created_at: number
  updated_at: number
  chunk_count: number
}

/** SSE 建议事件 */
export interface SuggestionEvent {
  route?: string
  suggestions: string[]
  event: 'react_start' | 'react' | 'error' | 'suggest'
  final?: boolean
  error?: { code: string; message: string }
}

/** Milvus 集合信息 */
export interface MilvusCollection {
  collection_name: string
  row_count: number
  exists: boolean
}

/** 订单信息 */
export interface OrderInfo {
  order_id: string
  status: string
  amount: number
  create_time: string
  update_time: string
}
