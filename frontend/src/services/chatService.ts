import * as api from './api'
import type { ChatResponse, SendMessageParams } from '../types/chat'
import type { SuggestionEvent } from '../types/api'

let currentES: EventSource | null = null

/** 发送聊天消息 */
export async function sendMessage(
  params: SendMessageParams,
  threadId?: string,
): Promise<ChatResponse> {
  const body: Record<string, unknown> = { query: params.content }
  if (threadId) body.thread_id = threadId
  if (params.images?.length) body.images = params.images
  if (params.audio) {
    body.audio = params.audio
    if (params.asrLanguage) body.asr_language = params.asrLanguage
    body.asr_itn = params.asrItn ?? true
  }
  if (params.quotedMessage) body.quoted_message = params.quotedMessage.content

  const resp = await api.sendMessage(body as Parameters<typeof api.sendMessage>[0])
  if (!resp.answer && resp.answer !== '') throw new Error('服务返回异常')
  return resp
}

/** 流式发送聊天消息 */
export async function sendMessageStream(
  params: SendMessageParams,
  threadId: string | undefined,
  onToken: (content: string) => void,
  onDone: (resp: ChatResponse) => void,
  onError?: (message: string) => void,
): Promise<void> {
  const body: Record<string, unknown> = { query: params.content }
  if (threadId) body.thread_id = threadId
  if (params.images?.length) body.images = params.images
  if (params.audio) {
    body.audio = params.audio
    if (params.asrLanguage) body.asr_language = params.asrLanguage
    body.asr_itn = params.asrItn ?? true
  }
  if (params.quotedMessage) body.quoted_message = params.quotedMessage.content

  await api.sendMessageStream(
    body as Parameters<typeof api.sendMessage>[0],
    onToken,
    (data) =>
      onDone({
        route: data.route,
        answer: data.answer,
        sources: data.sources,
      }),
    onError,
  )
}

/** 开始建议流 */
export function startSuggestionStream(
  threadId: string,
  callback: (evt: SuggestionEvent) => void,
): void {
  stopSuggestionStream()
  currentES = api.subscribeSuggestions(threadId, callback, () => {
    currentES = null
  })
}

/** 停止建议流 */
export function stopSuggestionStream(): void {
  if (currentES) {
    currentES.close()
    currentES = null
  }
}
