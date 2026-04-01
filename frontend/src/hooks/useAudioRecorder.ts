import { useState, useRef, useCallback } from 'react'

interface AudioRecorderHook {
  isRecording: boolean
  duration: number
  start: () => Promise<void>
  stop: () => Promise<string | null>
  cancel: () => void
}

/** 语音录制 Hook，返回 Base64 编码的音频 */
export function useAudioRecorder(): AudioRecorderHook {
  const [isRecording, setIsRecording] = useState(false)
  const [duration, setDuration] = useState(0)
  const mediaRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval>>()

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    chunksRef.current = []
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.start(200)
    mediaRef.current = recorder
    setIsRecording(true)
    setDuration(0)
    timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000)
  }, [])

  const stop = useCallback(async (): Promise<string | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRef.current
      if (!recorder || recorder.state === 'inactive') {
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        recorder.stream.getTracks().forEach((t) => t.stop())
        clearInterval(timerRef.current)
        setIsRecording(false)
        const reader = new FileReader()
        reader.onloadend = () => resolve(reader.result as string)
        reader.readAsDataURL(blob)
      }
      recorder.stop()
    })
  }, [])

  const cancel = useCallback(() => {
    const recorder = mediaRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = null
      recorder.stop()
      recorder.stream.getTracks().forEach((t) => t.stop())
    }
    clearInterval(timerRef.current)
    setIsRecording(false)
    setDuration(0)
  }, [])

  return { isRecording, duration, start, stop, cancel }
}
