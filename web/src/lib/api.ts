export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /* Reasoning models stream their thinking on a separate delta field before
     the answer. Kept apart from `content` so it can be rendered as its own
     block and excluded from what is sent back as conversation history. */
  reasoning?: string
  /* How an assistant turn's last generation ended: the server's finish_reason;
     "aborted" / "error" when the client stopped or lost the stream; or
     "incomplete" when the stream closed without a finish_reason, which the engine
     always sends last, so its absence means the reply was cut off. Kept on
     the message so it travels with the transcript through slot switches and
     archives; never sent to the server. */
  finish?: string
}

interface OpenAIError {
  error?: { message?: string }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
  /* Whether a message list ending on an assistant turn is continued rather than
     answered fresh (COLI_CONTINUE_ASSISTANT). Absent on older servers. */
  continue_assistant?: boolean
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export interface SessionInfo {
  email: string
  role: string
  is_admin: boolean
  /** Hard ceiling the gateway applies to max_tokens; 0 = no ceiling. */
  max_output_tokens: number
  /**
   * Live web grounding, reported by the gateway: the model was trained in 2023,
   * so the gateway can retrieve fresh pages and put them in the prompt. Absent
   * on a direct engine connection or an older gateway — treat it as disabled.
   */
  web_search?: {
    enabled: boolean
    mode: string
    provider: string
    results: number
    full_pages: boolean
  }
}

/**
 * Session details from the auth gateway. Only meaningful when the dashboard is
 * served *through* the gateway — talking to the engine directly returns 404,
 * so callers must treat a failure as "no session" rather than an error.
 */
export async function getMe(signal?: AbortSignal): Promise<SessionInfo | null> {
  try {
    const response = await fetch("/me", {
      headers: { Accept: "application/json" },
      signal,
      credentials: "same-origin",
    })
    if (!response.ok) return null
    return (await response.json()) as SessionInfo
  } catch {
    return null
  }
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  /** Ask the gateway to ground this turn in live web results. */
  webSearch?: boolean
  signal: AbortSignal
  onDelta: (text: string) => void
  onReasoning?: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      messages: options.messages.map(({ role, content }) => ({ role, content })),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      // Only sent when true: an explicit false would override the gateway's own
      // `auto` heuristic and switch grounding off for a time-sensitive question.
      ...(options.webSearch ? { web_search: true } : {}),
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    const reasoning = choice?.delta?.reasoning_content
    if (reasoning) options.onReasoning?.(reasoning)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}

/* Modalita brio: il modello non genera, assegna una probabilita a ogni opzione
 * ammessa. Il ciclo (fotografia del prefisso condiviso, una lettura per
 * opzione, normalizzazione per lunghezza) sta nel gateway: qui si manda una
 * richiesta e si riceve una distribuzione. */
export interface BrioChoice {
  option: string
  p: number
  logprob: number
  mean_logprob: number
  tokens: number
}

export interface BrioResponse {
  answer: string
  entropy: number
  normalize: "mean" | "sum"
  choices: BrioChoice[]
  usage: { prompt_tokens: number; completion_tokens: number; read_tokens: number; total_tokens: number }
}

export async function askBrio(
  baseUrl: string,
  apiKey: string,
  model: string,
  state: string,
  question: string,
  options: string[],
  signal?: AbortSignal,
): Promise<BrioResponse> {
  const response = await fetch(endpoint(baseUrl, "brio"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ model, state, question, options }),
    signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as BrioResponse
}
