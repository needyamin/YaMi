import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { resendFrom } from "./chat"

afterEach(() => vi.unstubAllGlobals())

const user = (content: string): ChatMessage => ({ id: "u", role: "user", content })

const assistant = (content: string): ChatMessage => ({ id: "a", role: "assistant", content })

describe("resendFrom", () => {
  it("returns the last user turn's text and the history before it", () => {
    const messages = [user("what is this?"), assistant("a cat")]
    expect(resendFrom(messages, 1)).toEqual({
      text: "what is this?",
      previous: [],
    })
  })

  it("uses the nearest user turn and leaves earlier history alone", () => {
    const first: ChatMessage = { id: "u0", role: "user", content: "hi" }
    const reply: ChatMessage = { id: "a0", role: "assistant", content: "hello" }
    const second: ChatMessage = { id: "u1", role: "user", content: "and this?" }
    const last: ChatMessage = { id: "a1", role: "assistant", content: "a bird" }
    const messages = [first, reply, second, last]
    expect(resendFrom(messages, 3)).toEqual({
      text: "and this?",
      previous: [first, reply],
    })
  })

  it("returns null when there is no user turn to resend", () => {
    expect(resendFrom([assistant("orphan")], 0)).toBeNull()
  })
})

describe("request body", () => {
  it("sends message content as a plain string", async () => {
    const retry = resendFrom([user("describe this"), assistant("skipped")], 1)
    const fetchMock = vi.fn().mockResolvedValue(new Response("data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    }))
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "u", role: "user", content: retry!.text }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      messages: Array<{ content: unknown }>
    }
    // This model is text-only: content must stay a string, never the
    // content-array form with image_url parts.
    expect(body.messages[0].content).toBe("describe this")
  })
})
