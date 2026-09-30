import { describe, expect, it, vi } from "vitest"

import { migrate, persistPublicSettings, stored, type StringStorage } from "./storage"

function memoryStorage(initial: Record<string, string> = {}): StringStorage & { values: Map<string, string> } {
  const values = new Map(Object.entries(initial))
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
    removeItem: (key) => { values.delete(key) },
  }
}

describe("browser settings persistence", () => {
  it("persists endpoint and model but removes legacy API credentials", () => {
    const storage = memoryStorage({ "colibri.apiKey": "legacy-secret" })
    persistPublicSettings(storage, "https://localhost/v1", "test-model")
    expect(Object.fromEntries(storage.values)).toEqual({
      "yami.baseUrl": "https://localhost/v1",
      "yami.model": "test-model",
    })
  })

  it("does not attempt to write an API key", () => {
    const storage = memoryStorage()
    const setItem = vi.spyOn(storage, "setItem")
    persistPublicSettings(storage, "http://localhost/v1", "test-model")
    expect(setItem).not.toHaveBeenCalledWith("yami.apiKey", expect.anything())
  })

  it("uses a fallback when storage access is unavailable", () => {
    expect(stored({ getItem: () => { throw new Error("denied") } }, "key", "fallback")).toBe("fallback")
  })
})

describe("colibri.* → yami.* migration", () => {
  it("carries an existing user's settings across the rename", () => {
    const storage = memoryStorage({
      "colibri.baseUrl": "https://box.local/v1",
      "colibri.model": "YaMi v1.0",
      "colibri.theme": "light",
    })
    migrate(storage)
    expect(Object.fromEntries(storage.values)).toEqual({
      "yami.baseUrl": "https://box.local/v1",
      "yami.model": "YaMi v1.0",
      "yami.theme": "light",
    })
  })

  it("drops a legacy API key rather than carrying it over", () => {
    const storage = memoryStorage({ "colibri.apiKey": "leaked-secret" })
    migrate(storage)
    expect(storage.values.has("yami.apiKey")).toBe(false)
    expect(storage.values.has("colibri.apiKey")).toBe(false)
  })

  it("never overwrites a value already set under the new prefix", () => {
    const storage = memoryStorage({
      "colibri.baseUrl": "https://stale/v1",
      "yami.baseUrl": "https://current/v1",
    })
    migrate(storage)
    expect(storage.values.get("yami.baseUrl")).toBe("https://current/v1")
  })

  it("is idempotent and tolerates empty or restricted storage", () => {
    const storage = memoryStorage({ "colibri.model": "m" })
    migrate(storage)
    migrate(storage)
    expect(storage.values.get("yami.model")).toBe("m")

    const broken: StringStorage = {
      getItem: () => { throw new Error("denied") },
      setItem: () => { throw new Error("denied") },
      removeItem: () => { throw new Error("denied") },
    }
    expect(() => migrate(broken)).not.toThrow()
  })
})
