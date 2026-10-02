import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFile } from 'node:fs/promises'
import { GET } from './route'

vi.mock('node:fs/promises', () => ({ readFile: vi.fn() }))

const mockReadFile = vi.mocked(readFile)

afterEach(() => {
  vi.resetAllMocks()
  vi.unstubAllEnvs()
})

describe('GET /api/tested-models', () => {
  it('returns public pair metadata without Triton implementation names', async () => {
    mockReadFile.mockResolvedValue(JSON.stringify({
      schemaVersion: 1,
      generatedAt: '2026-01-01T00:00:00Z',
      pairs: [{ id: 'model__h200', modelId: 'org/model', systemId: 'h200_sxm', tritonModel: 'internal_name', dashboardPath: 'dashboard/model__h200.json' }],
    }) as never)

    const response = await GET()
    const body = await response.json()

    expect(body).toEqual({
      status: 'available',
      generatedAt: '2026-01-01T00:00:00Z',
      pairs: [{ id: 'model__h200', modelId: 'org/model', systemId: 'h200_sxm', dashboardPath: 'dashboard/model__h200.json' }],
    })
    expect(JSON.stringify(body)).not.toContain('tritonModel')
  })

  it('returns an unavailable empty response when the registry is absent', async () => {
    mockReadFile.mockRejectedValue(new Error('missing'))
    const response = await GET()
    expect(response.status).toBe(200)
    expect(await response.json()).toMatchObject({ status: 'unavailable', pairs: [] })
  })
})
