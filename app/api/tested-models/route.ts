import { readFile } from 'node:fs/promises'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

interface RegistryPair {
  id: string
  modelId: string
  systemId: string
  metrics?: unknown
  thresholds?: unknown
  dashboardPath?: string
}

interface Registry {
  schemaVersion: number
  generatedAt: string
  pairs: RegistryPair[]
}

function registryPath(): string {
  return `${process.env.TESTED_MODELS_DIR ?? '/tested-models'}/registry.json`
}

export async function GET() {
  try {
    const registry = JSON.parse(await readFile(registryPath(), 'utf8')) as Registry
    if (registry.schemaVersion !== 1 || !Array.isArray(registry.pairs)) {
      throw new Error('Invalid tested-model registry')
    }

    return NextResponse.json(
      {
        status: 'available',
        generatedAt: registry.generatedAt,
        pairs: registry.pairs.map(({ id, modelId, systemId, metrics, thresholds, dashboardPath }) => ({
          id,
          modelId,
          systemId,
          metrics,
          thresholds,
          dashboardPath,
        })),
      },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  } catch {
    return NextResponse.json(
      { status: 'unavailable', generatedAt: null, pairs: [] },
      { headers: { 'Cache-Control': 'no-store' } },
    )
  }
}
