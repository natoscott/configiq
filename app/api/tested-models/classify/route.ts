import { readFile } from 'node:fs/promises'
import { NextResponse } from 'next/server'
import { buildFeatureVector, type FeatureSchema } from '@/lib/tested-models/feature-vector'
import type { CanonicalClassifierInput } from '@/lib/tested-models/types'

export const dynamic = 'force-dynamic'

interface RegistryPair {
  id: string
  modelId: string
  systemId: string
  classifier: string
  tritonModel: string
  featureSchema: FeatureSchema
  thresholds?: {
    within_region_probability?: number
    uncertain_probability?: number
  }
}

interface Registry {
  schemaVersion: number
  pairs: RegistryPair[]
}

function registryPath(): string {
  return `${process.env.TESTED_MODELS_DIR ?? '/tested-models'}/registry.json`
}

function statusForProbability(probability: number, thresholds: RegistryPair['thresholds']) {
  const within = thresholds?.within_region_probability ?? 0.85
  const uncertain = thresholds?.uncertain_probability ?? 0.5
  if (probability >= within) return 'within_validated_range' as const
  if (probability >= uncertain) return 'near_validated_boundary' as const
  return 'outside_validated_range' as const
}

export async function POST(request: Request) {
  try {
    const input = await request.json() as CanonicalClassifierInput
    const registry = JSON.parse(await readFile(registryPath(), 'utf8')) as Registry
    const pair = registry.pairs.find(candidate =>
      candidate.modelId === input.effective.model_id && candidate.systemId === input.effective.system_id)
    if (!pair) {
      return NextResponse.json({ status: 'validation_unavailable', confidence: null, pair_id: null, classifier: null, reasons: ['No classifier is available for this model and GPU system.'] })
    }

    const tritonUrl = process.env.TRITON_INFERENCE_URL ?? 'http://host.containers.internal:8000'
    const vector = buildFeatureVector(input, pair.featureSchema)
    const response = await fetch(`${tritonUrl}/v2/models/${encodeURIComponent(pair.tritonModel)}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        inputs: [{ name: 'input__0', shape: [1, vector.length], datatype: 'FP32', data: vector }],
        outputs: [{ name: 'output__0' }],
      }),
      signal: AbortSignal.timeout(5000),
    })
    if (!response.ok) throw new Error(`Triton returned ${response.status}`)
    const result = await response.json() as { outputs?: Array<{ data?: unknown[] }> }
    const data = result.outputs?.[0]?.data
    const probability = Array.isArray(data) && typeof data[0] === 'number' ? data[0] : null
    if (probability == null || !Number.isFinite(probability)) throw new Error('Triton returned no within-region probability')

    return NextResponse.json({
      status: statusForProbability(probability, pair.thresholds),
      confidence: probability,
      pair_id: pair.id,
      classifier: pair.classifier,
      reasons: [],
    })
  } catch {
    return NextResponse.json({ status: 'validation_unavailable', confidence: null, pair_id: null, classifier: null, reasons: ['Performance-envelope classification is unavailable.'] })
  }
}
