import type {
  CanonicalClassifierInput,
  ClassifierEffectiveConfiguration,
} from './types'

export interface FeatureSchema {
  feature_names: string[]
  numeric_medians: Record<string, number>
}

function systemFamily(systemId: string): string {
  const normalized = systemId.toLowerCase()
  if (normalized.startsWith('h200')) return 'H200'
  if (normalized.startsWith('b200')) return 'B200'
  if (normalized.startsWith('b300')) return 'B300'
  if (normalized.startsWith('mi300x')) return 'MI300X'
  if (normalized.startsWith('mi355x')) return 'MI355X'
  return systemId
}

function backendVersionLabel(effective: ClassifierEffectiveConfiguration): string {
  if (!effective.backend_version) return 'unknown'
  const backend = effective.backend.toLowerCase()
  const prefix = backend === 'vllm' ? 'vLLM'
    : backend === 'trt-llm' ? 'TRT-LLM'
      : backend === 'rhaiis' ? 'RHAIIS'
        : backend === 'sglang' ? 'sglang'
          : effective.backend
  return `${prefix}-${effective.backend_version}`
}

function modelConfigNumber(config: Record<string, unknown> | null, keys: string[]): number | null {
  for (const key of keys) {
    const value = config?.[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return null
}

export function buildFeatureValues(input: CanonicalClassifierInput): Record<string, string | number> {
  const effective = input.effective
  const modelConfig = input.model_config
  const tp = effective.tp_size || 1
  const pp = effective.pp_size || 1
  const dp = effective.dp_size || 1
  const ep = effective.moe_ep_size || 1
  const cp = effective.cp_size || 1
  const isl = effective.isl || 0
  const osl = effective.osl || 0
  const concurrency = effective.concurrency || 0
  const sharedPrefix = effective.shared_prefix_tokens || 0
  const parallelism = tp * pp * dp * ep * cp
  const tokenBudget = (isl + osl) * Math.max(concurrency, 1)

  return {
    model_id: effective.model_id,
    system_id: effective.system_id,
    backend: effective.backend,
    version: backendVersionLabel(effective),
    accelerator: systemFamily(effective.system_id),
    request_type: 'unknown',
    precision: effective.weight_precision ?? 'unknown',
    weight_quantization: effective.weight_precision ?? 'unknown',
    kv_quantization: effective.kv_cache_precision ?? 'unknown',
    tp,
    pp,
    dp,
    ep,
    cp,
    isl,
    osl,
    concurrency,
    prefix_tokens: sharedPrefix,
    prefix_count: 0,
    prefix_caching: effective.prefix_caching_enabled == null ? 0 : Number(effective.prefix_caching_enabled),
    shared_prefix: sharedPrefix,
    moe_num_experts: modelConfigNumber(modelConfig, ['num_experts', 'num_local_experts']) ?? 0,
    moe_top_k: modelConfigNumber(modelConfig, ['num_experts_per_tok', 'num_selected_experts']) ?? 0,
    token_budget: tokenBudget,
    parallelism,
    memory_pressure: tokenBudget / Math.max(parallelism, 1),
    prefix_pressure: sharedPrefix * Math.max(concurrency, 1),
    sequence_pressure: (isl + osl) / Math.max(parallelism, 1),
  }
}

export function buildFeatureVector(input: CanonicalClassifierInput, schema: FeatureSchema): number[] {
  const values = buildFeatureValues(input)
  return schema.feature_names.map((name) => {
    const separator = name.indexOf('=')
    if (separator >= 0) {
      const field = name.slice(0, separator)
      const expected = name.slice(separator + 1)
      return String(values[field] ?? '') === expected ? 1 : 0
    }
    const value = values[name]
    return typeof value === 'number' && Number.isFinite(value)
      ? value
      : schema.numeric_medians[name] ?? 0
  })
}
