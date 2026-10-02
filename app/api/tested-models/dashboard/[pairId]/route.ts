import { readFile } from 'node:fs/promises'
import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(_request: Request, context: { params: Promise<{ pairId: string }> }) {
  const { pairId } = await context.params
  if (!/^[a-z0-9_]+$/.test(pairId)) {
    return NextResponse.json({ error: 'Invalid pair ID' }, { status: 400 })
  }
  try {
    const path = `${process.env.TESTED_MODELS_DIR ?? '/tested-models'}/dashboard/${pairId}.json`
    return NextResponse.json(JSON.parse(await readFile(path, 'utf8')), { headers: { 'Cache-Control': 'no-store' } })
  } catch {
    return NextResponse.json({ error: 'Dashboard data unavailable' }, { status: 404 })
  }
}
