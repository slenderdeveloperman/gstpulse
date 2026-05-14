/**
 * api/embed.js — internal embedding endpoint (Supabase Edge Function)
 *
 * Accepts { text: string }, returns { embedding: number[] } using
 * Transformers.js running in the V8 edge runtime via WASM.
 * Uses all-MiniLM-L6-v2 — same model as the Python ingest pipeline.
 *
 * INTERNAL ONLY — called by api/query.js, not exposed to end users.
 * Protected by a shared secret (EMBED_SECRET env var) so that knowing
 * the Supabase anon key alone is not sufficient to call this endpoint.
 *
 * Env vars:
 *   EMBED_SECRET — shared secret set in both this function and query.js
 */

import { pipeline } from '@xenova/transformers';

export const config = { runtime: 'edge' };

// Module-level singleton — warm across requests in the same isolate
let extractor = null;

async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractor;
}

// Restrict to calls from query.js (same Vercel deployment origin)
const CORS = {
  'Access-Control-Allow-Origin': process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : 'http://localhost:3000',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Embed-Secret',
  'Content-Type': 'application/json',
};

function err(msg, status) {
  return new Response(JSON.stringify({ error: msg }), { status, headers: CORS });
}

export default {
  async fetch(request) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== 'POST') {
      return err('method_not_allowed', 405);
    }

    // Shared-secret gate — prevents abuse by callers who only know the anon key
    const secret = process.env.EMBED_SECRET;
    if (secret && request.headers.get('X-Embed-Secret') !== secret) {
      return err('unauthorized', 401);
    }

    let text;
    try {
      ({ text } = await request.json());
    } catch {
      return err('invalid_json', 400);
    }
    if (!text || typeof text !== 'string') {
      return err('missing_text', 400);
    }

    try {
      const embed = await getExtractor();
      const output = await embed(text.slice(0, 512), { pooling: 'mean', normalize: true });
      const embedding = Array.from(output.data);
      return new Response(JSON.stringify({ embedding }), { status: 200, headers: CORS });
    } catch (e) {
      console.error('[embed] error:', e);
      return err('embed_failed', 502);
    }
  },
};
