/**
 * api/query.js — GST Foresight query endpoint (Vercel Edge Function)
 *
 * Flow:
 *   POST /api/query { query: string }
 *     → IP rate limit (Upstash Redis, 5 req / 30 days free tier)
 *     → Semantic search (Upstash Vector, all-MiniLM-L6-v2 built-in)
 *     → RAG prompt → Sarvam sarvam-m
 *     → { answer, sources, remaining_queries }
 *
 * Env vars required (set in Vercel dashboard):
 *   SARVAM_API_KEY
 *   UPSTASH_REDIS_REST_URL
 *   UPSTASH_REDIS_REST_TOKEN
 *   UPSTASH_VECTOR_REST_URL
 *   UPSTASH_VECTOR_REST_TOKEN
 */

import { ipAddress } from '@vercel/functions';
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';
import { Index } from '@upstash/vector';

export const config = { runtime: 'edge' };

// ─── Clients ─────────────────────────────────────────────────────────────────

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

// 5 queries per 30-day rolling window per IP (free tier)
const ratelimit = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(5, '30 d'),
  prefix: 'gst_rl',
});

const vectorIndex = new Index({
  url: process.env.UPSTASH_VECTOR_REST_URL,
  token: process.env.UPSTASH_VECTOR_REST_TOKEN,
});

// ─── CORS ─────────────────────────────────────────────────────────────────────

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Content-Type': 'application/json',
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: CORS });
}

// ─── Prompt builder ───────────────────────────────────────────────────────────

function buildPrompt(query, chunks) {
  const context = chunks
    .map((c, i) => {
      const meta = c.metadata ?? {};
      const source = meta.source_id ?? 'unknown';
      const date = meta.date ? ` (${meta.date.slice(0, 10)})` : '';
      const topics = meta.topic_tags ? ` [${meta.topic_tags}]` : '';
      return `[${i + 1}] ${source}${date}${topics}\n${c.data ?? ''}`;
    })
    .join('\n\n---\n\n');

  return `You are a GST regulatory foresight analyst for India.

Using ONLY the corpus excerpts below, answer the user's query with:
1. A probability assessment of whether the regulatory change is likely (low / medium / high)
2. The specific signals from the documents that drive this assessment
3. Expected timeframe (next council meeting / next budget / 2–3 quarters / next FY)
4. Concrete things the user should monitor or prepare for

Stay strictly grounded in the documents. If the corpus does not contain enough signal, say so clearly rather than speculating.

CORPUS EXCERPTS:
${context}

USER QUERY: ${query}

Respond in this format:
**Likelihood**: [Low / Medium / High] — [one-line reason]
**Timeframe**: [expected horizon]
**Key signals**:
- [signal 1 with source reference]
- [signal 2 with source reference]
**What to watch**: [specific monitoring advice]
**Confidence note**: [any caveats about data coverage]`;
}

// ─── Handler ─────────────────────────────────────────────────────────────────

export default {
  async fetch(request) {
    // Preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== 'POST') {
      return json({ error: 'method_not_allowed' }, 405);
    }

    // Parse body
    let query;
    try {
      ({ query } = await request.json());
    } catch {
      return json({ error: 'invalid_json' }, 400);
    }
    if (!query || typeof query !== 'string' || query.trim().length < 5) {
      return json({ error: 'query_too_short', message: 'Please enter a more specific question.' }, 400);
    }

    const cleanQuery = query.trim().slice(0, 500); // cap input length

    // Rate limit
    const ip = ipAddress(request) ?? '0.0.0.0';
    const { success, remaining, reset } = await ratelimit.limit(ip);
    if (!success) {
      return json(
        {
          error: 'rate_limited',
          message: 'You have used all 5 free queries for this month.',
          reset_at: new Date(reset).toISOString(),
        },
        429,
      );
    }

    // Semantic search — Upstash embeds the query using all-MiniLM-L6-v2
    let chunks;
    try {
      chunks = await vectorIndex.query({
        data: cleanQuery,
        topK: 8,
        includeData: true,
        includeMetadata: true,
      });
    } catch (e) {
      console.error('[vector] query failed:', e);
      return json({ error: 'search_error', message: 'Vector search unavailable.' }, 502);
    }

    if (!chunks.length) {
      return json({
        error: 'no_context',
        message: 'No relevant documents found in the corpus. The ingest pipeline may not have run yet.',
      });
    }

    // Call Sarvam
    let sarvamRes;
    try {
      sarvamRes = await fetch('https://api.sarvam.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'api-subscription-key': process.env.SARVAM_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'sarvam-m',
          messages: [
            {
              role: 'system',
              content:
                'You are a GST regulatory foresight analyst for India. Provide structured, evidence-grounded assessments of upcoming GST regulatory changes based strictly on the documents provided.',
            },
            { role: 'user', content: buildPrompt(cleanQuery, chunks) },
          ],
          temperature: 0.2,
          max_tokens: 1024,
          stream: false,
        }),
      });
    } catch (e) {
      console.error('[sarvam] fetch failed:', e);
      return json({ error: 'llm_unavailable', message: 'Analysis service temporarily unavailable.' }, 502);
    }

    if (!sarvamRes.ok) {
      const errText = await sarvamRes.text();
      console.error('[sarvam] error response:', errText);
      return json({ error: 'llm_error', message: 'Failed to generate analysis.' }, 502);
    }

    const sarvamData = await sarvamRes.json();
    const answer = sarvamData.choices?.[0]?.message?.content ?? '';

    return json({
      answer,
      sources: chunks.slice(0, 4).map((c) => ({
        source_id: c.metadata?.source_id,
        date: c.metadata?.date,
        topic_tags: c.metadata?.topic_tags,
        excerpt: (c.data ?? '').slice(0, 250),
      })),
      remaining_queries: remaining,
      query: cleanQuery,
    });
  },
};
