import { createClient } from '@supabase/supabase-js';

export const config = { runtime: 'edge' };

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Content-Type': 'application/json',
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: CORS });
}

function buildPrompt(query, chunks) {
  const context = chunks
    .map((c, i) => {
      const date = c.date ? ` (${c.date.slice(0, 10)})` : '';
      const topics = c.topic_tags ? ` [${c.topic_tags}]` : '';
      return `[${i + 1}] ${c.source_id}${date}${topics}\n${c.content}`;
    })
    .join('\n\n---\n\n');

  return `You are a GST regulatory foresight analyst for India.

Using ONLY the corpus excerpts below, answer the user's query with:
1. A probability assessment of whether the regulatory change is likely (low / medium / high)
2. The specific signals from the documents that drive this assessment
3. Expected timeframe (next council meeting / next budget / 2–3 quarters / next FY)
4. Concrete things the user should monitor or prepare for

Stay strictly grounded in the documents. If the corpus does not contain enough signal, say so clearly.

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

export default async function handler(request) {
  try {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }
    if (request.method !== 'POST') {
      return json({ error: 'method_not_allowed' }, 405);
    }

    let query;
    try {
      ({ query } = await request.json());
    } catch {
      return json({ error: 'invalid_json' }, 400);
    }
    if (!query || typeof query !== 'string' || query.trim().length < 5) {
      return json({ error: 'query_too_short', message: 'Please enter a more specific question.' }, 400);
    }

    const cleanQuery = query.trim().slice(0, 500);

    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY,
      { auth: { persistSession: false } },
    );

    // ── Rate limit ─────────────────────────────────────────────────────────────
    const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? '0.0.0.0';
    const { data: rl, error: rlErr } = await supabase.rpc('check_and_increment_usage', {
      client_ip: ip,
      free_limit: 5,
    });

    if (rlErr) {
      console.error('[rate-limit]', rlErr.message);
      // fail open — don't block on rate limit DB error
    } else if (rl && !rl.allowed) {
      return json({ error: 'rate_limited', message: 'You have used all 5 free queries for this month.', reset_at: rl.reset_at }, 429);
    }

    // ── Embed query via Supabase edge function ─────────────────────────────────
    let embedding;
    try {
      const embedRes = await fetch(`${process.env.SUPABASE_URL}/functions/v1/embed`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.SUPABASE_ANON_KEY}`,
          'Content-Type': 'application/json',
          ...(process.env.EMBED_SECRET ? { 'X-Embed-Secret': process.env.EMBED_SECRET } : {}),
        },
        body: JSON.stringify({ text: cleanQuery }),
      });
      if (!embedRes.ok) {
        const errText = await embedRes.text();
        throw new Error(`embed ${embedRes.status}: ${errText}`);
      }
      ({ embedding } = await embedRes.json());
    } catch (e) {
      console.error('[embed]', e.message);
      return json({ error: 'embed_error', message: 'Failed to embed query.' }, 502);
    }

    // ── Vector search ──────────────────────────────────────────────────────────
    const { data: chunks, error: searchErr } = await supabase.rpc('match_chunks', {
      query_embedding: embedding,
      match_count: 8,
    });

    if (searchErr) {
      console.error('[match_chunks]', searchErr.message);
      return json({ error: 'search_error', message: 'Vector search unavailable.' }, 502);
    }
    if (!chunks?.length) {
      return json({ error: 'no_context', message: 'No relevant documents found.' });
    }

    // ── Sarvam ─────────────────────────────────────────────────────────────────
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
            { role: 'system', content: 'You are a GST regulatory foresight analyst for India. Provide structured, evidence-grounded assessments based strictly on the documents provided.' },
            { role: 'user', content: buildPrompt(cleanQuery, chunks) },
          ],
          temperature: 0.2,
          max_tokens: 1024,
          stream: false,
        }),
      });
    } catch (e) {
      console.error('[sarvam fetch]', e.message);
      return json({ error: 'llm_unavailable', message: 'Analysis service temporarily unavailable.' }, 502);
    }

    if (!sarvamRes.ok) {
      const errText = await sarvamRes.text();
      console.error('[sarvam]', sarvamRes.status, errText);
      return json({ error: 'llm_error', message: 'Failed to generate analysis.' }, 502);
    }

    const sarvamData = await sarvamRes.json();
    const answer = sarvamData.choices?.[0]?.message?.content ?? '';

    return json({
      answer,
      sources: chunks.slice(0, 4).map(c => ({
        source_id: c.source_id,
        date: c.date,
        topic_tags: c.topic_tags,
        excerpt: (c.content ?? '').slice(0, 250),
      })),
      remaining_queries: rl?.remaining ?? null,
      query: cleanQuery,
    });

  } catch (e) {
    // top-level safety net — should never fire but prevents opaque 500s
    console.error('[query unhandled]', e.message, e.stack);
    return json({ error: 'internal', message: e.message }, 500);
  }
}
