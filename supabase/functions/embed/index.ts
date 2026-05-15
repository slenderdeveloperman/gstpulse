/**
 * supabase/functions/embed/index.ts
 *
 * Generates a 384-dim embedding for a text string using Supabase's
 * built-in gte-small model (hosted — zero local model load).
 *
 * Called by:
 *   - api/query.js  (Vercel Edge Function) at query time
 *   - processors/embedder.py (Python ingest) at index time
 *
 * Both callers must send:
 *   Authorization: Bearer <SUPABASE_ANON_KEY>
 *   X-Embed-Secret: <EMBED_SECRET>   (set in Supabase dashboard → Functions → Secrets)
 *
 * POST { "text": "..." }  →  { "embedding": [0.1, 0.2, ...] }   (384 floats)
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, apikey, content-type, x-embed-secret",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// Lazy-initialised — model stays warm across requests in the same isolate
let session: Supabase.ai.Session | null = null;

async function getSession(): Promise<Supabase.ai.Session> {
  if (!session) {
    // gte-small: 384-dim, fast, good multilingual coverage
    session = new Supabase.ai.Session("gte-small");
  }
  return session;
}

Deno.serve(async (req: Request) => {
  // CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  if (req.method !== "POST") {
    return Response.json(
      { error: "method_not_allowed" },
      { status: 405, headers: CORS_HEADERS },
    );
  }

  // Shared-secret gate — required, not optional.
  // EMBED_SECRET must be set in Supabase → Functions → embed → Secrets.
  // If missing, return 503 so misconfiguration is loud rather than silently open.
  const embedSecret = Deno.env.get("EMBED_SECRET");
  if (!embedSecret) {
    console.error("[embed] EMBED_SECRET is not configured — refusing all requests");
    return Response.json(
      { error: "service_misconfigured" },
      { status: 503, headers: CORS_HEADERS },
    );
  }
  const provided = req.headers.get("x-embed-secret");
  if (!provided || provided !== embedSecret) {
    return Response.json(
      { error: "unauthorized" },
      { status: 401, headers: CORS_HEADERS },
    );
  }

  // Parse body
  let text: string;
  try {
    ({ text } = await req.json());
  } catch {
    return Response.json(
      { error: "invalid_json" },
      { status: 400, headers: CORS_HEADERS },
    );
  }

  if (!text || typeof text !== "string") {
    return Response.json(
      { error: "missing_text" },
      { status: 400, headers: CORS_HEADERS },
    );
  }

  try {
    const model = await getSession();
    const output = await model.run(text.slice(0, 512), {
      mean_pool: true,
      normalize: true,
    });
    const embedding = Array.from(output as Float32Array);

    return Response.json(
      { embedding },
      { headers: CORS_HEADERS },
    );
  } catch (e) {
    console.error("[embed] model error:", e);
    return Response.json(
      { error: "embed_failed" },
      { status: 502, headers: CORS_HEADERS },
    );
  }
});
