-- GST Foresight — Supabase schema
-- Run this once in the Supabase SQL editor before first ingest.

-- Enable pgvector extension
create extension if not exists vector;

-- ── chunks table ─────────────────────────────────────────────────────────────
-- Stores embedded document chunks from the ingest pipeline.
-- Written by Python (service_role key), read by Edge Function (anon key via RPC).

create table if not exists chunks (
  id           text primary key,          -- chunk_id from chunker.py
  doc_id       text not null,
  source_id    text not null,
  date         text,
  topic_tags   text,                      -- comma-separated
  chunk_index  int,
  content      text not null,
  embedding    vector(384),               -- all-MiniLM-L6-v2 dimensions
  inserted_at  timestamptz default now()
);

-- HNSW index for fast cosine similarity search
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- ── usage table ──────────────────────────────────────────────────────────────
-- Tracks query counts per IP for free-tier rate limiting.
-- One row per IP, reset_at marks when the 30-day window expires.

create table if not exists usage (
  ip           text primary key,
  query_count  int not null default 0,
  reset_at     timestamptz not null default (now() + interval '30 days')
);

-- ── match_chunks RPC ─────────────────────────────────────────────────────────
-- Called by the Edge Function to run semantic search.
-- Returns top-k chunks ordered by cosine similarity.

create or replace function match_chunks (
  query_embedding  vector(384),
  match_count      int     default 8,
  match_threshold  float   default 0.3
)
returns table (
  id          text,
  doc_id      text,
  source_id   text,
  date        text,
  topic_tags  text,
  content     text,
  similarity  float
)
language sql stable
security definer
set search_path = public, extensions
as $$
  select
    id,
    doc_id,
    source_id,
    date,
    topic_tags,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from chunks
  where 1 - (embedding <=> query_embedding) >= match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ── check_and_increment_usage RPC ────────────────────────────────────────────
-- Atomically checks rate limit and increments counter.
-- Returns { allowed: bool, remaining: int } so the Edge Function
-- makes a single DB call instead of two.
--
-- SECURITY DEFINER: runs with the function owner's privileges (service_role),
-- not the caller's. This means anon users can call this RPC but cannot
-- directly read or modify the usage table — all writes go through this
-- function which enforces the rate-limit logic.

create or replace function check_and_increment_usage (
  client_ip   text,
  free_limit  int default 5
)
returns json
language plpgsql
security definer
set search_path = public
as $$
declare
  rec        usage%rowtype;
begin
  -- Ensure the row exists before we lock it.
  -- ON CONFLICT DO NOTHING means only the first concurrent INSERT wins;
  -- all others skip silently, then proceed to the SELECT FOR UPDATE below.
  insert into usage (ip, query_count, reset_at)
  values (client_ip, 0, now() + interval '30 days')
  on conflict (ip) do nothing;

  -- Row-level lock: the second of two concurrent requests blocks here
  -- until the first commits, so only one request at a time executes the
  -- check-and-increment logic. This closes the TOCTOU race completely.
  select * into rec from usage where ip = client_ip for update;

  -- Reset window if expired before checking limit
  if rec.reset_at < now() then
    update usage
    set query_count = 1, reset_at = now() + interval '30 days'
    where ip = client_ip
    returning * into rec;
    return json_build_object(
      'allowed',   true,
      'remaining', free_limit - 1,
      'reset_at',  rec.reset_at
    );
  end if;

  -- Blocked: already at or over limit
  if rec.query_count >= free_limit then
    return json_build_object(
      'allowed',   false,
      'remaining', 0,
      'reset_at',  rec.reset_at
    );
  end if;

  -- Allowed: atomically increment
  update usage set query_count = query_count + 1 where ip = client_ip;
  return json_build_object(
    'allowed',   true,
    'remaining', free_limit - rec.query_count - 1,
    'reset_at',  rec.reset_at
  );
end;
$$;

-- ── Row-level security ────────────────────────────────────────────────────────
-- chunks: no direct REST access for anon. All reads go through match_chunks RPC
-- which runs as SECURITY DEFINER (owner privileges, bypasses RLS). This prevents
-- bulk corpus extraction via GET /rest/v1/chunks?select=*.
alter table chunks enable row level security;
-- No anon select policy — direct REST reads are blocked.
-- match_chunks is SECURITY DEFINER so it reads chunks as the function owner,
-- not as the calling anon role, and therefore bypasses this RLS restriction.
create policy "service_role full access on chunks"
  on chunks for all to service_role using (true) with check (true);

-- usage: no direct anon access — all reads/writes go through the
-- check_and_increment_usage SECURITY DEFINER function above.
-- Direct REST access to /rest/v1/usage is blocked for anon users,
-- preventing rate-limit bypass (reset own counter) or DoS (exhaust others).
alter table usage enable row level security;
create policy "service_role only on usage"
  on usage for all to service_role using (true) with check (true);
