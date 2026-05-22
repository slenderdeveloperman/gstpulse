
export const config = { runtime: 'edge' };

// Razorpay webhook handler — activates Pro subscriptions after confirmed payment.
// This endpoint is NOT called by the browser; it's called by Razorpay's servers.
// Razorpay signs each webhook body with HMAC-SHA256 using the webhook secret,
// so we verify the signature before trusting any payload data.

const cleanEnv = v => (v ?? '').replace(/[^\x20-\x7E]/g, '');

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// Razorpay sends X-Razorpay-Signature: hex(HMAC-SHA256(body, webhook_secret))
async function verifyRazorpaySignature(body, signature, secret) {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const mac = await crypto.subtle.sign('HMAC', key, encoder.encode(body));
  const hex = Array.from(new Uint8Array(mac)).map(b => b.toString(16).padStart(2, '0')).join('');
  return hex === signature;
}

// Map Razorpay plan IDs to subscription plan names in the DB.
// These must match the plan IDs created in the Razorpay dashboard.
const PLAN_MAP = {
  'plan_pro_individual': { plan: 'pro_individual', days: 365 },
  'plan_pro_firm':       { plan: 'pro_firm',        days: 365 },
};

export default async function handler(request) {
  if (request.method !== 'POST') {
    return json({ error: 'method_not_allowed' }, 405);
  }

  const supabaseUrl = cleanEnv(process.env.SUPABASE_URL);
  const serviceKey  = cleanEnv(process.env.SUPABASE_SERVICE_KEY);
  const webhookSecret = cleanEnv(process.env.RAZORPAY_WEBHOOK_SECRET);

  if (!supabaseUrl || !serviceKey || !webhookSecret) {
    console.error('[activate] missing env vars');
    return json({ error: 'config_error' }, 500);
  }

  const rawBody = await request.text();
  const signature = request.headers.get('x-razorpay-signature') ?? '';

  const valid = await verifyRazorpaySignature(rawBody, signature, webhookSecret);
  if (!valid) {
    console.error('[activate] signature mismatch');
    return json({ error: 'invalid_signature' }, 401);
  }

  let payload;
  try {
    payload = JSON.parse(rawBody);
  } catch {
    return json({ error: 'invalid_json' }, 400);
  }

  // Only handle payment.captured — ignore other events (refund, failed, etc.)
  if (payload?.event !== 'payment.captured') {
    return json({ ok: true, skipped: payload?.event });
  }

  const payment = payload?.payload?.payment?.entity;
  if (!payment) return json({ error: 'missing_payment_entity' }, 400);

  const orderId   = payment.order_id;
  const paymentId = payment.id;
  const notes     = payment.notes ?? {};   // notes are set at order creation time

  // notes.user_id and notes.plan_id are set by the frontend when creating the Razorpay order.
  const userId = notes.user_id;
  const planId = notes.plan_id;

  if (!userId || !planId) {
    console.error('[activate] missing user_id or plan_id in notes', notes);
    return json({ error: 'missing_notes' }, 400);
  }

  const planConfig = PLAN_MAP[planId];
  if (!planConfig) {
    console.error('[activate] unknown plan_id', planId);
    return json({ error: 'unknown_plan' }, 400);
  }

  const validUntil = new Date(Date.now() + planConfig.days * 86_400_000).toISOString();

  // Insert subscription. razorpay_payment_id has a UNIQUE constraint — Razorpay
  // can retry webhooks, so a duplicate insert is silently ignored via ON CONFLICT.
  const supabaseHeaders = {
    'Authorization': `Bearer ${serviceKey}`,
    'apikey': serviceKey,
    'Content-Type': 'application/json',
    'Prefer': 'resolution=ignore-duplicates',
  };

  const insertRes = await fetch(`${supabaseUrl}/rest/v1/subscriptions`, {
    method: 'POST',
    headers: supabaseHeaders,
    body: JSON.stringify({
      user_id: userId,
      plan: planConfig.plan,
      valid_until: validUntil,
      razorpay_order_id: orderId,
      razorpay_payment_id: paymentId,
    }),
  });

  if (!insertRes.ok) {
    const errText = await insertRes.text();
    console.error('[activate] subscription insert failed', insertRes.status, errText);
    return json({ error: 'db_error' }, 502);
  }

  console.log(`[activate] Pro activated: user=${userId} plan=${planConfig.plan} until=${validUntil}`);
  return json({ ok: true, plan: planConfig.plan, valid_until: validUntil });
}
