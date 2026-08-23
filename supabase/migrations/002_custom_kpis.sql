-- Custom KPIs (KPI Studio) — optional; local SQLite used when Supabase unset
CREATE TABLE IF NOT EXISTS public.user_custom_kpis (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    spec_json JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE public.user_custom_kpis ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own custom kpis" ON public.user_custom_kpis
    FOR ALL USING (auth.uid() = user_id);
