-- Analytics Forge v2 — Phase 3 Schema
-- Run in Supabase SQL Editor or via supabase db push

-- Profiles
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT,
    org TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own profile" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Users can insert own profile" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, display_name)
    VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- User Uploads
CREATE TABLE IF NOT EXISTS public.user_uploads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    upload_meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE public.user_uploads ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own uploads" ON public.user_uploads FOR ALL USING (auth.uid() = user_id);

-- User Mappings
CREATE TABLE IF NOT EXISTS public.user_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    domain TEXT DEFAULT 'generic',
    mapping_json JSONB DEFAULT '{}',
    source_columns_json JSONB DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, name)
);
ALTER TABLE public.user_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own mappings" ON public.user_mappings FOR ALL USING (auth.uid() = user_id);

-- User Sessions
CREATE TABLE IF NOT EXISTS public.user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_name TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE public.user_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own sessions" ON public.user_sessions FOR ALL USING (auth.uid() = user_id);

-- SAP Configs
CREATE TABLE IF NOT EXISTS public.sap_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    sap_host TEXT DEFAULT '',
    sap_client TEXT DEFAULT '',
    sap_user TEXT DEFAULT '',
    sap_password TEXT DEFAULT '',
    service_url TEXT DEFAULT '',
    odata_path TEXT DEFAULT '',
    schedule TEXT DEFAULT 'disabled',
    enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE public.sap_configs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own sap config" ON public.sap_configs FOR ALL USING (auth.uid() = user_id);

-- Cron Jobs
CREATE TABLE IF NOT EXISTS public.cron_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    job_type TEXT NOT NULL DEFAULT 'monday_report',
    schedule TEXT DEFAULT '0 7 * * 1',
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT false,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, job_type)
);
ALTER TABLE public.cron_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users manage own cron jobs" ON public.cron_jobs FOR ALL USING (auth.uid() = user_id);
