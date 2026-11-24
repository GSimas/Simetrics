-- Simetrics: projetos na nuvem por usuário (aplicada em scientata-lab-development).
-- Metadados aqui; o registro completo (gzip) e os PDFs ficam no bucket privado `simetrics`,
-- sob a pasta do próprio usuário: {user_id}/projects/{id}.json.gz e {user_id}/pdfs/{hash}.*
create table public.simetrics_projects (
  id uuid primary key,
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  name text not null,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  doc_count integer not null default 0,
  source_files jsonb not null default '[]'::jsonb,
  dedup_strategy text not null default 'none'
);
comment on table public.simetrics_projects is 'Simetrics: metadados dos projetos salvos na nuvem; o registro completo fica em storage simetrics/{user_id}/projects/{id}.json.gz.';
create index simetrics_projects_user_updated on public.simetrics_projects (user_id, updated_at desc);

alter table public.simetrics_projects enable row level security;
create policy "simetrics_projects_own_select" on public.simetrics_projects for select to authenticated using (user_id = (select auth.uid()));
create policy "simetrics_projects_own_insert" on public.simetrics_projects for insert to authenticated with check (user_id = (select auth.uid()));
create policy "simetrics_projects_own_update" on public.simetrics_projects for update to authenticated using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy "simetrics_projects_own_delete" on public.simetrics_projects for delete to authenticated using (user_id = (select auth.uid()));
grant select, insert, update, delete on public.simetrics_projects to authenticated;
revoke all on public.simetrics_projects from anon;

insert into storage.buckets (id, name, public, file_size_limit)
values ('simetrics', 'simetrics', false, 52428800);

create policy "simetrics_storage_own_select" on storage.objects for select to authenticated
  using (bucket_id = 'simetrics' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "simetrics_storage_own_insert" on storage.objects for insert to authenticated
  with check (bucket_id = 'simetrics' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "simetrics_storage_own_update" on storage.objects for update to authenticated
  using (bucket_id = 'simetrics' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'simetrics' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "simetrics_storage_own_delete" on storage.objects for delete to authenticated
  using (bucket_id = 'simetrics' and (storage.foldername(name))[1] = (select auth.uid())::text);
