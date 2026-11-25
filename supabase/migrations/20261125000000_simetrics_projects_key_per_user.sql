-- O id do projeto nasce no navegador: duas contas usadas no mesmo navegador podem ter o
-- mesmo id. A chave passa a ser (user_id, id), e cada conta tem a sua cópia.
alter table public.simetrics_projects drop constraint simetrics_projects_pkey;
alter table public.simetrics_projects add primary key (user_id, id);
