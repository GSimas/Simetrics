import { useState } from 'react';
import { Check, Copy, Download, FolderOpen, Pencil, Trash2, X } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Input } from '@/components/ui/input';
import type { ProjectMeta } from '@/lib/project';
import { useLocale } from '@/state/locale.store';

export interface ProjectCardProps {
  project: ProjectMeta;
  onOpen: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onDuplicate: (id: string) => void;
  onExport: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ProjectCard({
  project,
  onOpen,
  onRename,
  onDuplicate,
  onExport,
  onDelete,
}: ProjectCardProps) {
  const { locale, t } = useLocale();
  const [isEditing, setIsEditing] = useState(false);
  const [draftName, setDraftName] = useState(project.name);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);

  const updatedAt = new Date(project.updatedAt).toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });

  const commitRename = (): void => {
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== project.name) onRename(project.id, trimmed);
    setIsEditing(false);
  };

  const cancelRename = (): void => {
    setDraftName(project.name);
    setIsEditing(false);
  };

  return (
    <Card className="flex flex-col shadow-xs transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        {isEditing ? (
          <div className="flex items-center gap-1.5">
            <Input
              autoFocus
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') commitRename();
                if (event.key === 'Escape') cancelRename();
              }}
              className="h-8"
            />
            <Button variant="ghost" size="icon" className="size-8 shrink-0 cursor-pointer" onClick={commitRename}>
              <Check className="size-4" aria-hidden />
              <span className="sr-only">{t('project_card_rename_save')}</span>
            </Button>
            <Button variant="ghost" size="icon" className="size-8 shrink-0 cursor-pointer" onClick={cancelRename}>
              <X className="size-4" aria-hidden />
              <span className="sr-only">{t('project_card_rename_cancel')}</span>
            </Button>
          </div>
        ) : (
          <CardTitle className="truncate text-base" title={project.name}>
            {project.name}
          </CardTitle>
        )}
        <CardDescription>
          {t('project_card_docs').replace('{count}', project.docCount.toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US'))}
          {' · '}
          {t('project_card_updated').replace('{date}', updatedAt)}
        </CardDescription>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-between gap-4 pt-0">
        {project.sourceFiles.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {project.sourceFiles.map((file, index) => (
              <Badge key={`${file.name}-${index}`} variant="outline" className="font-normal">
                {file.name}
              </Badge>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            variant="gradient"
            size="sm"
            onClick={() => onOpen(project.id)}
            className="flex-1 cursor-pointer font-medium"
          >
            <FolderOpen className="size-4" aria-hidden />
            {t('project_card_open')}
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="size-8 cursor-pointer"
            title={t('project_card_rename')}
            aria-label={t('project_card_rename')}
            onClick={() => setIsEditing(true)}
          >
            <Pencil className="size-4" aria-hidden />
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="size-8 cursor-pointer"
            title={t('project_card_duplicate')}
            aria-label={t('project_card_duplicate')}
            onClick={() => onDuplicate(project.id)}
          >
            <Copy className="size-4" aria-hidden />
          </Button>

          <Button
            variant="outline"
            size="icon"
            className="size-8 cursor-pointer"
            title={t('project_card_export')}
            aria-label={t('project_card_export')}
            onClick={() => onExport(project.id)}
          >
            <Download className="size-4" aria-hidden />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="size-8 cursor-pointer text-muted-foreground hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-950 dark:hover:text-red-300"
            title={t('project_card_delete')}
            aria-label={t('project_card_delete')}
            onClick={() => setConfirmDeleteOpen(true)}
          >
            <Trash2 className="size-4" aria-hidden />
          </Button>
        </div>
      </CardContent>

      <ConfirmDialog
        open={confirmDeleteOpen}
        onOpenChange={setConfirmDeleteOpen}
        title={t('project_delete_confirm_title')}
        description={t('project_delete_confirm_desc').replace('{name}', project.name)}
        confirmLabel={t('project_card_delete')}
        cancelLabel={t('project_delete_cancel')}
        onConfirm={() => onDelete(project.id)}
      />
    </Card>
  );
}
