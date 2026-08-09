---
description: Commit current work on a feature branch, merge into main, delete the branch, and push
argument-hint: [optional short description of the work]
allowed-tools: Bash(git:*)
---

## Current repository state
- Current branch: !`git branch --show-current`
- Status: !`git status --short`
- Changed files: !`git diff --name-only HEAD`

## Your task

Using the git CLI, perform these steps in order:

1. Create a new feature branch named `feature/<short-kebab-summary>`, where the summary
   is derived from the changed files above (or from "$ARGUMENTS" if it was provided).
2. Stage all changes and commit with a clear conventional commit message you generate
   from the diff — format `type: summary` (e.g. `feat: add pdf ingestion endpoint`,
   `fix: handle empty document upload`).
3. Switch to `main` and merge the feature branch into it.
4. After a successful merge, delete the feature branch.
5. Push `main` to `origin`.
6. Print a concise summary: the branch used, the commit message, the files changed,
   and the push result.

If any step fails, stop immediately and report the exact error instead of continuing.