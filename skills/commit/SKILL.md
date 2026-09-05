---
name: commit
description: Guide for writing a good commit message and creating a commit via bash
---

# Commit Skill

Use this skill when you need to create a git commit.

## How to write a good commit message

- Subject line: concise, imperative style (e.g., "Add skills support"), max ~72 chars.
- Body (optional): explain *why* not just *what*, wrap at ~72 chars per line.
- Include what changed and why, reference issue or context if relevant.
- Keep commits focused — one logical change per commit.
- Avoid generic messages like "fix" or "update"; be specific.

## How to commit via bash

Follow these steps using the `bash` tool:

1. Check status and recent logs:
   ```bash
   git status
   git diff --staged
   git diff
   git log --oneline -n 10
   ```

2. Stage the files you want to commit:
   ```bash
   git add <files>
   # or stage all relevant changes, but avoid unrelated files
   git add -A
   ```

3. Create the commit with a good message:
   ```bash
   git commit -m "Your concise subject line

Optional body explaining why and what changed.
Wrap body lines around 72 chars.

Co-Authored-By: <optional>"
   ```

   For multi-line messages, use heredoc or multiple -m flags:
   ```bash
   git commit -m "Add skills support" -m "Implements skills/<name>/SKILL.md loading, read_skill tool, and commit example skill."
   ```

4. Verify:
   ```bash
   git log --oneline -n 5
   git show --stat HEAD
   ```

Tips:
- Run tests before committing (`pytest -q`).
- Ensure no leftover debug files.
- If you need to amend, use `git commit --amend`.
