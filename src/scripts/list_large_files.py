#!/usr/bin/env python3
"""List large files in a repo and report whether they're tracked by Git and Git LFS.

Usage: python scripts/list_large_files.py [--min-mb 1] [--top 100] [path]
"""
import argparse
import os
import subprocess
import sys
from fnmatch import fnmatch


def run(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, shell=False)
        return out.decode('utf-8', errors='replace')
    except subprocess.CalledProcessError:
        return None


def load_lfs_names(repo):
    out = run(['git', 'lfs', 'ls-files', '--name-only'], cwd=repo)
    if not out:
        return set()
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return set(names)


def is_git_tracked(repo, relpath):
    # git ls-files --error-unmatch <path> returns 0 when tracked
    try:
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', relpath], cwd=repo, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def is_git_ignored(repo, relpath):
    try:
        subprocess.check_call(['git', 'check-ignore', '-q', relpath], cwd=repo,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def human(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='.', help='Repository root to scan')
    p.add_argument('--min-mb', type=float, default=1.0, help='Minimum file size in MB to report')
    p.add_argument('--top', type=int, default=200, help='Top N files to show')
    args = p.parse_args()

    repo = os.path.abspath(args.path)
    if not os.path.isdir(repo):
        print('Path not found:', repo)
        sys.exit(1)

    # Preload LFS list
    lfs_names = load_lfs_names(repo)

    files = []
    min_bytes = int(args.min_mb * 1024 * 1024)
    for root, dirs, filenames in os.walk(repo):
        # skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        for fn in filenames:
            fp = os.path.join(root, fn)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not os.path.isfile(fp):
                continue
            size = st.st_size
            if size < min_bytes:
                continue
            rel = os.path.relpath(fp, repo).replace('\\','/')
            files.append((size, rel))

    files.sort(reverse=True, key=lambda x: x[0])

    print(f"Scanning: {repo}\nFound {len(files)} files >= {args.min_mb} MB. Showing top {args.top}.\n")
    print(f"{'Size':>10}  {'LFS':3}  {'GIT':3}  {'IGN':3}  Path")
    print('-'*80)
    shown = 0
    for size, rel in files:
        tracked_git = is_git_tracked(repo, rel)
        tracked_lfs = rel in lfs_names
        ignored = is_git_ignored(repo, rel)
        s = human(size)
        lfs_status = 'YES' if tracked_lfs else 'NO '
        git_status = 'YES' if tracked_git else 'NO '
        ignore_status = 'YES' if ignored else 'NO '
        print(f"{s:>10}  {lfs_status:3}  {git_status:3}  {ignore_status:3}  {rel}")
        shown += 1
        if shown >= args.top:
            break

    if len(files) == 0:
        print('No files above threshold found.')


if __name__ == '__main__':
    main()
