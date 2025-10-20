import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def pwsh_available():
    return shutil.which('pwsh') or shutil.which('powershell')


@pytest.mark.skipif(not pwsh_available(), reason='PowerShell not available')
def test_init_new_repo_and_run_pytest():
    # Run init_new_repo.ps1 into a temporary directory and assert pytest in the generated project succeeds
    here = os.path.dirname(__file__)
    template_root = os.path.abspath(os.path.join(here, '..'))
    script = os.path.join(template_root, 'init_new_repo.ps1')
    with tempfile.TemporaryDirectory() as td:
        repo_name = 'my_new_template_test'
        dest = os.path.join(td, repo_name)
        # Call PowerShell script
        cmd = ['pwsh', '-NoProfile', '-NonInteractive', '-File', script, '-Destination', dest, '-RepoName', repo_name]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"init script failed: {proc.stdout}\n{proc.stderr}"

        # Run pytest in the generated project to exercise the skeleton
        proc2 = subprocess.run(['pytest', '-q'], cwd=dest, capture_output=True, text=True)
        assert proc2.returncode == 0, f"pytest in generated project failed: {proc2.stdout}\n{proc2.stderr}"
