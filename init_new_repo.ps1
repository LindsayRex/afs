<#
    init_new_repo.ps1

    Purpose:
        Create a new repository skeleton from the `repo-template` folder. This script
        copies files, replaces template placeholders, initializes a git repo, and can
        optionally create a remote GitHub repository via the `gh` CLI.

    Safety features:
        - `-DryRun` prints what would be changed without writing files.
        - `-VenvName` allows choosing a local venv folder name (default `.venv`).
        - `-ForceUpdate` updates placeholders in-place when the destination already
                contains a `.git` repository.

    Typical usage:
        .\init_new_repo.ps1 -Destination "C:\dev\my-new-project" -RepoName "my-new-project"

    Notes:
        - The script intentionally limits replacements to files smaller than 1MB to
            avoid touching binary or large data files.
        - When creating a remote, the script calls `gh repo create ...`. Ensure
            `gh` is installed and you are logged in (`gh auth login`).
#>
param(
        [Parameter(Mandatory=$true)] [string] $Destination,
        [Parameter(Mandatory=$true)] [string] $RepoName,
        # Optional GitHub owner used when creating a remote repository
        [Parameter(Mandatory=$false)] [string] $GitHubOwner = "GITHUB_OWNER_PLACEHOLDER",
        # Switch to create remote via gh
        [switch] $CreateRemote,
        # Dry-run: print changes but don't write files or init git
        [switch] $DryRun,
        # Use this venv name when populating workspace paths (default: .venv)
        [Parameter(Mandatory=$false)] [string] $VenvName = ".venv",
        # Force update in-place if destination already contains .git
        [switch] $ForceUpdate
)

function Copy-Template {
    <#
      Copy-Template

      Helper function to recursively copy the template folder contents to the
      destination. Uses Copy-Item with -Recurse and -Force so existing files are
      overwritten when copying into a non-empty directory.

      Parameters:
        $src - absolute path to the template root
        $dst - absolute path to the destination project root
    #>
    param($src, $dst)
    Write-Host "Copying template from $src to $dst"
    Copy-Item -Path $src -Destination $dst -Recurse -Force
}

<#
  Establish paths and basic checks

  $scriptRoot: the folder that contains this script (expected to be repo-template)
  $templateRoot: the parent folder of the script (where template content lives)
  $destinationFull: absolute path to the user's requested destination

  We normalize to FullName because path strings can differ by trailing slashes or
  relative segments; using FullName makes equality checks reliable on Windows.
#>
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Resolve the template root (this script lives inside repo-template)
$templateRoot = Resolve-Path (Join-Path $scriptRoot '..')

if (-Not (Test-Path $Destination)) {
    # Create the destination directory if it doesn't exist yet
    New-Item -ItemType Directory -Path $Destination | Out-Null
}

# Normalize full paths for safe comparisons
$destinationFull = (Get-Item -LiteralPath $Destination).FullName
$scriptRootFull = (Get-Item -LiteralPath $scriptRoot).FullName
$templateRootFull = $templateRoot.Path

# Prevent copying into the template itself — common mistake when testing
if ($destinationFull -eq $scriptRootFull -or $destinationFull -eq $templateRootFull) {
    Write-Error "Destination must not be the template folder itself"
    return
}

<#
    Handle existing git repos

    If the destination already contains a `.git` folder we assume it's an
    existing repository. Default behavior is to abort so we don't clobber a
    pre-existing repo. Use `-ForceUpdate` to allow quiet in-place updates of
    placeholders (useful when reapplying the template to an existing project).
#>
$existingGit = Test-Path (Join-Path $destinationFull '.git')
if ($existingGit -and -not $ForceUpdate -and -not $DryRun) {
        Write-Error "Destination already contains a .git repository. To update placeholders in-place, re-run with -ForceUpdate."
        return
}

# DryRun: simulate replacements without copying or writing files
if ($DryRun) {
    # Dry-run mode: report which files would be changed and the replacement
    # values, but do not write anything to disk. This is useful to preview
    # effects on CI or developer machines before making irreversible changes.
    Write-Host "DryRun: scanning template files under $templateRootFull"
    Get-ChildItem -Path $templateRootFull -Recurse -File | Where-Object { $_.Length -lt 1048576 } | ForEach-Object {
        $rel = $_.FullName.Substring($templateRootFull.Length).TrimStart('\')
        $content = Get-Content -Raw -LiteralPath $_.FullName -ErrorAction SilentlyContinue
        if ($null -eq $content) { return }
        $found = @()
        foreach ($k in $placeholders.Keys) {
            if ($content.Contains($k)) { $found += $k }
        }
        if ($found.Count -gt 0) {
            Write-Host "Would update: $rel"
            foreach ($f in $found) { Write-Host "  - $f -> ${placeholders[$f]}" }
        }
    }
    Write-Host "DryRun complete. No files were written."
    return
}

# Otherwise, perform copy unless we're updating in-place
$doCopy = -not ($existingGit -and $ForceUpdate)
if ($doCopy) {
    # Copy the template contents into the destination. This will overwrite
    # files if the destination is not empty — we only copy when not updating
    # in-place via -ForceUpdate.
    Copy-Template -src $templateRootFull -dst $destinationFull

    # After copying, remove any .git folder that accidentally came from the
    # template (we want a fresh git init for the new project, not the
    # template's git metadata).
    try {
        $gitFolder = Join-Path $destinationFull '.git'
        if (Test-Path $gitFolder) {
            Write-Host "Removing copied .git folder at $gitFolder"
            Remove-Item -LiteralPath $gitFolder -Recurse -Force -ErrorAction Stop
        }
    } catch {
        Write-Warning "Couldn't remove copied .git folder: $_"
    }
}

# Replace placeholders in copied files
#
# The placeholders map a token found in template files (e.g. REPO_NAME_PLACEHOLDER)
# to the desired runtime value for the new project. We use literal string
# replacements (not regex) to avoid escape/character-class surprises.
$placeholders = @{
    'REPO_NAME_PLACEHOLDER' = $RepoName
    'GITHUB_OWNER_PLACEHOLDER' = $GitHubOwner
    'PYTHON_VERSION_PLACEHOLDER' = '3.12'
    # The workspace interpreter placeholder expands to something like:
    # ${workspaceFolder}\.venv\Scripts\python.exe — use $VenvName to let
    # teams choose a different venv folder name if desired.
    'PYTHON_INTERPRETER_PLACEHOLDER' = '${workspaceFolder}\' + $VenvName + '\\Scripts\\python.exe'
    'ENV_FOLDER_PLACEHOLDER' = $VenvName
    'REPO_ROOT_PATH_PLACEHOLDER' = $destinationFull
}

# Branch/path token defaults (YAML-friendly replacements)
$branchListYaml = "[ main, dev, test ]"
$pathsIgnoreYaml = @"
- 'archive/**'
- 'background/**'
- 'Design/**'
- 'docs/**'
- '**/*.md'
"@
$placeholders['__BRANCHES__'] = $branchListYaml
$placeholders['__PATHS_IGNORE__'] = $pathsIgnoreYaml
$placeholders['__POLICY_BRANCHES__'] = $branchListYaml

<#
  Perform replacements

  Iterate over files under the destination and do literal string Replace() on
  each placeholder token. We intentionally limit this to files < 1MB to avoid
  accidentally touching fixtures, data, or binary assets.
#>
Get-ChildItem -Path $destinationFull -Recurse -File | Where-Object { $_.Length -lt 1048576 } | ForEach-Object {
    $path = $_.FullName
    # Skip any files inside a .git folder just in case
    if ($path -match "\\\.git\\\") { return }
    $content = Get-Content -Raw -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $content) { return }
    $changed = $false
    foreach ($k in $placeholders.Keys) {
        if ($content.Contains($k)) {
            # Use literal Replace to avoid regex escape issues
            $content = $content.Replace($k, $placeholders[$k])
            $changed = $true
        }
    }
    if ($changed) { Set-Content -LiteralPath $path -Value $content -Encoding UTF8 }
}

Push-Location $destinationFull
<#
  Git initialization

  If the destination did not contain a .git folder (fresh copy or new folder),
  initialize a git repo and make the initial commit. We check that `git` is
  available on PATH and fail early with a helpful message if it's not.
#>
# Ensure git exists before attempting to initialize
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "'git' is not available in PATH. Please install Git before running this script."
    Pop-Location
    return
}

if (-Not (Test-Path .git)) {
    # Init a new git repo and commit the initial snapshot
    git init
    git add .
    git commit -m "Initial template import for $RepoName"
}

if ($CreateRemote) {
    # Creating a remote requires GitHub CLI (`gh`). Check for it and call it
    # in a standard way: create public repo, set origin, and push the initial
    # commit.
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Error "GitHub CLI 'gh' not found in PATH. Install it or run without -CreateRemote."
        Pop-Location
        return
    }
    $fullName = "$GitHubOwner/$RepoName"
    gh repo create $fullName --public --source=. --remote=origin --push
}

Pop-Location
Write-Host "Template initialized at $Destination"
