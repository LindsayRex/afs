# PowerShell script to search C: drive for Python environments
# Searches common locations for Python installations and environments

Write-Host "Searching for Python environments on C: drive..." -ForegroundColor Green

# Common search patterns for installations
$installPatterns = @(
    "C:\Python*",
    "C:\Program Files\Python*",
    "C:\Program Files (x86)\Python*",
    "C:\Users\*\AppData\Local\Programs\Python*",
    "C:\ProgramData\Anaconda*",
    "C:\ProgramData\Miniconda*",
    "C:\Users\*\AppData\Local\Continuum\anaconda*",
    "C:\Users\*\AppData\Local\Continuum\miniconda*"
)



# Patterns for virtual environments (venv/virtualenv)
$venvPatterns = @(
    "C:\Users\*\*\venv",
    "C:\Users\*\*\.venv",
    "C:\Users\*\*\env",
    "C:\Users\*\*\.env",
    "C:\Users\*\*\virtualenv",
    "C:\Program Files\*\venv",
    "C:\Program Files\*\.venv"
)

$foundEnvs = @()

# Search for installations
foreach ($pattern in $installPatterns) {
    try {
        $dirs = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $pythonExe = Join-Path $dir.FullName "python.exe"
            if (Test-Path $pythonExe) {
                # Try to get version
                try {
                    $version = & $pythonExe --version 2>&1 | Out-String
                    $version = $version.Trim()
                } catch {
                    $version = "Unknown version"
                }

                $foundEnvs += [PSCustomObject]@{
                    Path = $dir.FullName
                    Version = $version
                    Type = if ($dir.FullName -like "*conda*" -or $dir.FullName -like "*anaconda*" -or $dir.FullName -like "*miniconda*") { "Conda" } elseif ($dir.FullName -like "*Python*") { "Python" } else { "Other" }
                }
            }
        }
    } catch {
        # Skip inaccessible paths
    }
}

# Search for virtual environments (look for pyvenv.cfg)
Write-Host "Searching for virtual environments..." -ForegroundColor Yellow
foreach ($pattern in $venvPatterns) {
    try {
        $dirs = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $pyvenvCfg = Join-Path $dir.FullName "pyvenv.cfg"
            if (Test-Path $pyvenvCfg) {
                $pythonExe = Join-Path $dir.FullName "Scripts\python.exe"
                if (Test-Path $pythonExe) {
                    try {
                        $version = & $pythonExe --version 2>&1 | Out-String
                        $version = $version.Trim()
                    } catch {
                        $version = "Unknown version"
                    }

                    $foundEnvs += [PSCustomObject]@{
                        Path = $dir.FullName
                        Version = $version
                        Type = "venv"
                    }
                }
            }
        }
    } catch {
        # Skip
    }
}

# Also search recursively for pyvenv.cfg in user directories
Write-Host "Searching recursively for pyvenv.cfg files..." -ForegroundColor Yellow
try {
    $pyvenvFiles = Get-ChildItem -Path "C:\Users" -Filter "pyvenv.cfg" -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $pyvenvFiles) {
        $venvDir = $file.DirectoryName
        $pythonExe = Join-Path $venvDir "Scripts\python.exe"
        if (Test-Path $pythonExe) {
            try {
                $version = & $pythonExe --version 2>&1 | Out-String
                $version = $version.Trim()
            } catch {
                $version = "Unknown version"
            }

            $foundEnvs += [PSCustomObject]@{
                Path = $venvDir
                Version = $version
                Type = "venv"
            }
        }
    }
} catch {
    # Skip
}

# Search for Conda environments
Write-Host "Searching for Conda environments..." -ForegroundColor Yellow
$condaEnvPaths = @(
    "C:\Users\*\.conda\envs\*",
    "C:\ProgramData\Miniconda*\envs\*",
    "C:\ProgramData\Anaconda*\envs\*"
)

foreach ($pattern in $condaEnvPaths) {
    try {
        $dirs = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $pythonExe = Join-Path $dir.FullName "python.exe"
            if (Test-Path $pythonExe) {
                try {
                    $version = & $pythonExe --version 2>&1 | Out-String
                    $version = $version.Trim()
                } catch {
                    $version = "Unknown version"
                }

                $foundEnvs += [PSCustomObject]@{
                    Path = $dir.FullName
                    Version = $version
                    Type = "Conda"
                }
            }
        }
    } catch {
        # Skip
    }
}

# Remove duplicates
$foundEnvs = $foundEnvs | Sort-Object -Property Path -Unique

# Display results
if ($foundEnvs.Count -eq 0) {
    Write-Host "No Python environments found on C: drive." -ForegroundColor Yellow
} else {
    Write-Host "`nFound Python environments:" -ForegroundColor Cyan
    $foundEnvs | Format-Table -AutoSize
}

# Check PATH
Write-Host "`nCurrent PATH entries containing 'python' or 'conda':" -ForegroundColor Cyan
$env:PATH -split ';' | Where-Object { $_ -like "*python*" -or $_ -like "*conda*" } | ForEach-Object { Write-Host "  $_" }

Write-Host "`nScript completed." -ForegroundColor Green
