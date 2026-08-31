<#
.SYNOPSIS
Create a desktop shortcut that runs RustPainter straight from this source tree.

Unlike install.ps1, nothing is built or copied: the shortcut launches main.py
with pythonw.exe, so code changes take effect the next time you open it.

.PARAMETER PythonwPath
Path to pythonw.exe. Defaults to the first one found on PATH.

.EXAMPLE
.\install-dev.ps1

.EXAMPLE
.\install-dev.ps1 -PythonwPath C:\Python314\pythonw.exe
#>
[CmdletBinding()]
param(
    [string] $PythonwPath
)

$ErrorActionPreference = "Stop"

if ($PythonwPath) {
    if (-not (Test-Path -LiteralPath $PythonwPath)) {
        throw "No pythonw.exe at $PythonwPath"
    }
    $interpreter = (Resolve-Path -LiteralPath $PythonwPath).Path
} else {
    $found = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $found) {
        throw "pythonw.exe is not on PATH. Pass one with -PythonwPath."
    }
    $interpreter = $found.Source
}

$entryScript = Join-Path $PSScriptRoot "main.py"
if (-not (Test-Path -LiteralPath $entryScript)) {
    throw "No main.py next to this script"
}

$iconSource = Join-Path $PSScriptRoot "RustPainterIcon.png"
$iconDirectory = Join-Path $env:LOCALAPPDATA "RustPainter"
$icon = Join-Path $iconDirectory "RustPainterIcon.ico"
if (Test-Path -LiteralPath $iconSource) {
    New-Item -ItemType Directory -Path $iconDirectory -Force | Out-Null
    $iconScript = "from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2], format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])"
    $iconInterpreter = Join-Path (Split-Path -Parent $interpreter) "python.exe"
    if (-not (Test-Path -LiteralPath $iconInterpreter)) {
        $iconInterpreter = $interpreter
    }
    & $iconInterpreter -c $iconScript $iconSource $icon
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the shortcut icon from $iconSource"
    }
}

$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "RustPainter (Dev).lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $interpreter
$shortcut.Arguments = '"{0}"' -f $entryScript
$shortcut.WorkingDirectory = $PSScriptRoot
if (Test-Path -LiteralPath $icon) {
    $shortcut.IconLocation = "$icon,0"
}
$shortcut.Description = "RustPainter (runs from source)"
$shortcut.Save()

Write-Host "Created $shortcutPath"
Write-Host "Launches: $interpreter `"$entryScript`""
