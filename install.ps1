<#
.SYNOPSIS
Install RustPainter for the current user and create shortcuts.

.PARAMETER ExecutablePath
Path to RustPainter.exe. Defaults to the local build output produced by
build.ps1; pass a downloaded release executable to install that one instead.

.EXAMPLE
.\install.ps1
Installs the executable built by .\build.ps1.

.EXAMPLE
.\install.ps1 -ExecutablePath $env:USERPROFILE\Downloads\RustPainter.exe
Installs a downloaded release executable.
#>
[CmdletBinding()]
param(
    [string] $ExecutablePath
)

$ErrorActionPreference = "Stop"

if ($ExecutablePath) {
    if (-not (Test-Path -LiteralPath $ExecutablePath)) {
        throw "No executable at $ExecutablePath"
    }
    $sourceExecutable = (Resolve-Path -LiteralPath $ExecutablePath).Path
} else {
    $sourceExecutable = Join-Path $PSScriptRoot "dist\RustPainter.exe"
    if (-not (Test-Path -LiteralPath $sourceExecutable)) {
        throw "No executable found. Run .\build.ps1 first, or pass a downloaded release with -ExecutablePath."
    }
}

$installDirectory = Join-Path $env:LOCALAPPDATA "Programs\RustPainter"
$installedExecutable = Join-Path $installDirectory "RustPainter.exe"
New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
Copy-Item -LiteralPath $sourceExecutable -Destination $installedExecutable -Force

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

foreach ($shortcutPath in @(
    (Join-Path $desktop "RustPainter.lnk"),
    (Join-Path $startMenu "RustPainter.lnk")
)) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $installedExecutable
    $shortcut.WorkingDirectory = $installDirectory
    $shortcut.IconLocation = "$installedExecutable,0"
    $shortcut.Description = "RustPainter"
    $shortcut.Save()
}

Write-Host "Installed RustPainter to $installedExecutable"
Write-Host "Created Desktop and Start menu shortcuts named RustPainter"
