$ErrorActionPreference = "Stop"

$sourceExecutable = Join-Path $PSScriptRoot "dist\RustPainter.exe"
if (-not (Test-Path -LiteralPath $sourceExecutable)) {
    throw "Build RustPainter first by running .\build.ps1"
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
