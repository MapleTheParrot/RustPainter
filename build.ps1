$ErrorActionPreference = "Stop"

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name RustPainter `
    --icon RustPainterIcon.ico `
    --add-data "RustPainterIcon.png;." `
    --add-data "assets/ui;assets/ui" `
    main.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$executable = Join-Path $PSScriptRoot "dist\RustPainter.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "PyInstaller completed without creating $executable"
}

Write-Host "Built dist\RustPainter.exe"

$releaseDirectory = Join-Path $PSScriptRoot "release"
New-Item -ItemType Directory -Path $releaseDirectory -Force | Out-Null
$archive = Join-Path $releaseDirectory "RustPainter-Windows-x64.zip"
Compress-Archive -LiteralPath $executable -DestinationPath $archive -Force

Write-Host "Packaged release\RustPainter-Windows-x64.zip"
