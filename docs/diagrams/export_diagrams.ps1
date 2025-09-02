# Requires: Node.js and mermaid-cli (mmdc)
# Install: npm install -g @mermaid-js/mermaid-cli

param(
  [string]$InputDir = "$PSScriptRoot",
  [string]$OutDir = "$PSScriptRoot/out",
  [string]$Format = "png"  # png | svg | pdf
)

if (!(Get-Command mmdc -ErrorAction SilentlyContinue)) {
  Write-Host "Mermaid CLI (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli" -ForegroundColor Yellow
  exit 1
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$files = Get-ChildItem -Path $InputDir -Filter *.mmd -File
foreach ($f in $files) {
  $outFile = Join-Path $OutDir ("{0}.{1}" -f $f.BaseName, $Format)
  Write-Host "Exporting $($f.Name) -> $outFile"
  mmdc -i $f.FullName -o $outFile
}

Write-Host "Done. Outputs in: $OutDir" -ForegroundColor Green