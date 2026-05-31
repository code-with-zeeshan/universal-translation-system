[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Version
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Preparing release $Version..."

# Update versions
python "$repoRoot\scripts\version_manager.py" release $Version
if ($LASTEXITCODE -ne 0) { throw "Version update failed" }

# Run tests
Write-Host "Running tests..."
pytest "$repoRoot\tests"
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

# Build Android SDK
$androidDir = "$repoRoot\sdk\android\UniversalTranslationSDK"
if (Test-Path $androidDir) {
  Write-Host "Building Android SDK..."
  Set-Location $androidDir
  if (Test-Path "$androidDir\gradlew.bat") {
    ./gradlew.bat clean build
    if ($LASTEXITCODE -ne 0) { throw "Android build failed" }
  } else {
    Write-Warning "gradlew.bat not found; skipping Android build."
  }
  Set-Location $repoRoot
}

# Validate iOS SDK (requires Xcode/macOS)
Write-Host "Validating iOS SDK..."
$iosDir = "$repoRoot\sdk\ios\UniversalTranslationSDK"
if (Test-Path $iosDir) {
    try {
        Set-Location $iosDir
        swift --version > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            swift build
            if ($LASTEXITCODE -ne 0) { throw "iOS build failed" }
        } else {
            Write-Warning "swift not available; skipping iOS build validation."
        }
    } catch {
        Write-Warning "Skipping iOS validation: $_"
    } finally {
        Set-Location $repoRoot
    }
}

# Reminder
Write-Host "Don't forget to update CHANGELOG.md!"

# Create tag
Write-Host "Creating git tag..."

git add -A
if ((git diff --cached --quiet) -eq $false) {
  git commit -m "chore: release v$Version"
} else {
  Write-Host "No changes to commit."
}

git tag -a "v$Version" -m "Release version $Version"

Write-Host "Release prepared!"
Write-Host "Run 'git push && git push --tags' to publish"
