[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Version
)

$ErrorActionPreference = 'Stop'
$repoRoot = "c:\Users\DELL\universal-translation-system"
Set-Location $repoRoot

Write-Host "🚀 Preparing release $Version..."

# Update versions
python "$repoRoot\scripts\version_manager.py" release $Version

# Run tests
Write-Host "🧪 Running tests..."
pytest "$repoRoot\tests"

# Build Android
Write-Host "📱 Building Android SDK..."
$androidDir = "$repoRoot\android\UniversalTranslationSDK"
Set-Location $androidDir
./gradlew.bat clean build
Set-Location $repoRoot

# Validate iOS (requires Xcode/macOS; skip if tools are missing)
Write-Host "📱 Validating iOS SDK..."
$iosDir = "$repoRoot\ios\UniversalTranslationSDK"
if (Test-Path $iosDir) {
    try {
        Set-Location $iosDir
        swift --version > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            swift build
        } else {
            Write-Warning "swift not available on this system; skipping iOS build validation."
        }
    } catch {
        Write-Warning "Skipping iOS validation: $_"
    } finally {
        Set-Location $repoRoot
    }
}

# Update changelog reminder
Write-Host "📝 Don't forget to update CHANGELOG.md!"

# Create tag
Write-Host "🏷️  Creating git tag..."

git add -A
try {
    git commit -m "chore: release v$Version"
} catch {
    Write-Warning "No changes to commit or commit failed: $_"
}

git tag -a "v$Version" -m "Release version $Version"

Write-Host "✅ Release prepared!"
Write-Host "📤 Run 'git push && git push --tags' to publish"