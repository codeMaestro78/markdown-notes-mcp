# Clean unnecessary files from markdown-notes-mcp project
# Run this script from the project root directory

Write-Host "🧹 Starting cleanup of unnecessary files..." -ForegroundColor Green
Write-Host "⚠️  Make sure you're in the correct directory!" -ForegroundColor Yellow

# Confirm current directory
$currentDir = Get-Location
Write-Host "Current directory: $currentDir" -ForegroundColor Cyan

if ((Read-Host "Continue with cleanup? (y/N)") -ne 'y') {
    Write-Host "Cleanup cancelled." -ForegroundColor Red
    exit
}

# Remove Python cache files
Write-Host "Removing Python cache files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | ForEach-Object {
    $path = Join-Path $currentDir $_
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "Removed: $_" -ForegroundColor Red
    }
}

# Remove compiled Python files
Write-Host "Removing compiled Python files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include "*.pyc", "*.pyo", "*.pyd" | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "Removed: $($_.Name)" -ForegroundColor Red
}

# Remove log files
Write-Host "Removing log files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include "*.log" | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "Removed: $($_.Name)" -ForegroundColor Red
}

# Remove OS-specific files
Write-Host "Removing OS-specific files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include ".DS_Store", "Thumbs.db", "Desktop.ini", ".AppleDouble", ".LSOverride" | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "Removed: $($_.Name)" -ForegroundColor Red
}

# Remove temporary files
Write-Host "Removing temporary files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Include "*.tmp", "*.bak", "*.orig", "*.rej", "*~" | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Host "Removed: $($_.Name)" -ForegroundColor Red
}

# Remove IDE files (optional - comment out if you want to keep)
Write-Host "Removing IDE files..." -ForegroundColor Yellow
$ideFiles = @(".vscode", ".idea", "*.swp", "*.swo")
foreach ($pattern in $ideFiles) {
    Get-ChildItem -Path . -Recurse -Include $pattern | ForEach-Object {
        Remove-Item $_.FullName -Recurse -Force
        Write-Host "Removed: $($_.Name)" -ForegroundColor Red
    }
}

# Remove node_modules if any (from potential frontend additions)
Write-Host "Removing node_modules..." -ForegroundColor Yellow
$nodeModules = Join-Path $currentDir "node_modules"
if (Test-Path $nodeModules) {
    Remove-Item $nodeModules -Recurse -Force
    Write-Host "Removed: node_modules" -ForegroundColor Red
}

# Remove build artifacts
Write-Host "Removing build artifacts..." -ForegroundColor Yellow
$buildDirs = @("build", "dist", "*.egg-info", ".pytest_cache", ".mypy_cache", "htmlcov")
foreach ($dir in $buildDirs) {
    Get-ChildItem -Path . -Recurse -Directory -Name $dir | ForEach-Object {
        $path = Join-Path $currentDir $_
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force
            Write-Host "Removed: $_" -ForegroundColor Red
        }
    }
}

Write-Host "`n✅ Cleanup completed!" -ForegroundColor Green
Write-Host "📊 Summary of removed files:" -ForegroundColor Cyan

# Show remaining files count
$remainingFiles = Get-ChildItem -Path . -Recurse -File | Measure-Object | Select-Object -ExpandProperty Count
$remainingDirs = Get-ChildItem -Path . -Recurse -Directory | Measure-Object | Select-Object -ExpandProperty Count

Write-Host "Remaining files: $remainingFiles" -ForegroundColor White
Write-Host "Remaining directories: $remainingDirs" -ForegroundColor White

Write-Host "`n💡 Next steps:" -ForegroundColor Yellow
Write-Host "1. Review the changes: git status" -ForegroundColor White
Write-Host "2. Add to git: git add ." -ForegroundColor White
Write-Host "3. Commit: git commit -m 'Clean up unnecessary files'" -ForegroundColor White
Write-Host "4. Push: git push origin main" -ForegroundColor White