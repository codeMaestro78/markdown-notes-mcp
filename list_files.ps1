# List all files recursively
Get-ChildItem -Path . -Recurse -File | Select-Object FullName

# List files with sizes
Get-ChildItem -Path . -Recurse -File | Select-Object FullName, Length | Sort-Object Length -Descending

# List by file type
Get-ChildItem -Path . -Recurse -File | Group-Object Extension | Sort-Object Count -Descending

# List hidden files
Get-ChildItem -Path . -Recurse -File -Hidden

# List large files (>1MB)
Get-ChildItem -Path . -Recurse -File | Where-Object { $_.Length -gt 1MB } | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}

# List recent files (modified in last 7 days)
Get-ChildItem -Path . -Recurse -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } | Select-Object FullName, LastWriteTime