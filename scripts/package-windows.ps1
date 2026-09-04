$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repositoryRoot

if (-not $IsWindows) {
    throw "Windows packaging must run on Windows."
}

$architecture = if ($env:SAGE_WINDOWS_ARCH) { $env:SAGE_WINDOWS_ARCH.ToLowerInvariant() } else { "x64" }
if ($architecture -ne "x64") {
    throw "The preview installer is x64 only. Use the CI build job for other architecture checks."
}

$runtime = "win-x64"
$output = Join-Path $repositoryRoot "dist/windows/$runtime"
$installer = Join-Path $repositoryRoot "dist/windows/Sage-1.0.1-windows-x64-preview.exe"

if (Test-Path $output) { Remove-Item -Recurse -Force $output }
New-Item -ItemType Directory -Force -Path $output | Out-Null

cargo build --release --workspace --target x86_64-pc-windows-msvc
if ($LASTEXITCODE -ne 0) { throw "Rust Windows build failed with exit code $LASTEXITCODE." }
dotnet publish apps/windows/Sage.Windows/Sage.Windows.csproj `
    -c Release `
    -r $runtime `
    --self-contained true `
    -p:Platform=x64 `
    -o $output
if ($LASTEXITCODE -ne 0) { throw ".NET Windows publish failed with exit code $LASTEXITCODE." }

$rustOutput = Join-Path $repositoryRoot "target/x86_64-pc-windows-msvc/release"
Copy-Item "$rustOutput/sage-core.exe" $output
Copy-Item "$rustOutput/sage-browser-worker.exe" $output
Copy-Item "$rustOutput/sage-sandbox-worker.exe" $output
Copy-Item "$rustOutput/sage-privileged-helper.exe" $output

$iscc = (Get-Command ISCC.exe -ErrorAction Stop).Source
if (Test-Path $installer) { Remove-Item -Force $installer }
& $iscc "/DSourceDir=$output" "/DOutputDir=$(Join-Path $repositoryRoot 'dist/windows')" "/DOutputBaseFilename=Sage-1.0.1-windows-x64-preview" (Join-Path $repositoryRoot "scripts/Sage.iss")
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $installer)) {
    throw "Inno Setup did not produce the expected unsigned preview installer."
}
Write-Output $installer
