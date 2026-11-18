# Gives the running API a public https link (Cloudflare quick tunnel, free, no account).
# Anyone with the link can use it while this window is open. Close the window to stop it.
# Strongly recommended: set an API key before starting the API (see start_api.ps1).
$exe = Join-Path $PSScriptRoot "cloudflared.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Downloading cloudflared (official Cloudflare release)..."
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $exe
}
& $exe tunnel --url http://localhost:8010
