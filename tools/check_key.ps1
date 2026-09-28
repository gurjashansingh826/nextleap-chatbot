# Diagnostic: prove exactly which .env is being read, and whether the key is in it.
# Works from ANY directory - it locates the project itself, so no cd needed.
#   powershell -ExecutionPolicy Bypass -File "C:\Users\Gurjashan singh\New folder\tools\check_key.ps1"

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

# tools\ -> project root.  No recursion, no profile scan: this runs in under a second.
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root '.env'
$py = Join-Path $root '.venv\Scripts\python.exe'

Write-Host "`n=== 1. the .env files that exist (no deep scan) ===" -ForegroundColor Cyan
foreach ($p in @($envFile, (Join-Path $root '.env.example'), (Join-Path $env:USERPROFILE 'OneDrive\Desktop\.env'))) {
    if (Test-Path -LiteralPath $p) {
        $f = Get-Item -LiteralPath $p
        Write-Host ("  {0,-62} {1,6} bytes  {2}" -f $f.FullName, $f.Length, $f.LastWriteTime)
    } else {
        Write-Host ("  {0,-62} (does not exist)" -f $p) -ForegroundColor DarkGray
    }
}

Write-Host "`n=== 2. line 7 of the .env the app reads (value masked) ===" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Host "  !! $envFile does not exist" -ForegroundColor Red
} else {
    $line7 = (Get-Content -LiteralPath $envFile -Encoding UTF8)[6]
    if ($line7 -match '^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$') {
        $var, $val = $Matches[1], $Matches[2]
        if ([string]::IsNullOrWhiteSpace($val)) {
            Write-Host "  $var  ->  EMPTY   <-- the key is NOT in this file" -ForegroundColor Yellow
        } else {
            Write-Host "  $var  ->  SET ($($val.Length) chars)" -ForegroundColor Green
        }
    } else {
        Write-Host "  line 7 is not an assignment: $line7" -ForegroundColor Yellow
    }
}

Write-Host "`n=== 3. what the app's own loader sees ===" -ForegroundColor Cyan
& $py -c "import sys; sys.stdout.reconfigure(encoding='utf-8'); from mf_rag.config import settings, ENV_PATH, KEY_LINE_NUMBER; k=settings.groq_api_key; print('  file read :', ENV_PATH); print('  edit line :', KEY_LINE_NUMBER); print('  key       :', ('SET len=%d' % len(k)) if k else 'NOT SET'); print('  model     :', settings.llm_model)"

Write-Host "`n=== 4. if step 2 says EMPTY ===" -ForegroundColor Cyan
Write-Host "  paste the key after the = on line 7 of:"
Write-Host "    $envFile"
Write-Host "  or run:"
Write-Host "    `"$py`" `"$root\tools\set_groq_key.py`""
Write-Host ""
