Set-Location -LiteralPath $PSScriptRoot
python comparar.py
if ($LASTEXITCODE -eq 0) { Start-Process (Join-Path $PSScriptRoot 'resultados/comparador.html') }
