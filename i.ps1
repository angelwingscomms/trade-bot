param(
  [Alias("i")]
  [int]$Days = 3
)

$pythonBin = "python"
if (Test-Path ".\env\Scripts\python.exe") {
  $pythonBin = ".\env\Scripts\python.exe"
} elseif (Test-Path ".\env\bin\python") {
  $pythonBin = ".\env\bin\python"
}

$trainOutput = & $pythonBin nn.py
$trainOutput | ForEach-Object { $_ }
$modelFolder = ($trainOutput | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1)
if (-not $modelFolder) {
  throw "nn.py did not print a model folder name."
}

$dateRange = & $pythonBin -c @"
from datetime import date, timedelta
days = int(${Days})
end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=days - 1)
print(start_date.isoformat(), end_date.isoformat())
"@
$parts = $dateRange -split "\s+"
$fromDate = $parts[0]
$toDate = $parts[1]

& $pythonBin test.py --revision $modelFolder --from-date $fromDate --to-date $toDate
