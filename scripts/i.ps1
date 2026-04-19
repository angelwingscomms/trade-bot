param(
  [Alias("i")]
  [int]$Days = 3
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir

$pythonBin = "python"
if (Test-Path "$RootDir\env\Scripts\python.exe") {
  $pythonBin = "$RootDir\env\Scripts\python.exe"
} elseif (Test-Path "$RootDir\env\bin\python") {
  $pythonBin = "$RootDir\env\bin\python"
}

$trainOutput = & $pythonBin "$ScriptDir\nn.py"
$trainOutput | ForEach-Object { $_ }
$modelFolder = ($trainOutput | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1)
if (-not $modelFolder) {
  throw "nn.py did not print a model folder name."
}

$dateRange = & $pythonBin -c @"
from datetime import date, timedelta
days = $Days
end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=days - 1)
print(start_date.isoformat(), end_date.isoformat())
"@
$parts = $dateRange -split "\s+"
$fromDate = $parts[0]
$toDate = $parts[1]

& $pythonBin "$ScriptDir\test.py" --revision $modelFolder --from-date $fromDate --to-date $toDate
