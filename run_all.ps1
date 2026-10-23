# ELEC 475 Lab 2 - Runs every post-lab experiment in order (Windows PowerShell).
# Usage (from lab2_template):  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Safe to re-run: any run whose log already contains "Training completed" is skipped.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs, checkpoints, plots | Out-Null

# Keep Windows awake while this script runs (released automatically when it exits).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$runs = @(
  @("alexnet_none",          "--model alexnet --norm none --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_none.pth --plot-file plots/loss_alexnet_none.png"),
  @("alexnet_lrn",           "--model alexnet --norm lrn --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_lrn.pth --plot-file plots/loss_alexnet_lrn.png"),
  @("alexnet_bn",            "--model alexnet --norm bn --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_bn.pth --plot-file plots/loss_alexnet_bn.png"),
  @("alexnet_no_aug",        "--model alexnet --norm bn --dataset cifar10 --epochs 15 --no-augment --save-model checkpoints/alexnet_no_aug.pth --plot-file plots/loss_no_aug.png"),
  @("alexnet_with_aug",      "--model alexnet --norm bn --dataset cifar10 --epochs 15 --save-model checkpoints/alexnet_with_aug.pth --plot-file plots/loss_with_aug.png"),
  @("alexnet_tiny_imagenet", "--model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --save-model checkpoints/alexnet_tiny_imagenet.pth --plot-file plots/loss_tiny_imagenet.png"),
  @("zoo_alexnet",           "--model zoo_alexnet --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/zoo_alexnet.pth --plot-file plots/loss_zoo_alexnet.png"),
  @("lenet_tiny",            "--model lenet --dataset tiny_imagenet --epochs 5 --no-augment --save-model checkpoints/lenet_tiny.pth --plot-file plots/loss_lenet_tiny.png"),
  @("lenet_modern_tiny",     "--model lenet_modern --dataset tiny_imagenet --epochs 5 --no-augment --save-model checkpoints/lenet_modern_tiny.pth --plot-file plots/loss_lenet_modern_tiny.png"),
  @("lenet_seed1",           "--model lenet --dataset cifar10 --epochs 5 --seed 1 --no-augment --plot-file plots/loss_lenet_seed1.png"),
  @("lenet_seed2",           "--model lenet --dataset cifar10 --epochs 5 --seed 2 --no-augment --plot-file plots/loss_lenet_seed2.png"),
  @("lenet_modern_seed1",    "--model lenet_modern --dataset cifar10 --epochs 5 --seed 1 --no-augment --plot-file plots/loss_lenet_modern_seed1.png"),
  @("lenet_modern_seed2",    "--model lenet_modern --dataset cifar10 --epochs 5 --seed 2 --no-augment --plot-file plots/loss_lenet_modern_seed2.png")
)

foreach ($r in $runs) {
  $name = $r[0]; $log = "logs\$name.txt"
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Training completed" -Quiet)) {
    Write-Host "[skip] $name (already finished)"; continue
  }
  Write-Host "[$(Get-Date -Format 'HH:mm')] start $name"
  cmd /c "$py train.py $($r[1]) > $log 2>&1"
  Write-Host "[$(Get-Date -Format 'HH:mm')] end   $name (exit $LASTEXITCODE)"
}

Write-Host "[$(Get-Date -Format 'HH:mm')] profiling, demo and summary"
cmd /c "$py compare_models.py --dataset cifar10 > logs\compare_models.txt 2>&1"
cmd /c "$py demo_transforms.py > logs\demo_transforms.txt 2>&1"
cmd /c "$py make_ablation_plot.py > logs\summary.txt 2>&1"
Write-Host "[$(Get-Date -Format 'HH:mm')] ALL DONE"
