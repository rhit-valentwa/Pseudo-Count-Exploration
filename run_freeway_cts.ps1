# PowerShell wrapper to run the specified atari_dqn.py command
# Usage: .\run_freeway_cts.ps1

$python = "python"
$script = "atari_dqn.py"

$args = @(
    "--env", "ALE/Freeway-v5",
    "--density", "pixelcnn",
    "--beta", "0.05",
    "--n-step", "1",
    "--train-start", "1000",
    "--epsilon-decay-steps", "100000",
    "--epsilon-end", "0.01",
    "--target-update-freq", "5000",
    "--sticky-action-prob", "0.25",
    "--no-compile",
    "--log-freq", "1000",
    "--checkpoint-freq", "50000",
    "--log-dir", "logs",
    "--run-name", "freeway_pixelcnn_nstep1_seed9911",
    "--steps", "500000",
    "--seed", "9911"
)

Write-Host "Running: $python $script $($args -join ' ')" -ForegroundColor Cyan

& $python $script @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "Command exited with code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "Command completed successfully" -ForegroundColor Green
}
