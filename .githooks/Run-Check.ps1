param([Parameter(Mandatory=$true)][string]$RepoPath)
$ErrorActionPreference = 'Stop'
$hub = if ($env:OSS_GOVERNANCE_ROOT) { $env:OSS_GOVERNANCE_ROOT } else { Join-Path $env:USERPROFILE 'OneDrive\analytics\my_github\ArturSepp\scripts\repo_governance' }
& (Join-Path $hub 'Invoke-CommitCheck.ps1') -RepoPath $RepoPath -Task preflight
exit $LASTEXITCODE
