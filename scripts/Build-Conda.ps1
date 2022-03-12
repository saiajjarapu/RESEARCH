param(
    [ValidateSet("vision-damage-classification")]
    [string]
    $EnvName,
    [switch]
    $Clean,
    [switch]
    $Build,
    [switch]
    $Verbose
)

function PrintAndExecuteCommand {
    param ([string]$commandString)
    Write-Host "[ ${commandString} ]"
    & $([ScriptBlock]::Create($commandString))
}

function PrepareEnvironment {
    PrintAndExecuteCommand("conda deactivate")
    PrintAndExecuteCommand("conda activate base")
}

function CleanEnvironment {
    PrintAndExecuteCommand("conda env remove -n ${EnvName}")
    if (Test-Path -Path $envPath) {
        PrintAndExecuteCommand("Remove-Item -Recurse -Force -Path ${envPath}") 
    }
}

function BuildEnvironment {
    PrintAndExecuteCommand("conda env create -f .\conda.yaml ${verboseFlag}")
    PrintAndExecuteCommand("conda activate ${EnvName}")
}

if ([String]::IsNullOrEmpty($EnvName)) { Throw "-EnvName must be given" }
if ($Clean -or $Build) { PrepareEnvironment } else { Throw "Choose one or both of -Clean and -Build" }

$envPath = "${env:CONDA_PREFIX}\envs\${EnvName}"
$verboseFlag = if ($Verbose) { "-v" } else { "" }

if ($Clean) { CleanEnvironment }
if ($Build) { BuildEnvironment }