[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [ValidateSet("Auto", "USB", "GigE", "Both")]
    [string]$Interface = "Auto"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsDirectory = Join-Path $projectRoot ".tools\basler"
$runtimePath = Join-Path $toolsDirectory "pylon-runtime-8.1.0.exe"
$downloadPath = Join-Path $toolsDirectory "pylon-runtime-8.1.0.download.exe"
$runtimeUrl = "https://downloads-bsl.baslerweb.com/software/pylon%20Runtime%208.1.0.exe"
$runtimeSize = 1336906056L
$rebootCodes = @(1641, 3010)

function Get-BaslerDriverState {
    $inventory = & "$env:SystemRoot\System32\pnputil.exe" /enum-drivers 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Windows 드라이버 목록을 읽지 못했습니다. (exit $LASTEXITCODE)"
    }
    $text = $inventory -join "`n"
    return @{
        Usb = $text -match '(?im)\b(?:plnu3v|pylon\S*(?:usb|u3v)\S*)\.inf\b'
        GigE = $text -match '(?im)\b(?:pylongev|pylon\S*gige\S*)\.inf\b'
    }
}

function Test-BaslerSignature([string]$Path) {
    $signature = Get-AuthenticodeSignature -LiteralPath $Path
    if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch '\bBasler\b') {
        throw "Basler 전자서명이 유효하지 않습니다: $Path"
    }
}

function Get-BaslerInstaller {
    if ($env:BASLER_INSTALLER) {
        $installer = (Resolve-Path -LiteralPath $env:BASLER_INSTALLER).Path
        Test-BaslerSignature $installer
        return $installer
    }

    New-Item -ItemType Directory -Path $toolsDirectory -Force | Out-Null
    if (-not (Test-Path -LiteralPath $runtimePath)) {
        Write-Host "Basler pylon Runtime 8.1.0을 다운로드합니다. (약 1.25 GB)"
        & curl.exe --fail --location --progress-bar --continue-at - `
            --retry 2 --retry-all-errors --retry-delay 3 `
            --output $downloadPath $runtimeUrl
        if ($LASTEXITCODE -ne 0) {
            throw "pylon Runtime 다운로드에 실패했습니다. 네트워크를 확인한 뒤 다시 실행하세요."
        }
        if ((Get-Item -LiteralPath $downloadPath).Length -ne $runtimeSize) {
            throw "pylon Runtime 다운로드 크기가 올바르지 않습니다. 다시 실행하면 이어받습니다."
        }
        Test-BaslerSignature $downloadPath
        Move-Item -LiteralPath $downloadPath -Destination $runtimePath -Force
    }
    Test-BaslerSignature $runtimePath
    return $runtimePath
}

function Install-BaslerRuntime([string]$Installer) {
    $logPath = Join-Path $toolsDirectory "install.log"
    $features = switch ($Interface) {
        "USB" { "Cpp_Runtime;USB_Runtime;USB_Camera_Driver" }
        "GigE" { "Cpp_Runtime;GigE_Runtime;GigE_Filter_Driver" }
        default { "Cpp_Runtime;USB_Runtime;USB_Camera_Driver;GigE_Runtime;GigE_Filter_Driver" }
    }
    $arguments = @("/passive", "/norestart", "/install=$features", "/log", "`"$logPath`"")
    Write-Host "Basler USB/GigE 드라이버와 pylon Runtime을 설치합니다."
    Write-Host "관리자 권한(UAC) 창에서 설치를 승인하세요."
    $process = Start-Process -FilePath $Installer -ArgumentList $arguments `
        -Verb RunAs -WindowStyle Normal -Wait -PassThru
    if ($process.ExitCode -in $rebootCodes) {
        throw "Basler 설치 후 Windows 재시작이 필요합니다. 재시작 후 이 파일을 다시 실행하세요."
    }
    if ($process.ExitCode -ne 0) {
        throw "Basler 설치에 실패했습니다. exit=$($process.ExitCode), 로그=$logPath"
    }
}

function Ensure-BaslerRuntime {
    $state = Get-BaslerDriverState
    Write-Host "Basler 드라이버 상태: USB=$($state.Usb), GigE=$($state.GigE)"
    $driverReady = switch ($Interface) {
        "USB" { $state.Usb }
        "GigE" { $state.GigE }
        "Both" { $state.Usb -and $state.GigE }
        default { $state.Usb -or $state.GigE }
    }
    if ($driverReady) {
        Write-Host "필요한 Basler 드라이버가 이미 설치되어 있어 건너뜁니다."
        return
    }
    if ($CheckOnly) { throw "요청한 Basler $Interface 드라이버가 설치되어 있지 않습니다." }
    Install-BaslerRuntime (Get-BaslerInstaller)
    $verified = Get-BaslerDriverState
    $installed = switch ($Interface) {
        "USB" { $verified.Usb }
        "GigE" { $verified.GigE }
        "Both" { $verified.Usb -and $verified.GigE }
        default { $verified.Usb -or $verified.GigE }
    }
    if (-not $installed) {
        throw "설치 후 Basler 드라이버를 확인하지 못했습니다. Windows를 재시작한 뒤 다시 실행하세요."
    }
}

function Get-Uv {
    $localUv = Join-Path $projectRoot ".tools\uv\uv.exe"
    if (Test-Path -LiteralPath $localUv) { return $localUv }
    $installedUv = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($installedUv) { return $installedUv.Source }

    $uvDirectory = Split-Path -Parent $localUv
    $zipPath = Join-Path $uvDirectory "uv.zip"
    New-Item -ItemType Directory -Path $uvDirectory -Force | Out-Null
    Write-Host "Python 환경 구성을 위해 uv를 다운로드합니다."
    Invoke-WebRequest `
        -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" `
        -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $uvDirectory -Force
    if (-not (Test-Path -LiteralPath $localUv)) { throw "uv 설치에 실패했습니다." }
    return $localUv
}

function Ensure-PythonEnvironment {
    $uv = Get-Uv
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".uv-python"
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    Write-Host "프로젝트 Python 의존성을 설치/갱신합니다."
    & $uv sync --extra model --extra gui
    if ($LASTEXITCODE -ne 0) { throw "프로젝트 의존성 설치에 실패했습니다." }
    & $venvPython -c "from pypylon import pylon; pylon.TlFactory.GetInstance().EnumerateTls()"
    if ($LASTEXITCODE -ne 0) { throw "pypylon 또는 pylon Runtime 검증에 실패했습니다." }
}

Set-Location -LiteralPath $projectRoot
Ensure-BaslerRuntime
if (-not $CheckOnly) { Ensure-PythonEnvironment }
Write-Host "Basler 카메라 환경 구성이 완료되었습니다." -ForegroundColor Green
