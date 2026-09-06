<#
.SYNOPSIS
    一键搭好这个项目的运行环境（Windows）。

.DESCRIPTION
    建虚拟环境 -> 装对版本的 PyTorch -> 装其余依赖 -> 跑单测验证。

    装 PyTorch 有三个坑，本脚本全部处理掉（都是实际踩过的）：

    1. PyPI 上 Windows 版的 torch 是 CPU 构建。只有 116MB，装完
       torch.cuda.is_available() 返回 False。CUDA 版必须走 pytorch.org
       的独立索引，2.6GB。
    2. 官方索引在国内经常断。实测连断三次，每次停在 1.1GB 左右，
       而且 pip 自己的重试会丢弃已下载字节从头再来。所以失败后本脚本
       改用阿里云镜像 + curl 断点续传直接下 wheel 文件。
    3. 磁盘空间。2.6GB 的 wheel 加解压要约 6GB 临时空间，pip 默认写系统盘。
       盘满时报的错是 "connection interrupted"，会把人引到网络问题上去。

    注意：本文件必须保存为 **UTF-8 with BOM**。Windows PowerShell 5.1
    读 .ps1 时若没有 BOM 会按系统 ANSI 码页解释，中文会变乱码并且解析失败。

.PARAMETER Cpu
    强制装 CPU 版（只跑单测 / 看回放 / 画图；训练会慢约 20 倍）。

.PARAMETER Mirror
    直接用国内镜像，跳过官方源。

.PARAMETER VenvDir
    虚拟环境目录，默认 .venv

.PARAMETER SkipTest
    跳过最后的单测验证。

.EXAMPLE
    .\setup.ps1
    自动检测 GPU 并安装。

.EXAMPLE
    .\setup.ps1 -Mirror
    国内网络直接走镜像，省得先失败一轮。

.EXAMPLE
    .\setup.ps1 -Cpu
    只装 CPU 版。
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$Mirror,
    [string]$VenvDir = ".venv",
    [switch]$SkipTest
)

# 刻意用 Continue 而不是 Stop：Windows PowerShell 5.1 会把原生程序写到 stderr
# 的**任何**内容包成 ErrorRecord，Stop 之下就成了终止性错误。而 pygame 启动时
# 必然打一行 libpng 警告、pip 也常往 stderr 写进度 —— 装得好好的却会中途炸掉。
# 本脚本在每次调用原生程序后都显式检查 $LASTEXITCODE，不依赖这个设置。
$ErrorActionPreference = "Continue"

# CUDA 版本：cu126 是本项目实测通过的组合
$CudaTag      = "cu126"
$TorchVersion = "2.13.0"
$OfficialIdx  = "https://download.pytorch.org/whl/$CudaTag"
$OfficialCpu  = "https://download.pytorch.org/whl/cpu"
$AliyunBase   = "https://mirrors.aliyun.com/pytorch-wheels/$CudaTag"
$PypiCn       = "https://pypi.tuna.tsinghua.edu.cn/simple"
$Deps         = @("numpy", "opencv-python", "pygame", "pandas", "matplotlib", "pytest")

function Say([string]$m)  { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "[setup] $m" -ForegroundColor Yellow }
function Die([string]$m)  { Write-Host "[setup] $m" -ForegroundColor Red; exit 1 }

function Get-FreeGB([string]$path) {
    try {
        $d = (Get-Item $path).PSDrive.Name
        return (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${d}:'").FreeSpace / 1GB
    } catch { return 999 }
}

# ---------------------------------------------------------------- Python
$pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pyExe) {
    Die "找不到 python。请装 Python 3.9+ 并勾选 Add to PATH: https://www.python.org/downloads/"
}
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
$parts = $ver.Split('.')
$maj = [int]$parts[0]
$min = [int]$parts[1]
if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 9)) {
    Die "需要 Python 3.9+，当前是 $ver"
}
Say "python $ver  ($pyExe)"

# ---------------------------------------------------------------- GPU 检测
$flavour = "cpu"
if ($Cpu) {
    Say "按 -Cpu 参数强制安装 CPU 版"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        $flavour = "gpu"
        Say "检测到 GPU: $gpu"
        Say "将安装 CUDA 版 torch（约 2.6 GB，请耐心）"
    }
}
if ($flavour -eq "cpu" -and -not $Cpu) {
    Warn "没检测到 NVIDIA GPU，安装 CPU 版"
    Warn "  CPU 训练慢约 20 倍，train.py 会拒绝启动，除非显式加 --allow-cpu"
}

# ---------------------------------------------------------------- 虚拟环境
$venvPath = Join-Path (Get-Location) $VenvDir
$venvPy   = Join-Path $venvPath "Scripts\python.exe"
if (Test-Path $venvPy) {
    Say "复用已有虚拟环境 $VenvDir"
} else {
    Say "创建虚拟环境 $VenvDir"
    & python -m venv $venvPath
    if (-not (Test-Path $venvPy)) { Die "虚拟环境创建失败" }
}

# ---------------------------------------------------------------- 临时目录
# CUDA wheel 解压要约 6GB；系统盘不够就挪到项目所在盘
$savedTmp  = $env:TMP
$savedTemp = $env:TEMP
if ($flavour -eq "gpu") {
    $sysTmpFree = Get-FreeGB $env:TEMP
    if ($sysTmpFree -lt 6) {
        $projFree = Get-FreeGB (Get-Location).Path
        Warn ("系统临时目录只剩 {0:N1} GB，需要约 6 GB" -f $sysTmpFree)
        if ($projFree -lt 6) {
            $msg = "项目所在盘也只剩 {0:N1} GB。请腾出约 6 GB 再试。" -f $projFree
            Warn "注意：磁盘满时 pip 报的错是 connection interrupted，看起来像网络问题。"
            Die $msg
        }
        $altTmp = Join-Path $venvPath "_pip_tmp"
        New-Item -ItemType Directory -Force $altTmp | Out-Null
        $env:TMP  = $altTmp
        $env:TEMP = $altTmp
        Say "把 pip 的临时目录改到 $altTmp"
    }
}

try {
    Say "升级 pip"
    & $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check

    # ------------------------------------------------------------ torch
    $torchOk = $false
    if ($flavour -eq "cpu") {
        $idx = if ($Mirror) { $PypiCn } else { $OfficialCpu }
        Say "安装 CPU 版 torch，源: $idx"
        & $venvPy -m pip install torch --index-url $idx --disable-pip-version-check
        $torchOk = ($LASTEXITCODE -eq 0)
    } else {
        if (-not $Mirror) {
            Say "安装 CUDA 版 torch（官方源，约 2.6 GB）"
            & $venvPy -m pip install "torch==$TorchVersion" --index-url $OfficialIdx --disable-pip-version-check
            $torchOk = ($LASTEXITCODE -eq 0)
            if (-not $torchOk) { Warn "官方源失败，改用阿里云镜像 + 断点续传" }
        }
        if (-not $torchOk) {
            # 阿里云那个地址是文件列表不是 pip 索引，所以直接下 wheel 再本地安装。
            # curl 的内部 --retry 在断线时会丢弃已下载字节重来，所以这里不用它，
            # 由外层循环配 -C - 做真正的断点续传。
            $pyTag = "cp$maj$min"
            $whl   = "torch-$TorchVersion+$CudaTag-$pyTag-$pyTag-win_amd64.whl"
            $url   = "$AliyunBase/" + [uri]::EscapeDataString($whl)
            $dl    = Join-Path $venvPath "_wheels"
            New-Item -ItemType Directory -Force $dl | Out-Null
            $out = Join-Path $dl $whl

            $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
            if (-not $curl) { Die "需要 curl.exe（Windows 10 1803+ 自带）来做断点续传下载" }

            Say "从阿里云下载 $whl"
            $ok = $false
            for ($i = 1; $i -le 40; $i++) {
                & $curl -L --fail -C - --connect-timeout 30 --speed-time 45 --speed-limit 51200 -o $out $url
                if ($LASTEXITCODE -eq 0) { $ok = $true; break }
                if ($LASTEXITCODE -eq 33) {
                    Remove-Item $out -Force -ErrorAction SilentlyContinue
                    continue
                }
                $mb = 0
                if (Test-Path $out) { $mb = (Get-Item $out).Length / 1MB }
                Warn ("第 {0} 次中断，已下载 {1:N0} MB，续传中" -f $i, $mb)
            }
            if (-not $ok) {
                Warn "下载失败。可以手动下载后本地安装："
                Warn "  $url"
                Die  "  然后跑: $venvPy -m pip install <wheel 文件路径>"
            }
            Say ("下载完成 {0:N0} MB，安装中" -f ((Get-Item $out).Length / 1MB))
            & $venvPy -m pip install $out --disable-pip-version-check
            $torchOk = ($LASTEXITCODE -eq 0)
            if ($torchOk) { Remove-Item $dl -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }
    if (-not $torchOk) { Die "torch 安装失败。国内网络建议: .\setup.ps1 -Mirror" }

    # ------------------------------------------------------------ 其余依赖
    Say "安装其余依赖: $($Deps -join ', ')"
    if ($Mirror) {
        & $venvPy -m pip install @Deps --index-url $PypiCn --disable-pip-version-check
    } else {
        & $venvPy -m pip install @Deps --disable-pip-version-check
    }
    if ($LASTEXITCODE -ne 0) { Die "依赖安装失败" }
}
finally {
    $env:TMP  = $savedTmp
    $env:TEMP = $savedTemp
}

# ---------------------------------------------------------------- 验证
Say "验证安装"
# 多行脚本要写成临时文件再跑。PowerShell 把多行字符串传给原生程序的 -c 时
# 会按换行拆成多个参数，python 只收到第一行。
$check = @'
import torch, numpy, cv2, pygame
print("  torch  ", torch.__version__)
print("  numpy  ", numpy.__version__)
print("  opencv ", cv2.__version__)
print("  cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device :", torch.cuda.get_device_name(0))
'@
$checkFile = Join-Path ([IO.Path]::GetTempPath()) "flappy_setup_check.py"
Set-Content -Path $checkFile -Value $check -Encoding UTF8
try {
    & $venvPy $checkFile
    if ($LASTEXITCODE -ne 0) { Die "导入失败，环境不可用" }
} finally {
    Remove-Item $checkFile -Force -ErrorAction SilentlyContinue
}

if ($flavour -eq "gpu") {
    & $venvPy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Warn "要装 CUDA 版，但 torch.cuda.is_available() 是 False。"
        Warn "  多半是装成了 CPU wheel（PyPI 上 Windows 版 torch 就是 CPU 的）。"
        Warn "  重跑一次: .\setup.ps1 -Mirror"
    }
}

if (-not $SkipTest) {
    Say "跑单测验证（约 30 秒）"
    $env:PYTHONIOENCODING = "utf-8"
    & $venvPy test\test_env_and_buffer.py
    if ($LASTEXITCODE -ne 0) { Die "单测失败，环境有问题" }
}

Write-Host ""
Say "完成。常用命令："
Write-Host ""
Write-Host "  .\$VenvDir\Scripts\python.exe train.py --smoke      # 几分钟的管线自检" -ForegroundColor Green
Write-Host "  .\$VenvDir\Scripts\python.exe train.py              # 正式训练" -ForegroundColor Green
Write-Host "  .\$VenvDir\Scripts\python.exe monitor.py            # 另开终端实时监控" -ForegroundColor Green
Write-Host "  .\$VenvDir\Scripts\python.exe play.py --human       # 自己玩" -ForegroundColor Green
Write-Host ""
Write-Host "  或者先激活环境:  .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor DarkGray
Write-Host ""
if ($flavour -eq "gpu") {
    Write-Host "  内存提示: 默认 buffer 要 11.9 GB 常驻内存，16 GB 的机器用 --buffer 100000" -ForegroundColor DarkGray
}
