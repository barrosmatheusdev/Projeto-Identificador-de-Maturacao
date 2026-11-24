# Teste rapido do classificador embarcado com tomates reais, com previa ao vivo.
# Uso (feche o Monitor Serial do Arduino IDE antes):
#   powershell -ExecutionPolicy Bypass -File teste_esp.ps1 [-Porta COM5] [-Capturas 3]
# Abre uma janela com a imagem da camera (~1 quadro/s). O quadrado amarelo e o recorte que o
# modelo usa. Com a janela em foco:
#   V = classificar como verde (seu olho)   M = maduro   Espaco = classificar sem rotulo
#   Esc = sair
# Cada classificacao faz N capturas (comando "t"), vale a maioria; o fruto avanca quando ha rotulo.
# Grava o CSV e a ultima previa de cada fruto (testes\previas\fruto_NN.jpg) para conferir erros.
param([string]$Porta = "COM5", [int]$Capturas = 3,
      [string]$Saida = "$PSScriptRoot\..\testes\preliminar_etapa1.csv")

New-Item -ItemType Directory -Force (Split-Path $Saida) | Out-Null
$pastaPrevias = Join-Path (Split-Path $Saida) "previas"
New-Item -ItemType Directory -Force $pastaPrevias | Out-Null
if (-not (Test-Path $Saida)) {
  "data_hora,fruto,captura,rotulo_humano,classe,confianca,p_verde,p_maduro,t_inferencia_ms" | Out-File $Saida -Encoding utf8
}

$p = New-Object System.IO.Ports.SerialPort $Porta, 115200
$p.DtrEnable = $false; $p.RtsEnable = $false; $p.ReadTimeout = 5000
$p.Open(); Start-Sleep -Milliseconds 500; $null = $p.ReadExisting()

Add-Type -AssemblyName System.Windows.Forms, System.Drawing

function Get-Quadro {
  # Pede um quadro ("v" no firmware): resposta "IMG,<bytes>" seguida do JPEG cru.
  $null = $p.ReadExisting()
  $p.WriteLine("v")
  $limite = (Get-Date).AddSeconds(5); $n = 0
  while ((Get-Date) -lt $limite) {
    try { $l = $p.ReadLine() } catch { return $null }
    if ($l -like "IMG,*") { $n = [int]$l.Trim().Split(",")[1]; break }
    if ($l -like "ERRO,*") { Write-Host "  $l" -ForegroundColor Red; return $null }
  }
  if ($n -le 0) { return $null }
  $buf = New-Object byte[] $n; $lidos = 0
  while ($lidos -lt $n) {
    try { $lidos += $p.Read($buf, $lidos, $n - $lidos) } catch { return $null }
  }
  $ms = New-Object System.IO.MemoryStream(, $buf)
  try { return [System.Drawing.Image]::FromStream($ms) } catch { return $null }
}

function Invoke-Classificar([string]$rot) {
  # Faz $Capturas ciclos "t" e devolve a classe da maioria (ou $null se nada respondeu).
  $p.WriteLine("f $script:fruto"); Start-Sleep -Milliseconds 100; $null = $p.ReadExisting()
  $votos = @{ verde = 0; maduro = 0 }; $pm = @()
  for ($i = 1; $i -le $Capturas; $i++) {
    $p.WriteLine("t")
    $lin = $null
    $limite = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $limite) {
      try { $l = $p.ReadLine() } catch { break }
      if ($l -like "LOG,*") { $lin = $l; break }
    }
    if (-not $lin) { Write-Host "  sem resposta" -ForegroundColor Red; continue }
    $c = $lin.Trim().Split(",")
    # LOG,fruto,modo,classe,confianca,p_verde,p_maduro,int8_v,int8_m,t_sens_cap,t_preproc,t_inf,...
    $classe = $c[3]; $votos[$classe]++; $pm += $c[6]
    "{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f (Get-Date -Format s), $script:fruto, $i, $rot, $classe, $c[4], $c[5], $c[6], $c[11] |
      Out-File $Saida -Append -Encoding utf8
    $cor = if ($classe -eq "maduro") { "Red" } else { "Green" }
    Write-Host ("  captura {0}: {1,-6}  p(maduro)={2}  inferencia {3} ms" -f $i, $classe, $c[6], $c[11]) -ForegroundColor $cor
  }
  $null = $p.ReadExisting()
  if ($votos.verde + $votos.maduro -eq 0) { return $null }
  $final = if ($votos.maduro -gt $votos.verde) { "maduro" } else { "verde" }
  return @{ classe = $final; pm = ($pm -join " / ") }
}

# ---------------------------------------------------------------- janela
$f = New-Object System.Windows.Forms.Form
$f.ClientSize = New-Object System.Drawing.Size(640, 530)
$f.TopMost = $true; $f.KeyPreview = $true
$pb = New-Object System.Windows.Forms.PictureBox
$pb.Dock = "Fill"; $pb.SizeMode = "Zoom"
$lb = New-Object System.Windows.Forms.Label
$lb.Dock = "Bottom"; $lb.Height = 50; $lb.TextAlign = "MiddleCenter"
$lb.Font = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$lb.Text = "Posicione o fruto no quadrado amarelo e aperte V (verde) ou M (maduro)"
$f.Controls.Add($pb); $f.Controls.Add($lb)

$script:fruto = 1; $script:pedido = $null; $acertos = 0; $total = 0
$f.Add_KeyDown({
  switch ($_.KeyCode) {
    "V"      { $script:pedido = "verde" }
    "M"      { $script:pedido = "maduro" }
    "Space"  { $script:pedido = "" }
    "Escape" { $f.Close() }
  }
})
function Set-Titulo { $f.Text = "Fruto $script:fruto  |  V = verde   M = maduro   Espaco = sem rotulo   Esc = sair" }
Set-Titulo
$f.Show()

while ($f.Visible) {
  if ($null -ne $script:pedido) {
    $rot = $script:pedido; $script:pedido = $null
    $lb.BackColor = [System.Drawing.Color]::LightGray
    $lb.Text = "Fruto $script:fruto - classificando..."; $f.Refresh()
    if ($pb.Image) { $pb.Image.Save((Join-Path $pastaPrevias ("fruto_{0:D2}.jpg" -f $script:fruto)), [System.Drawing.Imaging.ImageFormat]::Jpeg) }
    Write-Host ("Fruto {0} (voce: {1})" -f $script:fruto, $(if ($rot) { $rot } else { "sem rotulo" }))
    $r = Invoke-Classificar $rot
    if (-not $r) {
      $lb.Text = "Sem resposta do ESP32"; $lb.BackColor = [System.Drawing.Color]::Orange
    } else {
      $txt = "Fruto {0}: ESP = {1}   p(maduro) {2}" -f $script:fruto, $r.classe.ToUpper(), $r.pm
      if ($rot) {
        $total++; $ok = ($r.classe -eq $rot); if ($ok) { $acertos++ }
        $txt += "`nvoce: {0}  ->  {1}    acertos {2}/{3}" -f $rot, $(if ($ok) { "OK" } else { "ERROU" }), $acertos, $total
        Write-Host ("  => {0}  {1}  acertos {2}/{3}" -f $r.classe, $(if ($ok) { "OK" } else { "ERROU" }), $acertos, $total)
        $script:fruto++; Set-Titulo
      }
      $lb.Text = $txt
      $lb.BackColor = if ($r.classe -eq "maduro") { [System.Drawing.Color]::FromArgb(255, 200, 200) } else { [System.Drawing.Color]::FromArgb(200, 240, 200) }
    }
    [System.Windows.Forms.Application]::DoEvents()
    $script:pedido = $null   # descarta teclas apertadas durante a classificacao
  }
  $img = Get-Quadro
  if ($img) {
    $bmp = New-Object System.Drawing.Bitmap $img
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $L = [Math]::Min($bmp.Width, $bmp.Height)
    $caneta = New-Object System.Drawing.Pen ([System.Drawing.Color]::Yellow), 2
    $g.DrawRectangle($caneta, [int](($bmp.Width - $L) / 2), [int](($bmp.Height - $L) / 2), $L - 1, $L - 1)
    $g.Dispose(); $caneta.Dispose(); $img.Dispose()
    $velho = $pb.Image; $pb.Image = $bmp; if ($velho) { $velho.Dispose() }
  }
  [System.Windows.Forms.Application]::DoEvents()
}
$f.Dispose()
$p.Close()
Write-Host "Resultados em $Saida (previas em $pastaPrevias)"
