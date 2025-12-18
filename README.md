
# MyoHand (Leap Motion + MyoSuite)

## Prasyarat

- Ultraleap **Gemini Hand Tracking Software** sudah ter-install dan service berjalan
- Kamera Ultraleap/Leap terhubung
- Python **>= 3.9** (untuk MyoSuite)

## Instalasi (Windows / PowerShell)

Jalankan dari folder project ini (`leapMyoHand`):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Set path LeapSDK (jika perlu)

Kalau Gemini tidak di lokasi default, set `LEAPSDK_INSTALL_LOCATION` ke folder `LeapSDK`:

```powershell
$env:LEAPSDK_INSTALL_LOCATION = "C:\Program Files\Ultraleap\LeapSDK"
```

## Build `leapc_cffi` (hanya jika import `leap` gagal)

Kalau muncul error seperti `ModuleNotFoundError: No module named 'leapc_cffi._leapc_cffi'`:

```powershell
pip install -r .\leapc-python-bindings\requirements.txt
python -m build .\leapc-python-bindings\leapc-cffi
```

## Run

```powershell
python .\leap_myosuite.py
```
