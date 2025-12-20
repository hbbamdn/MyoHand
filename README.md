
# MyoHand (Leap Motion + MyoSuite)

## Prasyarat

- Kamera Ultraleap/Leap terhubung
- Python **>= 3.9** (untuk MyoSuite)


## Instalasi (Windows / PowerShell)

```
powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Setup Leap Motion


Unduh dan instal Ultraleap **Gemini Hand Tracking Software** dari [ultraleap.com/downloads](https://www.ultraleap.com/downloads/) lalu pastikan servicenya berjalan.
Dengan kamera sudah terhubung dan virtual environment aktif (lihat bagian Instalasi di bawah), 
```powershell
python -m build .\leapc-python-bindings\leapc-cffi
pip install .\leapc-cffi\dist\leapc_cffi-0.0.1.tar.gz
pip install -e leapc-python-api
```

## Setup MyoSuite

```powershell
pip install -U myosuite
python -m myosuite_init
python -m myosuite.tests.test_myo
```

Langkah di atas mengunduh paket MyoSuite, menarik aset SimHive (akan meminta konfirmasi lisensi), dan menjalankan tes bawaan untuk memastikan semua environment terdaftar dengan benar.


## Run

```powershell
python .\leap_myosuite.py
```

## Cara record

Pastikan window OpenCV aktif/fokus, lalu:

- `r` : start/stop recording (butuh MyoSuite env; kalau tidak ada akan muncul warning)

Output:

- `recordings/ctrl_display_<YYYYmmdd_HHMMSS>_<session>.csv`

Format CSV:

- Kolom awal: `t`, `frame`
- Lalu: `ctrl_0 ... ctrl_(nu-1)` (nilai `env.sim.data.ctrl`)
