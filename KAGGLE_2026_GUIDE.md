# Menggunakan tracking NFL Big Data Bowl 2026

Sumber resmi: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data

Masuk melalui akun Kaggle sendiri dan penuhi ketentuan kompetisi yang ditampilkan. Unduh serta ekstrak dataset ke Drive/lokal. Jangan menyimpan API token dalam ZIP, notebook publik, atau repository.

Struktur yang dibaca:

```text
train/
  input_2023_w01.csv
  output_2023_w01.csv
  input_2023_w02.csv
  output_2023_w02.csv
```

Importer mengubah output frame relatif menjadi frame setelah akhir input, menghubungkan metadata pemain yang sudah diketahui, dan mempertahankan koordinat observasi. Ia tidak membawa lokasi pendaratan bola atau role target masa depan ke fitur.

```bash
python src/import_kaggle2026.py --data-dir /path/to/competition --output-dir data/imported2026 --max-files 4 --max-games 12
python src/playvision.py train --dataset-dir data/imported2026 --run-dir runs/kaggle2026 --budget 300
python src/playvision.py evaluate --dataset-dir data/imported2026 --run-dir runs/kaggle2026
```

Catatan penting:

- Ini **rolling short-horizon forecast**, bukan generator submission resmi. Jangan menyebut hasil lokal sebagai skor leaderboard Kaggle.
- Input/output competition tidak menyediakan semua fitur bola dan snap timestamp yang tersedia dalam sumber regional. Missingness dipertahankan; model harus dilatih ulang pada schema tersebut, bukan diasumsikan akurasi model regional langsung berlaku.
- Actor yang tidak ada di output tidak diisi dengan gerakan palsu. Konteks sesudah release dapat menjadi lebih jarang.
- Split dilakukan per game. Game ID diurutkan dan disampel menyebar dalam file yang dipilih; tanggal/kickoff perlu diperiksa sebelum klaim temporal murni.
- Adapter diuji dengan fixture kecil; akses data 2026 asli tidak tersedia pada pembuatan paket ini. Audit coverage dan hasil adapter sebelum training besar.
- Notebook bawaan memperlihatkan replay/hasil regional 2021 dan official 2017. Import 2026 tidak otomatis mengubah atribusi atau angka demo bawaan.
