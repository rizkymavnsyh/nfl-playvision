# NFL PlayVision

**An interactive NFL replay with short-horizon player-motion forecasts and receiver–defender geometry.**

**[Open the interactive dashboard](https://rizkymavnsyh.github.io/nfl-playvision/)** · [Download the offline HTML](PlayVision_Offline.html)

The dashboard uses an editorial sports layout and runs entirely in the browser. Its 21 clips are recorded historical plays; the displayed forecasts are retrospective demonstrations, not live game predictions.

## Pembaruan v2

Replay offline dalam paket ini memakai model gabungan: LightGBM khusus untuk horizon 0,5 detik, dan model asli untuk 1 detik. Model khusus dipilih berdasarkan error pada dua pertandingan validasi; versi 1 detik yang baru tidak dipakai karena hasil validasinya lebih buruk. Pada dua pertandingan tes 2021 yang sama, error 0,5 detik turun dari **0,251 ke 0,221 yard**, dan proporsi dalam 0,5 yard naik dari **93,41% ke 95,86%**. Hasil 1 detik tetap **0,919 yard** dan **61,60% dalam 1 yard**.

Tes 2021 ini pernah dilihat dalam proyek v1. Perbandingan v2 adalah benchmark berulang, **bukan tes buta baru**. Lihat `reports/RESULTS.md` dan `models/hybrid_v2/metrics.json`.

## Buka langsung — tanpa instalasi

Ekstrak ZIP lalu buka **`PlayVision_Offline.html`** dengan Chrome/Edge/Firefox. Semua JavaScript dan data replay sudah tertanam, sehingga tidak membutuhkan server atau internet.

- Pilih dari **21 replay nyata**: 18 play dari 12 pertandingan 2021 dan 3 play dari satu pertandingan 2017. Label `TEST`, `VALIDATION`, dan `TRAIN` menunjukkan peran game dalam pelatihan/evaluasi; hanya game `TEST` yang menyumbang angka tes 2021.
- Pilih receiver dan horizon 0,5 atau 1 detik.
- Klik Play atau geser frame; tombol panah juga tersedia.
- Diamond kuning limau adalah posisi prediksi; lingkaran abu-abu adalah baseline kecepatan tetap.
- Aktifkan **Show future path** untuk membandingkan dengan posisi aktual. Informasi masa depan ini adalah overlay evaluasi, bukan input model.
- Panel kanan menunjukkan jarak dan closing speed terhadap defensive back terdekat. Itu hubungan geometris, **bukan identifikasi assignment coverage atau kesalahan pemain**.

## Model yang benar-benar dilatih

Model v1 adalah temporal residual model dengan FLAML AutoML (LightGBM/Extra Trees). Model v2 memakai LightGBM tersendiri untuk horizon 0,5 detik. Model memakai fitur posisi historis, kecepatan/percepatan dari frame terdahulu, perubahan arah, jarak pemain lawan/teammate/bola saat ini, dan posisi resmi. Tidak ada model computer vision dari video dan tidak ada klaim GRU/Transformer telah dilatih.

Baseline: kecepatan tetap dan percepatan tetap yang diperkirakan dari masa lalu. Model belajar koreksi posisi terhadap baseline. Bobot campuran dipilih pada validasi, bukan tes. Frame saat ini dan frame terdahulu saja boleh menjadi predictor. Actor ID, play ID, lokasi pendaratan bola masa depan, hasil play, dan koordinat masa depan tidak menjadi fitur.

## Data dan lingkup bukti

- **2021 regional-event tracking:** 12 pertandingan, dari repositori publik ThompsonJamesBliss, dengan tracking NFL Next Gen Stats. Delapan pertandingan latihan (minggu 1–2), dua validasi (minggu 3), dua tes (minggu 4). Source commit dan SHA-256 disimpan.
- **2017 official NFL game:** KC–NE, sumber NFL Football Operations, untuk tiga replay lengkap dan evaluasi eksternal terpisah. Direction 2017 diinferensikan dari alignment QB dan bola saat snap; kasus ambigu dikeluarkan.
- **2026 Kaggle:** tidak dilatih dalam paket bawaan. Endpoint memberi 401 tanpa akun/persetujuan. Importer menerima file yang Anda unduh sendiri secara sah.

Data regional berhenti sekitar 0,3–0,5 detik setelah pass release. Maka evaluasi regional terutama berasal dari fase sebelum release dalam passing plays, **bukan evaluasi penuh satu detik selama bola di udara**. Laporan phase metrics menampilkan batas itu. Replay 2017 penuh adalah domain/era lain; angka evaluasinya dipisahkan.

Test berisi banyak frame yang saling berkorelasi dari hanya dua pertandingan. Persentase toleransi posisi tidak berarti akurasi hasil pertandingan, catch probability, atau akurasi seluruh NFL. Evaluasi lebih besar dan lebih banyak game diperlukan untuk klaim produksi.

Lihat **`reports/RESULTS.md`** untuk angka yang benar-benar diukur, baseline, moving-only, phase breakdown, dan evaluasi eksternal. Demo scenes dipilih berdasarkan play/cerita yang ditentukan, bukan berdasarkan error terkecil.

## Google Colab dan Drive

1. Upload ZIP sebagai **`NFL_PlayVision_v2.zip`** ke My Drive.
2. Ekstrak ZIP di komputer, lalu upload `NFL_PlayVision_v2_Colab.ipynb` ke Colab.
3. Run all menampilkan model gabungan v2 dan replay terbaru. `TRAIN_NEW=False` adalah default. Notebook v1 tetap tersedia sebagai arsip.
4. Ubah `TRAIN_NEW=True` untuk pelatihan ulang pada data regional. CPU cukup; GPU tidak diperlukan. Default budget 60 detik per sumbu x/y, ditambah pengunduhan, preprocessing, fit terakhir, dan evaluasi.
5. Hasil run baru, HTML, dan metadata disalin ke Drive. Raw snapshot dapat besar, dan disimpan saat pelatihan diminta.

Login Google Drive/Colab memerlukan sesi Anda dan tidak diuji di lingkungan pembuat. Sel Python notebook dan aplikasi lokal diuji secara terpisah.

## Lokal

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python src/playvision.py download
python src/playvision.py prepare
python src/playvision.py train --budget 60 --run-dir runs/my_run
python src/playvision.py evaluate --run-dir runs/my_run
python -m pytest tests -q
streamlit run app.py
```

Untuk mereproduksi eksperimen v2 setelah `download` dan `prepare`:

```bash
python experiments/horizon_models.py
python experiments/build_hybrid_v2.py
python src/export_demo.py --model-path models/hybrid_v2/model.joblib --metrics-path models/hybrid_v2/metrics.json
```

Model bawaan sudah dibekukan dan tidak ditimpa. Pilih run folder baru. Melatih ulang lalu melihat test yang sama adalah benchmark berulang, bukan tes independen baru.

## Data Kaggle 2026 milik Anda

Baca `KAGGLE_2026_GUIDE.md`. Importer membuang `ball_land_x`, `ball_land_y`, `player_role`, dan metadata target masa depan. Waktu snap yang tidak diketahui tetap missing; tidak direkayasa. Adapter diuji dengan fixture sintetis untuk struktur/leakage, **belum dengan file 2026 asli**. Tidak ada synthetic tracking yang dipakai sebagai data latihan/model score bawaan.

## Isi penting

- `PlayVision_Offline.html`: aplikasi offline lengkap.
- `demo/PlayVision_Replay.mp4`: video demo 10,7 detik dari tracking nyata.
- `MODEL_CARD.md`, `DATA_SOURCES.md`, `reports/RESULTS.md`: metode, sumber, angka dan batasan.
- `app.py`: wrapper Streamlit.
- `NFL_PlayVision_v2_Colab.ipynb`: alur Colab untuk model gabungan dan replay terbaru.
- `NFL_PlayVision_Colab.ipynb`: notebook v1 asli sebagai arsip.
- `models/verified/`: model, pilihan validasi, frozen hash, metrik.
- `models/hybrid_v2/`: model gabungan dan metrik terbaru; model v1 tetap tersedia.
- `experiments/horizon_models.py`, `experiments/build_hybrid_v2.py`: reproduksi eksperimen peningkatan dan keputusan pemilihan.
- `assets/replay_data.json`: 21 replay nyata yang diturunkan dari sumber beratribusi.
- `src/`: download, fitur kausal, model, adapter 2017/2026, exporter replay.
- `tests/`: fisika, future-leakage, rotasi arah, split game, importer, inferensi.
- `reports/`: angka nyata, log dan validasi.

Raw CSV sumber dan dataset processed lengkap tidak didistribusikan dalam ZIP. Source terms tetap mengikuti pemilik data. Project ini tidak berafiliasi dengan NFL, tim, Kaggle, atau pemasok data.
