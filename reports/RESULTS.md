# Hasil terverifikasi NFL PlayVision

## Pembaruan v2

Model baru khusus horizon 0,5 detik dipilih karena error validasinya **0,217 yard** dibanding **0,249 yard** pada v1. Kandidat khusus 1 detik memiliki error validasi **0,915 yard**, lebih buruk daripada v1 **0,901 yard**, sehingga v1 dipertahankan untuk horizon itu.

| Horizon | v1 error tes | v2 gabungan error tes | v1 dalam toleransi | v2 gabungan dalam toleransi |
|---|---:|---:|---:|---:|
| 0,5 detik, ±0,5 yard | 0,251 yard | **0,221 yard** | 93,41% | **95,86%** |
| 1 detik, ±1 yard | 0,919 yard | 0,919 yard | 61,60% | 61,60% |

Pengurangan error pada 0,5 detik adalah sekitar **11,9%** relatif terhadap v1. Sampel tes masih 16.799 dan 13.886 frame dari **dua pertandingan 2021 yang sama**. Tes tersebut telah dilihat dalam proyek v1, jadi angka v2 adalah benchmark berulang dan memerlukan tes baru pada lebih banyak game sebelum klaim generalisasi.

Pada satu pertandingan eksternal KC–NE 2017, tanpa melatih atau memilih ulang model, error 0,5 detik untuk semua fase turun dari **0,398 ke 0,354 yard**; pada 4.612 frame setelah bola dilepas turun dari **0,314 ke 0,248 yard**. Ini mendukung transfer terbatas ke pertandingan tersebut, tetapi belum menunjukkan performa konsisten seluruh liga atau musim. Detail ada di `reports/hybrid_v2_external2017.json`.

## Model v1 yang dibekukan

Prediksi perpindahan posisi pemain 0,5 dan 1 detik dari riwayat tracking, bukan prediksi hasil pertandingan.

| Horizon | Error rata-rata (yard) | Dalam 0,5 yard | Dalam 1 yard | Baseline percepatan: error |
|---|---:|---:|---:|---:|
| 0,5 detik | 0,251 | 93,41% | 99,97% | 0,372 |
| 1 detik | 0,919 | 24,23% | 61,60% | 1,369 |

Tes regional 2021 mencakup 2 pertandingan dan 116 passing plays, terpisah dari 8 game latihan dan 2 validasi. Jumlah sampel per horizon 16.799 dan 13.886; sampel dalam satu play berkorelasi. Error model lebih rendah sekitar 32–33% dari baseline percepatan tetap.

Data regional berakhir hanya 0,3–0,5 detik setelah release. Mayoritas evaluasi adalah gerakan sebelum release pada passing play; tidak ada sampel regional 1 detik dengan anchor setelah release. Angka tinggi pada horizon pendek tidak boleh dipasarkan sebagai akurasi menang, catch, atau performa mingguan.

Evaluasi tambahan pada satu game lengkap KC–NE tahun 2017, tanpa melatih ulang model: fase setelah release memiliki error rata-rata 0,314 yard (0,5 detik) dan 0,972 yard (1 detik); dalam 1 yard masing-masing 99,59% dan 58,15%. Ini uji transfer era pada satu game, bukan bukti generalisasi liga. Replay touchdown dipilih sebagai ilustrasi cerita, bukan sampel acak untuk evaluasi.

AutoML FLAML mencari LightGBM dan ExtraTrees, budget 60 detik per sumbu. Model akhir LightGBM memprediksi koreksi terhadap kecepatan tetap; campuran baseline ditentukan menggunakan validasi saja. Test dibuka setelah pembekuan model. Fitur input terbatas pada posisi/gerakan yang telah diamati dan geometri tetangga saat ini; future truth hanya untuk target/evaluasi.

Belum dikerjakan: pelatihan pada dataset kompetisi 2026 asli, evaluasi banyak musim/game penuh, interval prediksi terkalibrasi, identifikasi coverage assignment, dan video computer vision. Adapter 2026 diuji dengan fixture schema saja. Angka ini bukan skor leaderboard.
