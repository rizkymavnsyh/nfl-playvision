# Sumber dan provenance

1. Regional event NFL Big Data Bowl tracking 2021: https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data . Commit dipin di data/source_commit.txt; manifest hash di data/source_manifest.json; split di data/split_plan.json. Metadata PFF tersedia di sumber tetapi tidak digunakan sebagai fitur model. ZIP berisi subset replay turunan, bukan seluruh raw.
2. Official NFL Big Data Bowl 2017: https://github.com/nfl-football-ops/Big-Data-Bowl . Game 2017090700 (KC–NE), untuk replay penuh dan evaluasi eksternal. Manifest file yang diunduh ada di data/legacy_manifest.json. Arah serang diinferensikan dari posisi QB vs bola saat snap, play ambigu dilewati.
3. Kaggle NFL Big Data Bowl 2026: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction . Unduhan tanpa autentikasi tidak tersedia pada sesi ini. Dataset tersebut tidak dilatih/evaluasi dalam hasil bawaan. Gunakan unduhan resmi yang Anda berhak akses dengan panduan KAGGLE_2026_GUIDE.md; task rolling forecast tidak identik dengan submission kompetisi.

Koordinat dalam yard; data 2021 memiliki timestamp dibulatkan per detik. Laju 10Hz diverifikasi melalui perubahan timestamp setiap 10 frame, bukan mengasumsikan timestamp per baris berbeda 0,1 detik. Lihat reports/timing_audit.json.

Data dan merek bukan bagian dari lisensi MIT kode. Tinjau ketentuan pemilik sumber/Kaggle sebelum memublikasikan ulang dataset. Tidak ada izin komersial tambahan yang dijanjikan proyek ini.
