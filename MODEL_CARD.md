# NFL PlayVision — model card

## Versi aktif

Paket v2 memakai LightGBM residual khusus untuk prediksi 0,5 detik dan model FLAML v1 untuk 1 detik. Pemilihan dilakukan pada validasi per pertandingan. Hasil tes 0,5 detik: error rata-rata 0,221 yard; 95,86% dalam 0,5 yard. Hasil 1 detik tetap 0,919 yard; 61,60% dalam 1 yard. Tes v2 memakai dua pertandingan 2021 yang sudah pernah dievaluasi pada v1, jadi bukan tes buta baru.

Pada satu pertandingan eksternal 2017, error 0,5 detik turun dari 0,398 ke 0,354 yard secara keseluruhan, dan dari 0,314 ke 0,248 yard pada frame setelah pass release. Pengujian ini tetap terbatas pada satu game.

## Tujuan
Rolling positional forecast pada skill players dan defensive backs, horizon 0,5/1 detik. Input tracking 10 Hz dan metadata posisi pemain. Keluaran delta koordinat lapangan, satuan yard. Bukan model penilaian keseluruhan pemain, rekomendasi medis, atau prediksi hasil taruhan.

## Metode
38 fitur kausal, riwayat maksimal 9 frame; kecepatan, percepatan, putaran, geometri bola dan tetangga. Dua regressor residual AutoML FLAML, dipilih LightGBM. Imputer dan encoder dilatih pada train. Campuran per horizon menggunakan validasi. IDs game/pemain dan future landing/target role tidak masuk fitur.

## Reproduksi
Jalankan notebook atau CLI README. Artifact bawaan dibekukan beserta hash dan split. Versi dependency tercatat requirements.txt. Unduh raw dari sumber untuk pelatihan ulang; raw tidak didistribusikan dalam ZIP. Budget pencarian menyebabkan hasil retraining bisa berbeda antar mesin.

## Evaluasi dan keterbatasan
Lihat reports/RESULTS.md, metrics.json dan phase_metrics.json. Test hanya 2 game 2021, sebagian besar sebelum release; frame berkorelasi dan distribusi posisi terbatas. Tidak ada interval kepercayaan berbasis banyak pertandingan. Drift era, manuver tajam, bola tidak teramati, dan panjang horizon menurunkan ketepatan. Jangan menganggap confidence atau assigned defender dari nearest-neighbor geometry. Data 2017 hanya evaluasi dan demo, tidak untuk tuning.

## Lisensi dan atribusi
Kode proyek: MIT (LICENSE). Hak data sumber tetap milik penyedia; lihat DATA_SOURCES.md dan ketentuan sumber sebelum redistribusi atau penggunaan komersial. Tidak berafiliasi dengan NFL, NGS, PFF atau Kaggle.
