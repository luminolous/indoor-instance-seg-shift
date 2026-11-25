# Deklarasi penggunaan AI

## Alat yang saya pakai

| Alat | Untuk apa |
|---|---|
| Claude (claude.ai, chat) | Memahami soal, mengecek dataset dan foto, memilih model dan protokol, memandu anotasi di Roboflow, menyusun instruksi dan prompt untuk Claude Code, menulis draf logbook dan dokumen ini |
| Claude Code (lokal) | Menulis kode di `src/`, test, notebook, config, dan README; menjalankan smoke test; atas permintaan saya juga menjalankan `vram_probe` dan 6 run ablation |

Saya sendiri yang mengambil semua keputusan di logbook, memotret dan menganotasi 40 foto, menjalankan 3 baseline dan 27 job evaluasi, serta melakukan semua commit. Claude Code bekerja dengan instruksi tetap (`CLAUDE.md` dan folder `.agent/`, tidak di-commit): dataset read-only, tanpa training berat kecuali saya minta, tanpa commit, dan tanpa angka yang tidak berasal dari log.

## Prompt utama

Prompt untuk Claude Code, disingkat:

1. **Setup dan aturan kerja.** "Baca `CLAUDE.md` dan `.agent/`. Kerjakan roadmap satu task per giliran, laporkan dalam Bahasa Indonesia, beri commit message, lalu berhenti. Training berat saya yang jalankan."
2. **Protokol adil antar model.** "Seragamkan pemilihan checkpoint: evaluasi memakai `last` untuk semua model. Semua model memprediksi mask di resolusi S lalu upsample bilinear ke ukuran asli, tanpa `retina_masks`. Kalau grad accumulation aktif, LR diskalakan ke batch efektif."
3. **Import test set HP.** "Gabungkan split valid (1) dan test (39) export Roboflow menjadi 40 gambar. Cocokkan file ke `originals/` lewat nama asli, remap category id ke v5 berdasarkan nama, dan gagal kalau ada nama yang tidak dikenal. Gabungkan potongan `chair`, `door-stuff`, dan `floor-other-merged` dalam satu gambar menjadi satu anotasi multi-polygon, lalu tulis laporannya."
4. **Evaluasi.** "Pastikan `predict_and_evaluate.ipynb` mendukung set phone dengan breakdown `normal` vs `low_light`, lalu serahkan ke saya untuk 9 run × {valid, test, phone}."

Prompt ke Claude chat berupa percakapan biasa, misalnya "pahamilah tugas computer vision saya ini", "foto apa yang harus saya ambil", dan "cek repo saya dan bantu isi logbook".

## Saat AI salah atau tidak cocok

**1. Saran mengurangi kelas.** Claude chat menyarankan training hanya dengan sekitar 25 sampai 30 kelas "things". Saran ini bertentangan dengan soal: test set harus memakai kelas yang sama dengan dataset, dan variabel digit 8–9 menunjukkan "semua kelas" sebagai default. Saya mempertanyakannya, lalu kami beralih ke kebijakan modifikasi minimal: semua 75 kelas, split asli, hanya konversi format.

**2. Kriteria checkpoint berbeda antar model.** Wrapper pertama dari Claude Code memilih `best` YOLO dengan fitness Ultralytics, sedangkan Mask R-CNN memakai mask AP pycocotools. Perbandingannya jadi tidak setara. Saya meminta perbaikan: semua model dievaluasi dengan checkpoint `last` dan satu evaluator pycocotools (prompt 2).

**3. Bug evaluator id 0.** Evaluator awal menghitung match ke anotasi GT berid 0 sebagai false positive, karena pycocotools memakai id itu sebagai boolean. Sanity test (GT sebagai prediksi harus AP 1.0) menangkapnya, dan Claude Code memperbaikinya dengan menggeser id mulai dari 1 (commit 1f03000).

**4. VRAM probe meleset.** Probe pertama crash dengan `AcceleratorError: CUDA error: out of memory`. Setelah ditulis ulang, probe memperkirakan peak `m2f_s960` 5432 MB, padahal saat training peak mencapai 9918 MB dan run kena cap di sekitar epoch 10 dari 50. Saya menerima ini sebagai keterbatasan dan melaporkan angka 960 Mask2Former sebagai undertrained.

**5. Klaim di logbook.** Draf logbook dari Claude menulis bahwa spill VRAM `m2f_s960` dimulai di epoch 5. Claude Code mengecek `results.csv` dan TensorBoard, lalu menemukan perlambatan sudah mulai setelah epoch 1. Saya mengoreksi entrinya.

**6. Salah mengenali objek.** Claude chat menyebut piring anyaman saya sebagai keranjang dan menyarankan tidak menganotasinya. Saya mengoreksi, dan label `plate` tetap dipakai.
