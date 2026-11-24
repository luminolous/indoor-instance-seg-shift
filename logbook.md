# Logbook

Catatan kendala dan keputusan selama pengerjaan. Entri terbaru di bawah.

Format entri:

```markdown
### YYYY-MM-DD: <judul singkat>

- **Masalah:** apa yang terjadi atau perlu diputuskan.
- **Keputusan:** apa yang saya lakukan dan alasannya.
- **Bukti:** file, log, atau commit.
```

## Ringkasan GPU time

Diambil dari `runs/gpu_time_log.csv`, tidak diketik manual.

| Run id | Sesi | GPU time (jam) | Cap hit | Catatan |
|---|---|---|---|---|
| `yolo_s640_seed42` | 1 | 0.35 | tidak |  |
| `maskrcnn_s640_seed42` | 1 | 0.42 | tidak |  |
| `m2f_s640_seed42` | 1 | 1.32 | tidak |  |
| `yolo_s320_seed42` | 1 | 0.20 | tidak |  |
| `yolo_s960_seed42` | 1 | 0.72 | tidak |  |
| `maskrcnn_s320_seed42` | 1 | 0.37 | tidak |  |
| `maskrcnn_s960_seed42` | 1 | 0.65 | tidak |  |
| `m2f_s320_seed42` | 1 | 0.75 | tidak |  |
| `m2f_s960_seed42` | 1 | 2.05 | ya | berhenti di ~10/50 epoch (VRAM spill) |
| **Total** | 9 | 6.80 | | |

## Entri

### 2026-11-17: Versi dan format dataset

- **Masalah:** Slide meminta Panoptic Indoor Segmentation v5. Format YOLO TXT di Roboflow masuk kategori "Convert To Object Detection" dan format OBB hanya menyimpan box, jadi keduanya membuang mask.
- **Keputusan:** Saya export v5 (922 gambar) dalam format COCO Segmentation. Label YOLO-seg saya turunkan dari file ini.
- **Bukti:** `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/README.roboflow.txt`

### 2026-11-17: Dataset dipakai apa adanya

- **Masalah:** v5 punya 75 kategori (id 0 `indoor-objects` kosong), kelas kembar (`scissor`/`scissors`, `paper`/`paper-merged`, `dining table`/`table-merged`, `wall`/`wall-other-merged`), kelas noise `st-` (1 instance), dan banyak kelas stuff.
- **Keputusan:** Saya tidak merge, drop, atau re-split kelas. Slide meminta test set sendiri memakai kelas yang sama, dan variabel digit 8–9 menunjukkan "semua kelas" adalah setting default. Class index = `category_id` (`num_classes = 75`). Transformasi satu-satunya adalah konversi format COCO ke YOLO-seg. Analisis subset (things vs stuff, kelas yang tidak ada di train) hanya di tahap evaluasi.
- **Bukti:** `configs/base.yaml`, commit 622bab9

### 2026-11-17: Model, hardware, dan variabel NRP

- **Masalah:** Butuh minimal 3 model dari 2 keluarga. NRP saya berakhiran 0, jadi variabelnya input size 320 / 640 / 960.
- **Keputusan:** YOLO11s-seg (one-stage CNN), Mask R-CNN R50-FPN v2 dari torchvision (two-stage CNN), dan Mask2Former Swin-T dari HF transformers (transformer), semuanya pretrained COCO. Saya tidak memakai detectron2 karena sulit diinstal di Windows. Training di laptop (RTX 4050 Laptop, 6 GB). S = sisi terpanjang gambar setelah resize. Cap training 2 jam per run untuk semua run.
- **Bukti:** `configs/models/`, `configs/runs/`

### 2026-11-17: Test set asli sudah bergeser dari train

- **Masalah:** Train dan valid hampir semuanya 640×480. Test punya 216 ukuran berbeda (sisi terpanjang 225–3398 px). Distribusi kelas juga beda jauh (train vs test): `dining table` 4 vs 112, `handbag` 5 vs 125, `wallet` 197 vs 5, `stairs` 175 vs 6. Beberapa kelas test tidak punya data train sama sekali (`keyboard`, `mouse`, `toothbrush`, dll.).
- **Keputusan:** Saya mengevaluasi di tiga level: valid (in-distribution), test asli (bergeser), dan test HP (bergeser lebih jauh).
- **Bukti:** `results/data_validation.md`, `results/eda/class_shift.csv`

### 2026-11-18: Empat gambar terotasi terhadap anotasinya

- **Masalah:** 4 file gambar (train 182169107, 182169121; valid 182169108, 182169109) berukuran 480×640, padahal JSON dan polygon-nya 640×480. Tidak ada EXIF orientation.
- **Keputusan:** Saya memutar gambar saat loading lewat `data.rotation_fixes` di `configs/base.yaml` (90° atau 270° CCW, dicek dengan overlay). File dataset tidak saya ubah.
- **Bukti:** commit c7b2e75, `results/data_validation.md` (PASS)

### 2026-11-18: Daftar stuff untuk breakdown evaluasi

- **Keputusan:** Saya memakai 28 kategori stuff COCO panoptic yang ada di v5, ditambah `wall`. Konsekuensinya `pillow`, `blanket`, `towel`, `curtain`, `light`, `shelf`, `counter`, dan `stairs` ikut dihitung stuff, sesuai definisi COCO. Daftar ini hanya untuk breakdown evaluasi.
- **Bukti:** `configs/base.yaml` (`stuff_classes`)

### 2026-11-18: Sanity check evaluator

- **Masalah:** Saya menguji evaluator dengan GT sebagai prediksi. Ada dua temuan. Pertama, match ke anotasi GT id 0 dihitung false positive karena pycocotools memakai id itu sebagai boolean. Kedua, `maxDets=100` berlaku per gambar per kelas, dan satu gambar test punya 111 `book`.
- **Keputusan:** Id anotasi GT saya geser mulai dari 1 saat loading. Protokol COCO standar tetap dipakai, dengan plafon mask AP 1.0 di valid dan 0.9971 di test.
- **Bukti:** commit 1f03000, `results/eval/sanity_gt_as_pred/`

### 2026-11-18: Protokol supaya perbandingan adil

- **Masalah:** YOLO memilih `best` dengan fitness Ultralytics, sedangkan Mask R-CNN memakai mask AP pycocotools. Selain itu, paste mask Mask R-CNN langsung di resolusi 12 MP butuh sekitar 5 GB VRAM.
- **Keputusan:** Checkpoint yang dievaluasi adalah `last` untuk semua model. Semua model memprediksi mask di resolusi S lalu upsample bilinear ke ukuran asli (tanpa `retina_masks`). Kalau memakai gradient accumulation, LR diskalakan ke effective batch.
- **Bukti:** `src/iiss/predict.py`, wrapper model di `src/iiss/models/`

### 2026-11-23: VRAM probe crash karena OOM

- **Masalah:** Run pertama `vram_probe` crash di YOLO S=960 batch 16: `AcceleratorError: CUDA error: out of memory`.
- **Keputusan:** Notebook yang crash saya simpan sebagai bukti. Probe saya tulis ulang: setiap konfigurasi jalan di subprocess sendiri, dan peak di atas 6141 MB (spill ke shared memory) tidak dihitung muat. Override batch hanya di run yang terpaksa: `yolo_s960` batch 4, `m2f_s960` batch 1 × accumulate 2.
- **Bukti:** `results/vram_probe/attempt1_crashed/`, `results/vram_probe/recommendation.csv`, commit a9b8c91

### 2026-11-23: m2f_s960 tetap dijalankan lokal

- **Masalah:** Probe memperkirakan 50 epoch butuh 2,58 jam, di atas cap 2 jam.
- **Keputusan:** Saya tetap menjalankannya lokal dengan cap 2 jam dan melaporkan cap hit. Mengurangi epoch hanya untuk run ini akan melanggar aturan epoch sama untuk semua S.
- **Bukti:** `configs/runs/m2f_s960_seed42.yaml`, commit 84d06a3

### 2026-11-23: Pencatatan sesi yang mati mendadak

- **Masalah:** Timer hanya menulis baris `gpu_time_log.csv` saat sesi selesai normal. Kalau laptop mati atau proses di-kill, GPU time sesi itu hilang dan cap 2 jam tidak lagi kumulatif. Ini perbaikan pencegahan; di 9 run tidak ada sesi yang mati mendadak.
- **Keputusan:** Setiap sesi menulis heartbeat file (`runs/gpu_time_log.open/<run_id>.json`). Saat run dimulai dan menemukan heartbeat sisa, timer menulis baris `status=killed` dulu.
- **Bukti:** commit bce32e4

### 2026-11-23: Baseline S=640 selesai

- **Masalah:** Ketiga baseline selesai tanpa cap hit: YOLO 100 epoch (1257 s), Mask R-CNN 24 epoch (1494 s), Mask2Former 50 epoch (4751 s). Mask mAP50-95 valid dari validator training: YOLO 0.258, Mask R-CNN 0.168, Mask2Former 0.304. Mask2Former mencapai puncak 0.329 di epoch 15 lalu turun.
- **Keputusan:** Angka final tetap diambil dari `iiss.evaluate`. Penurunan Mask2Former setelah epoch 15 saya catat sebagai kandidat overfitting.
- **Bukti:** `runs/<run_id>/`, `runs/gpu_time_log.csv`

### 2026-11-24: Epoch dikunci

- **Keputusan:** YOLO 100 epoch (batch 8, di 960 batch 4). Mask R-CNN 24 epoch (milestones 16/22, batch 2, lr 0.0025). Mask2Former 50 epoch (batch 2, di 960 batch 1 × accumulate 2, base lr 1e-4). Sama untuk semua S dalam satu model.
- **Bukti:** commit ef1f003

### 2026-11-24: Ablation S=320 dan S=960

- **Masalah:** Enam run ablation perlu dijalankan berurutan (00:39–05:24).
- **Keputusan:** Saya meminta Claude Code menjalankan keenamnya lewat notebook training. Setiap run punya salinan notebook yang sudah dieksekusi sebagai bukti.
- **Bukti:** `notebooks/runs/`, `runs/ablation_driver.log`, commit 2e3e93b

### 2026-11-24: m2f_s960 kena cap karena VRAM spill

- **Masalah:** Peak memory mencapai 9918 MB, di atas VRAM 6 GB. Windows memindahkan sebagian ke shared memory mulai epoch 5, dan waktu per epoch naik dari ~207–515 s menjadi ~950–990 s. Cap menghentikan training di sekitar epoch 10 dari 50, sebelum LR poly turun. Segm AP valid saat berhenti 0.306 dan masih naik. Probe meremehkan peak karena hanya menjalankan beberapa iterasi.
- **Keputusan:** Saya menerima ini sebagai keterbatasan dan tidak menjalankan ulang di Colab. Angka 960 Mask2Former dilaporkan dengan catatan undertrained (~10/50 epoch), sehingga tidak setara dengan 320 dan 640. Run tidak di-resume karena cap bersifat final.
- **Bukti:** `runs/m2f_s960_seed42/`, `runs/gpu_time_log.csv`

### 2026-11-24: Observasi awal dari validator training

- **Masalah:** Skor valid checkpoint `last` (S = 320 / 640 / 960): YOLO 0.192 / 0.258 / 0.240, Mask R-CNN 0.134 / 0.168 / 0.178, Mask2Former 0.257 / 0.304 / 0.306 (capped). Gambar train berukuran 640×480, jadi di S=960 semua gambar di-upsample.
- **Keputusan:** Hipotesis untuk diuji: YOLO tidak mendapat manfaat dari input yang di-upsample, sedangkan Mask R-CNN terbantu oleh fitur RoI yang lebih besar. Kesimpulan menunggu hasil `iiss.evaluate`.
- **Bukti:** `runs/ablation_driver.log`, `results/eda/summary.md`

### 2026-11-24: Foto test set HP

- **Masalah:** Saya hanya memotret di kamar, jadi variasi domainnya sempit. Upload pertama lewat WhatsApp menghapus semua EXIF (0/40) dan mengompres ke 1600×1600. Banyak foto berupa close-up objek yang sama: 40 foto hanya mencakup 9 adegan. Semua 12 foto sulit adalah low light, sebagian hampir hitam (rata-rata kecerahan ~12/255).
- **Keputusan:** Saya mengganti foto WhatsApp dengan file asli dari HP (Xiaomi M2101K7BNY, 3472×3472, EXIF ada). Saya memakai 40 foto ini (28 `normal`, 12 `low_light`) dan mencatat adegan tiap foto di `metadata.csv`. Objek di luar kelas v5, misalnya rice cooker, tidak saya anotasi.
- **Bukti:** `dataset/manual-benchmark/originals/`, `dataset/manual-benchmark/metadata.csv`

### 2026-11-24: Anotasi test set HP di Roboflow

- **Masalah:** Saya menganotasi 40 foto secara manual dengan Smart Polygon, tanpa auto-label: 205 anotasi, 17 kelas, semua nama kelas ada di v5. Roboflow memaksa split (valid 1, test 39) dan menambahkan kategori kosong id 0.
- **Keputusan:** Benda anyaman cokelat bulat di meja adalah piring saya, jadi saya beri label `plate` (7 instance). Export Roboflow saya simpan tanpa perubahan. Importer menggabungkan kedua split menjadi satu test set 40 gambar, mencocokkan file ke `originals/`, dan me-remap category id ke v5 berdasarkan nama.
- **Bukti:** `dataset/manual-benchmark/roboflow-export/`, screenshot di `docs/evidence/roboflow/`

### 2026-11-24: Satu objek teranotasi sebagai beberapa instance

- **Masalah:** Objek yang tertutup sebagian tersimpan sebagai beberapa anotasi terpisah. Contohnya, kursi di bawah tas pada `IMG_20261124_172304` menjadi 7 anotasi `chair`, dan lantainya 5 anotasi `floor-other-merged`. Evaluasi COCO menghitung setiap potongan sebagai objek sendiri, sehingga recall turun walaupun model benar.
- **Keputusan:** Setiap foto di set ini paling banyak punya satu kursi, satu pintu, dan satu lantai. Jadi importer menggabungkan semua potongan `chair`, `door-stuff`, dan `floor-other-merged` dalam satu gambar menjadi satu anotasi multi-polygon. Ada 15 pasangan yang terdampak: `chair` di 172303 (5), 172304 (7), 172306 (2), 172624 (4), 172627 (3); `door-stuff` di 172021 (4), 172401 (3); `floor-other-merged` di 171944 (2), 172303 (5), 172304 (5), 172306 (4), 172556 (2), 172624 (3), 172627 (3), 172647 (2). Kelas lain tidak digabung karena bisa berisi beberapa objek asli. Export mentah tetap tidak diubah.
- **Bukti:** overlay `IMG_20261124_172304` dan `172021`, jumlah anotasi per gambar dari JSON export
