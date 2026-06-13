# Klasifikasi Penyakit Payudara (Breast Disease Classification)

Proyek ini bertujuan untuk mengklasifikasikan citra histopatologi kanker payudara menggunakan model *Deep Learning* dengan arsitektur mutakhir **EfficientNet-B5**. Model ini telah dioptimalkan agar mencapai akurasi tinggi menggunakan parameter seperti input resolusi tinggi (456x456), teknik augmentasi lanjutan (*RandAugment*, *MixUp*, *CutMix*), *Test-Time Augmentation (TTA)*, serta dilengkapi dengan antarmuka web interaktif yang menampilkan *Grad-CAM* untuk melihat interpretasi letak fokus dari prediksi model.

## Sumber Dataset
**BreaKHis - Breast Cancer Histopathological Database**  
Dataset dapat diunduh pada tautan berikut: [Mendeley Data - BreaKHis](https://data.mendeley.com/datasets/jxwvdwhpc2/1)

## Persiapan Lingkungan (*Environment Setup*)
1. Letakkan dataset yang telah diekstrak ke dalam folder `data\raw\` berdasarkan skala perbesaran (40X, 100X, 200X, 400X) dengan mematuhi struktur di bawah ini:
```text
C:.
├───data
│   ├───raw
│   │   ├───40X
│   │   │   ├───adenosis
│   │   │   ├───ductal_carcinoma
│   │   │   ├───fibroadenoma
│   │   │   ├───lobular_carcinoma
│   │   │   ├───mucinous_carcinoma
│   │   │   ├───papillary_carcinoma
│   │   │   ├───phyllodes_tumor
│   │   │   └───tubular_adenoma
│   │   ├───100X
│   │   │   └─── ...
│   │   ├───200X
│   │   │   └─── ...
│   │   └───400X
│   │       └─── ...
```

2. Buat *virtual environment*:
```bash
python -m venv venv
```

3. Aktifkan *virtual environment*:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

4. Instal paket dependensi pendukung:
```bash
pip install -r requirements.txt
```

## Persiapan Data (*Data Preparation*)
Jalankan skrip berikut secara beurutan guna mengekstrak metadata dari hirarki folder dataset dan kemudian membaginya sesuai dengan proporsi pelatihan (*training*) dan pengujian (*testing*):
1. Menggabungkan informasi dan membuat *metadata* ke dalam file CSV:
```bash
python src/make_metadata_all.py
```
2. Membagi dataset (*Train/Test Split*):
```bash
python src/split_data_all.py
```

## Pelatihan dan Evaluasi Model (*Model Training & Evaluation*)
Model saat ini dilatih menggunakan standar arsitektur **EfficientNet-B5** yang memanfaatkan citra 456x456. Seluruh proses *learning rate scheduling* dan *early stopping* ditangani di dalam pipeline.

1. Menjalankan secara penuh proses pelatihan:
```bash
python src/train_all.py
```
2. Mengevaluasi akurasi performa model menggunakan data pengujian (*testing set*):
```bash
python src/evaluate_all.py
```

## Prediksi Manual dan Aplikasi Web (*Inference & Web App*)
Anda dapat menjalankan prediksi dengan menggunakan antarmuka via *Command Line* atau melalui GUI di peramban otomatis dengan berbasis **Streamlit**.

1. Melakukan prediksi (*Inference*) manual untuk gambar tertentu melalui terminal:
```bash
python src/infer.py
```

2. **Menjalankan Aplikasi Web (Sangat Direkomendasikan):**
Aplikasi web ini akan memuat model terbaru dan memfasilitasi Anda untuk mengunggah gambar payudara, dan menampilkan kemungkinan klasifikasinya bersamaan dengan *Heatmap / Grad-CAM* untuk melacak penalaran internal model.
```bash
streamlit run app/app.py
```

## Checklist Deploy Publik
Sebelum dipublikasikan, pastikan poin berikut sudah beres:
1. Repository sudah di-push ke GitHub dan branch yang dipakai untuk deploy sudah final.
2. File model yang dibutuhkan tersedia, atau repo Hugging Face `Locelyy/HistopathAI` bisa diakses oleh server deploy.
3. `requirements.txt` sudah hanya berisi dependency yang dibutuhkan app dan tidak ada duplikasi paket.
4. Paket `opencv-python-headless` dipakai untuk deploy server agar tidak bergantung pada komponen GUI.
5. Entry point deploy diarahkan ke `app/app.py`.
6. Coba jalankan lokal dulu dengan `streamlit run app/app.py` sebelum deploy.
7. Setelah deploy, upload satu gambar uji untuk memastikan inference dan Grad-CAM tampil normal.