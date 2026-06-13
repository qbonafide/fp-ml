# Klasifikasi Penyakit Payudara (Breast Disease Classification)

Proyek ini bertujuan untuk mengklasifikasikan citra histopatologi kanker payudara menggunakan model *Deep Learning* dengan arsitektur mutakhir **EfficientNet-B5**. Model ini telah dioptimalkan agar mencapai akurasi tinggi menggunakan parameter seperti input resolusi tinggi (456x456), teknik augmentasi lanjutan (*RandAugment*, *MixUp*, *CutMix*), *Test-Time Augmentation (TTA)*, serta dilengkapi dengan antarmuka web interaktif yang menampilkan *Grad-CAM* untuk melihat interpretasi letak fokus dari prediksi model.

## Sumber Dataset
**BreaKHis - Breast Cancer Histopathological Database**  
Dataset dapat diunduh pada tautan berikut: [Mendeley Data - BreaKHis](https://data.mendeley.com/datasets/jxwvdwhpc2/1)

## Environment Setup
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

## Data Preparation
Jalankan skrip berikut secara beurutan guna mengekstrak metadata dari hirarki folder dataset dan kemudian membaginya sesuai dengan proporsi pelatihan (*training*) dan pengujian (*testing*):
1. Menggabungkan informasi dan membuat *metadata* ke dalam file CSV:
```bash
python src/make_metadata_all.py
```
2. Membagi dataset (*Train/Test Split*):
```bash
python src/split_data_all.py
```

## Model Training & Evaluation
Model saat ini dilatih menggunakan standar arsitektur **EfficientNet-B5** yang memanfaatkan citra 456x456. Seluruh proses *learning rate scheduling* dan *early stopping* ditangani di dalam pipeline.

1. Menjalankan secara penuh proses pelatihan:
```bash
python src/train_all.py
```
2. Mengevaluasi akurasi performa model menggunakan data testing:
```bash
python src/evaluate_all.py
```

## Prediksi Manual & Web App
Anda dapat menjalankan prediksi dengan menggunakan antarmuka via *Command Line* atau melalui GUI di peramban otomatis dengan berbasis **Streamlit**.

1. Melakukan prediksi (*Inference*) manual untuk gambar tertentu melalui terminal:
```bash
python src/infer.py
```

2. **Menjalankan Aplikasi Web:**
Aplikasi web ini akan memuat model terbaru dan memfasilitasi Anda untuk mengunggah gambar payudara, dan menampilkan kemungkinan klasifikasinya bersamaan dengan *Heatmap / Grad-CAM* untuk melacak penalaran internal model.
```bash
streamlit run app/app.py
```

## Deploy Notes
Jika file model tidak ikut ke GitHub, deploy Streamlit tetap bisa jalan selama repo Hugging Face `qbonafide/HistopathAI-models` berisi file berikut:
```text
models/resnet50.pth
models/densenet121.pth
models/efficientnet_b5.pth
```
App akan membaca repo itu lewat env `HF_MODEL_REPO`. Kalau perlu ganti repo, set secret/environment variable tersebut di Streamlit Community Cloud.