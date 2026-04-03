Source dataset:
BreakHis - Breast Cancer Histopathological Database
https://data.mendeley.com/datasets/jxwvdwhpc2/1?

Environment setup
1. put the dataset into data\raw\\(40X, 100X, 200X, 400X)
C:.
├───raw
│   ├───100X
│   │   ├───adenosis
│   │   ├───ductal_carcinoma
│   │   ├───fibroadenoma
│   │   ├───lobular_carcinoma
│   │   ├───mucinous_carcinoma
│   │   ├───papillary_carcinoma
│   │   ├───phyllodes_tumor
│   │   └───tubular_adenoma
│   ├───200X
│   │   ├───adenosis
│   │   ├───ductal_carcinoma
│   │   ├───fibroadenoma
│   │   ├───lobular_carcinoma
│   │   ├───mucinous_carcinoma
│   │   ├───papillary_carcinoma
│   │   ├───phyllodes_tumor
│   │   └───tubular_adenoma
│   ├───400X
│   │   ├───adenosis
│   │   ├───ductal_carcinoma
│   │   ├───fibroadenoma
│   │   ├───lobular_carcinoma
│   │   ├───mucinous_carcinoma
│   │   ├───papillary_carcinoma
│   │   ├───phyllodes_tumor
│   │   └───tubular_adenoma
│   └───40X
│       ├───adenosis
│       ├───ductal_carcinoma
│       ├───fibroadenoma
│       ├───lobular_carcinoma
│       ├───mucinous_carcinoma
│       ├───papillary_carcinoma
│       ├───phyllodes_tumor
│       └───tubular_adenoma
└───test
2. python -m venv venv
3. venv\Scripts\activate
4. pip install -r requirements.txt

Data preparation
1. python src/make_metadata_all.py
2. python src/split_data_all.py

Model training and evaluation
1. python src/train_all.py
2. python src/evaluate_all.py

Manual prediction
1. python src/infer.py