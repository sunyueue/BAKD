# BAKD


## 1. prepare the dataset and environment 


### 1.1 put the Vaihingen on folder ./vaihingen256_stride  

download: https://drive.google.com/file/d/13Pq6lTtw4aZjphXMF0k8bc0X0425JAJF/view?usp=sharing

```
vaihingen256_stride
├── ann_dir
│   ├── test
│   ├── train
│   └── val
└── gtCoarse 
    ├── train (Get it using the folder ./dilation)
    └── train_color (Get it using the folder ./dilation)
    └── exp  (Get it using the folder ./distance_weight)
└── ima_dir
│   ├── test
│   ├── train
│   └── val
```

### 1.2 conda environment

```
conda env create -f BAKD.yaml
```
### 1.3 prepare the weight files 

download https://drive.google.com/file/d/1CmjImdww50jmUDHm034bs1WjBTNTZI1h/view?usp=sharing

## 2. train BAKD

```
bash train_scripts/exp_yaogan_noisy/yaogan_BAKD_noisy_vaihingen.sh
```

## 3. test
```
python eval.py/prediction.py
```
