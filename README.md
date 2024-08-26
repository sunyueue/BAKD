# BAKD


## 1. prepare the dataset and environment 


### 1.1 put the Vaihingen on folder ./vaihingen256_stride

```
vaihingen256_stride
├── ann_dir
│   ├── test
│   ├── train
│   └── val
└── gtCoarse Get it using the folder ./dilation
    ├── train
    └── train_color
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



## 2. train BAKD

```
bash train_scripts/exp_yaogan_noisy/yaogan_BAKD_noisy_vaihingen.sh
```

## 3. text
```
python eval.py/prediction.py
```
