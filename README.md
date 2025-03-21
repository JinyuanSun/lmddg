# lmddg
A landing project

1. Train data: S2648.tsv, Test data: S669.tsv
2. esm_lgbm_baseline is the demo.
```bash
cd esm_lgbm_baseline/
python main.py # trian & test
python predict.py --model esm2_650m_model.txt --output s669_predict.csv # run test and save results
```