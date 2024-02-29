# preparation

```
cd TEC-miTarget
mkdir output
mkdir output/exp-miRAW
mkdir output/exp-DeepMirTar
mkdir output/exp-deepTargetPro
mkdir output/exp-miRAW/model
mkdir output/exp-miRAW/evaluate
mkdir output/exp-miRAW/predict
mkdir output/exp-DeepMirTar/model
mkdir output/exp-DeepMirTar/evaluate
mkdir output/exp-DeepMirTar/predict
mkdir output/exp-deepTargetPro/model
mkdir output/exp-deepTargetPro/evaluate
mkdir output/exp-deepTargetPro/predict
```

# setup the enviroment

```
conda env create --file environment.yml
```

# activate the enviroment

```
conda activate TEC_miTarget
```

# train

```
cd output/exp-miRAW
python ../../code/train.py --train ../../datasets/data/miRAW/train_seed_1234.txt --outdir ./model/ --no-w --valid ../../datasets/data/miRAW/valid_seed_1234.txt 
```

```
cd output/exp-DeepMirTar
python ../../code/train.py --train ../../datasets/data/DeepMirTar/train_seed_1234.txt --outdir ./model/ --no-w --valid ../../datasets/data/DeepMirTar/valid_seed_1234.txt
```

```
cd output/exp-deepTargetPro
python ../../code/train.py --train ../../datasets/data/deepTargetPro/train_seed_1234.txt --outdir ./model/ --no-w --valid ../../datasets/data/deepTargetPro/valid_seed_1234.txt
```

# evaluate
## sequence level

```
cd output/exp-miRAW
python ../../code/evaluate_sequence_level.py --test ../../datasets/data/miRAW/test_seed_1234.txt --device 0 --outdir ./evaluate/ --model ./model/best_model/model.sav
```

## transcript level

```
cd output/exp-deepTargetPro
python ../../code/evaluate_transcript_level.py  --mirna_file ../../datasets/data/deepTargetPro/mirna.fasta --mrna_file ../../datasets/data/deepTargetPro/mrna.fasta --query_file ../../datasets/data/deepTargetPro/test_split_0.csv --model ./model/best_model/model.sav --outfile ./evaluate/ --device 0
```

# predict

```
cd output/exp-miRAW
python ../../code/predict_sequence_level.py --pairs ../../datasets/data/miRAW/test_seed_1234.txt --model ./model/best_model/model.sav --device 0 --outfile ./predict/
```


