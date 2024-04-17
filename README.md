TEC-miTarget is a model utilizing deep learning methods for microRNA target prediction. TEC-miTarget uses a pair of miRNA and candidate target site sequences as inputs and predicts the interaction probability between them. 

The model weights we have trained are available on Google Drive (https://drive.google.com/file/d/1L9eQYseXn1cctfl9jEHZ8Z_mpeA_vcKF/view?usp=drive_link). If you use TEC-miTarget in your work, please cite the following publication:

TEC-miTarget: enhancing microRNA target prediction based on deep learning of ribonucleic acid sequences

# Preparation

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

# Setup the enviroment

```
conda env create --file environment.yml
```

# Activate the enviroment

```
conda activate TEC_miTarget
```

# Train

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

# Evaluate
## Sequence level

```
cd output/exp-miRAW
python ../../code/evaluate_sequence_level.py --test ../../datasets/data/miRAW/test_seed_1234.txt --device 0 --outdir ./evaluate/ --model ./model/best_model/model.sav
```

## Transcript level

```
cd output/exp-deepTargetPro
python ../../code/evaluate_transcript_level.py  --mirna_file ../../datasets/data/deepTargetPro/mirna.fasta --mrna_file ../../datasets/data/deepTargetPro/mrna.fasta --query_file ../../datasets/data/deepTargetPro/test_split_0.csv --model ./model/best_model/model.sav --outfile ./evaluate/ --device 0
```

# Predict

Users need to prepare their data in the form of [pairs-need-predict.txt](pairs-need-predict.txt) (miRNA-sequence  mRNA-sequence), and then run the following command.

```
cd output/exp-miRAW
python ../../code/predict_sequence_level.py --pairs ../../pairs-need-predict.txt --model ./model/best_model/model.sav --device 0 --outfile ./predict/
```
The results will be shown in the predict folder as predict.tsv
| miRNA-sequence | mRNA-sequence | p |  
| :--: | :--: | :--: |  
| >CGUGUACACGUGUGUCGGCCCAC | >TGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCAAGA | 0.00021147158986423165 |  
| GAUGGACGUGCUUGUCGUGAAAC | TGTCTAAAGGTATACTGTCCAACTCTTAAGCACTTTATAT | 0.9998706579208374 |  
| ... | ... | ... |

where p is the interaction probability
