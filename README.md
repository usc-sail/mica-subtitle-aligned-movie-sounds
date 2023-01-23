# Subtitle Aligned Movie Sounds (SAM-S)

Github repo for the paper titled "A dataset for Audio-Visual Sound Event Detection in Movies"


## Dataset
Details to download the dataset, features and pretrained models will be provided upon acceptance.

## Dependencies
torch - 1.8.1  
torchaudio - 0.8.1  
timm - 0.4.5  
transformers - 4.17.0

## Training 
Install dependencies 
```
conda create -n env astmm2 
conda activate astmm2
pip install requirements.txt
```
Edit training parameters in run.sh/run_mm.sh

Change paths to audio/video features in data/*json

### Unimodal baseline (Audio only):
Results are stored in eval_results.csv file. Complete log will be written in log.txt in the experiment directory.
For baseline shown in the paper, the following parameters are used:
```
fshape=tshape=1     ## Convolution kernel-size for audio spectrogram patch embeddings
fstride=tstride=10   ## Convolution stride for audio spectrogram patch embeddings
mixup=0.5           ## Probability with which two samples and their labels will be mixed
freqm=48            ## Maximum Frequency Masking Strip
timem=192           ## Maximum Time Masking Strip
n_class=120         ## Number of sound classes
```

The following script will initiate training and evaluation for audio event detection.

```
bash/sbatch run.sh
```

### Multimodal baseline:
```
bash/sbatch run.sh
```

## Results
The following results are obtained on 2xTESLA V100 GPUs with 32GB memory each.

|Model|Modality|mAP|AUC|d-prime|
|-----|-----|-----|----|----|
|VGGSound|A|14.1|87|1.59|
|AST|A|34.76|95.02|2.33|
| AST-MM (sinusoid) | AV | 35.67 | 95.05 | 2.33 |
| AST-MM (BERT) | AV | 35.82 | 95.11 | 2.34 |
| AST-MM (Learnable) | AV | 36.3 | 95.25 | 2.36 | 


## Citing 
Details will be added upon acceptance
