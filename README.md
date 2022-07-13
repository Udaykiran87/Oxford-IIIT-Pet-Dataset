## CUDA and cuDNN version
```bash
CUDA: 11.2
cuDNN: 8.1
```

## Oxford-IIIT-Pet-Dataset
Oxford-IIIT-Pet-Dataset object detection using TFOD2.

## STEPS-
### STEP 00- Clone repository and chnage directory to Oxford-IIIT-Pet-Dataset directory

```bash
git clone https://github.com/Udaykiran87/Oxford-IIIT-Pet-Dataset.git

cd Oxford-IIIT-Pet-Dataset
```
### STEP 01- Create a conda environment and install necessary packages after opening the repository in VSCODE

```bash
init_setup.sh
```
### STEP 02- Activate the environment
```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 03- Create necessary folder structures: Stage 01
```bash
python src/components/stage_01_TFOD_setup.py
```

### STEP 04- Install TFOD2: Stage 02
```bash
python src/components/stage_02_install_TFOD.py
```

### STEP 05- Download dataset and prepare data: Stage 03
```bash
python src/components/stage_03_prepare_data.py
```

### STEP 06- Custom model training: Stage 04
```bash
python src/components/stage_04_TFOD_custom_train_eval.py
```
### STEP 07- Model prediction: Stage 05
```bash
python src/components/stage_05_predict.py
```
### STEP 08- Freeze model graph: Stage 06
```bash
python src/components/stage_06_freeze_graph.py
```
### STEP 09- Tflite model conversion: Stage 07
```bash
python src/components/stage_07_tflite_conversion.py
```