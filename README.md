# VisaodeMaquinaPy
Implementa algoritmo para identificação de carros, pedestres ou motos em videos

# Setup

Requisitos: Python>=3.7.0, conda e PyTorch>=1.7

```Shell
git clone https://github.com/ultralytics/yolov5 yolo
cd yolo
conda create --name yolo python=3.8
conda activate yolo #ativa o ambiente
pip install -r requirements.txt
```

# Rodar o modelo
Na pasta yolo:
```Shell
python detect.py --source #caminho pro arquivo que deseja classificar
```
O resultado sera salvo na pasta runs

# Ativar/desativar ambiente conda configurado

Para ativar o ambiente:
```Shell
conda activate yolo
conda deactivate
```
Para desativar o ambiente:
```Shell
conda activate yolo
conda deactivate
```