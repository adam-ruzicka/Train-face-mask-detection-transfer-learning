# Trénovanie vlastného modelu na detekciu masky na tvári použitím metódy preneseného učenia

## PROJEKT
### https://github.com/Piyush2912/Real-Time-Face-Mask-Detection

## DATASET
### https://medium.com/analytics-vidhya/covid-19-face-mask-detection-system-with-tensorflow-and-opencv-1bd19a14125e

## TRÉNOVANIE
```python train.py --dataset dataset```

## KONVERZIA DO FORMÁTU TFLITE
```python convert.py```

## TESTOVANIE
```python detect.py```

## Vyhodnotenie

### Proces trénovania
![Proces trénovania](https://github.com/adam-ruzicka/Train-face-mask-detection-transfer-learning/blob/main/plot.png)

### Konfúzna matica
![Konfúzna matica](https://github.com/adam-ruzicka/Train-face-mask-detection-transfer-learning/blob/main/cm.png)

### ROC krivka
![ROC krivka](https://github.com/adam-ruzicka/Train-face-mask-detection-transfer-learning/blob/main/roc.png)
