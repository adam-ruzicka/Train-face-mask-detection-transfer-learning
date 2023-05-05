# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, default="dataset", help="cesta k datasetu")
ap.add_argument("-r", "--roc", type=str, default="roc.png", help="cesta k vystupnemu obrazku ROC krivky")
ap.add_argument("-c", "--cm", type=str, default="cm.png", help="cesta k vystupnemu obrazku konfuznej matice")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="cesta k vystupnemu obrazku presnosti a chybovosti")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="cesta k vystupnemu modelu")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and class images
print("nacitavam obrazky...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25%
# for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("kompilujem model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("trenujem hlavy siete...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("vyhodnocujem siet....")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted
# probability
predIdxs = np.argmax(predIdxs, axis=1)

Y_test = np.argmax(testY, axis=1)

cm = confusion_matrix(Y_test, predIdxs)
TN, FP, FN, TP = cm.ravel()
# TP = cm[0][0]
# FN = cm[0][1]
# FP = cm[1][0]
# TN = cm[1][1]

fpr_roc, tpr_roc, thresholds = roc_curve(Y_test, predIdxs)
roc_auc = auc(fpr_roc, tpr_roc)

plt.title('ROC krivka')  # Receiver Operating Characteristic
plt.plot(fpr_roc, tpr_roc, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig(args["roc"], dpi=300)

cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(10, 10))

ax = sns.heatmap(cm, annot=True, annot_kws={"fontsize": 28}, fmt='d')

plt.title("Konfúzna matica", fontsize=30)
ax.set_xlabel('Skutočnosť', fontsize=20)
ax.set_ylabel('Odhad', fontsize=20)
ax.tick_params(length=0, labeltop=True, labelbottom=False)
ax.xaxis.set_label_position('top')
ax.set_xticklabels(['Pozitívne', 'Negatívne'])
ax.set_yticklabels(['Pozitívne', 'Negatívne'], rotation=90, va='center')

plt.savefig(args["cm"], dpi=300)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)

# Specificity or true negative rate
# TNR = TN/(TN+FP)

# Precision or positive predictive value
# PPV = TP/(TP+FP)

# Negative predictive value
# NPV = TN/(TN+FN)

# Fall out or false positive rate
FPR = FP / (FP + TN)

# False negative rate
# FNR = FN/(TP+FN)

# False discovery rate
# FDR = FP/(TP+FP)

print("TPR: ", TPR, ", FPR: ", FPR)

# Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("ukladam model...")
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="trénovacia_chybovosť")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validačná_chybovosť")
plt.plot(np.arange(0, N), H.history["accuracy"], label="trénovacia_presnosť")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validačná_presnosť")
plt.title("Chybovosť a presnosť trénovania")
plt.xlabel("Epocha #")
plt.xticks(np.arange(0, N), np.arange(1, N + 1).astype(int))
plt.ylabel("Chybovosť/Presnosť")
plt.legend(loc="lower left")
plt.savefig(args["plot"], dpi=300)
