from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
import os
import argparse

from models.keras_models import KeypointDetectorModel
from datasets.keypoints_dataset import KeyPointDatasetBuilder

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--data_dir", default="data_keypoints/", type=str)
parser.add_argument("--x_train_dir", default="train/images", type=str)
parser.add_argument("--y_train_dir", default="train/annotations", type=str)
parser.add_argument("--x_test_dir", default="test/images", type=str)
parser.add_argument("--y_test_dir", default="test/annotations", type=str)
parser.add_argument("--backbone", default="efficientnetb3", type=str)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weights", default=None, type=str)
opt = parser.parse_args()

print("-" * 10)
print("Building model")
print("-" * 10)
name_model = "FPN_" + opt.backbone + "_" + str(opt.lr) + "_" + str(opt.batch_size) + ".h5"
print("Saving the best model weights to {}".format(name_model))

full_model = KeypointDetectorModel(backbone=opt.backbone, num_classes=29, input_shape=(320, 320))

if opt.weights is not None:
    full_model.load_weights(opt.weights)

model = full_model.model
preprocessing_fn = full_model.preprocessing

# define optimizer
optim = keras.optimizers.Adam(opt.lr)
# define loss function
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimizer, loss, and metrics
model.compile(optim, total_loss, metrics)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        name_model, save_weights_only=True, save_best_only=True, mode="min"
    ),
    keras.callbacks.ReduceLROnPlateau(
        patience=10, verbose=1, cooldown=10, min_lr=0.00000001
    ),
]

model.summary()

print("-" * 10)
print("Building dataset")
print("-" * 10)

x_train_dir = os.path.join(opt.data_dir, opt.x_train_dir)
kp_train_dir = os.path.join(opt.data_dir, opt.y_train_dir)
x_test_dir = os.path.join(opt.data_dir, opt.x_test_dir)
kp_test_dir = os.path.join(opt.data_dir, opt.y_test_dir)

full_dataset = KeyPointDatasetBuilder(
    img_train_dir=x_train_dir,
    img_test_dir=x_test_dir,
    mask_train_dir=kp_train_dir,
    mask_test_dir=kp_test_dir,
    batch_size=opt.batch_size,
    preprocess_input=preprocessing_fn,
)

train_dataloader, valid_dataloader = full_dataset._get_dataloader()

print("-" * 10)
print("Launching the training")
print("-" * 10)

model.fit(
    train_dataloader,
    epochs=opt.epochs,
    callbacks=callbacks,
    validation_data=valid_dataloader,
)
