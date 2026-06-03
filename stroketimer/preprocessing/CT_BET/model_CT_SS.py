#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definition and utilities for CT_BET (2D/3D U-Net) with optional per-slice export.

Original header:
Created on Tue Aug 15 13:28:01 2017
@author: m131199
"""

import os
import numpy as np
import nibabel as nb
import SimpleITK as sitk

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation,
    Dropout, Flatten, Reshape, Cropping3D, Dense, ZeroPadding2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta

from sklearn import metrics
from sklearn.model_selection import KFold

# keep original import path names for compatibility
from  scipy.ndimage.interpolation import zoom as interp3D

# user local modules
from deepModels import Unet, Unet3D
from load3Ddata import arrangeData as arrange3Ddata
from load3Ddata import arrange3DtestImage, arrange3DtestLabel


class Unet_CT_SS(object):

    def __init__(self, root_folder='', code_folder='', image_folder='image_data',
                 mask_folder='mask_data', save_folder='', training_folder='',
                 test_folder='', pred_folder='', testingMode=False, savePredMask='False',
                 testLabelFlag=True, testMetricFlag=False, dataAugmentation=False,
                 includeNC=False, add_images='', add_masks='', sC=2, model='unet', lr=1e-5, decay=1e-12,
                 timeStamp='', logFileName='', oLabel='', datagen='', checkWeightFileName='',
                 afold=3, numEpochs=1, bs=1, img_row=512, img_col=512, channel=1, nb_classes=2,
                 classifier='softmax', optimizer='', dtype='float32', dtypeL='uint8',
                 wType='slice', loss='categorical_crossentropy', metric='accuracy'):

        self.root_folder = root_folder
        self.code_folder = code_folder
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.save_folder = save_folder
        self.add_images = add_images
        self.add_masks = add_masks
        self.training_folder = training_folder
        self.test_folder = test_folder
        self.pred_folder = pred_folder
        self.logFileName = logFileName
        self.testingMode = testingMode
        self.testLabelFlag = testLabelFlag
        self.testMetricFlag = testMetricFlag
        self.dataAugmentation = dataAugmentation
        self.savePredMask = savePredMask

        self.afold = afold
        self.numEpochs = numEpochs
        self.optimizer = optimizer
        self.oLabel = oLabel
        self.checkWeightFileName = checkWeightFileName
        self.datagen = datagen
        self.dtype = dtype
        self.dtypeL = dtypeL
        self.wType = wType
        self.img_row = img_row
        self.img_col = img_col
        self.channel = channel
        self.classifier = classifier
        self.bs = bs
        self.sC = sC
        self.nb_classes = nb_classes
        self.includeNC = includeNC
        self.model = model
        self.lr = lr
        self.decay = decay
        self.loss = loss
        self.metric = metric

        # Optional: default slice export format; can be overridden externally (e.g., launcher)
        self._slice_fmt = "npy"

    def __str__(self):
        string = 'Model parameters:\n'
        string += '  root folder: ' + str(self.root_folder) + '\n'
        string += '  code folder: ' + str(self.code_folder) + '\n'
        string += '  image_folder: ' + str(self.image_folder) + '\n'
        string += '  mask folder: ' + str(self.mask_folder) + '\n'
        string += '  save folder: ' + str(self.save_folder) + '\n'
        string += '  training folder: ' + str(self.training_folder) + '\n'
        string += '  prediction folder: ' + str(self.pred_folder) + '\n'
        string += '  log file name: ' + str(self.logFileName) + '\n'
        string += '  testing mode: ' + str(self.testingMode) + '\n'
        string += '  testLabelFlag: ' + str(self.testLabelFlag) + '\n'
        string += '  testMetricFlag: ' + str(self.testMetricFlag) + '\n'
        string += '  dataAugmentation: ' + str(self.dataAugmentation) + '\n'
        string += '  savePredMask: ' + str(self.savePredMask) + '\n'
        string += '  augmentation fold: ' + str(self.afold) + '\n'
        string += '  number of epochs: ' + str(self.numEpochs) + '\n'
        string += '  output label: ' + str(self.oLabel) + '\n'
        string += '  checkWeightFileName: ' + str(self.checkWeightFileName) + '\n'
        string += '  data augmentation parameters:\n' + str(self.datagen) + '\n'
        string += '  dtype: ' + str(self.dtype) + '\n'
        string += '  dtypeL: ' + str(self.dtypeL) + '\n'
        string += '  wType: ' + str(self.wType) + '\n'
        string += '  input shape: ' + str([self.img_row, self.img_col, self.channel]) + '\n'
        string += '  classifier: ' + str(self.classifier) + '\n'
        string += '  batch size: ' + str(self.bs) + '\n'
        string += '  model: ' + str(self.model) + '\n'
        return string

    # -------------------- Path helpers --------------------
    def arrangeDataPath(self, data_folder, image_folder, mask_folder):
        """
        Arrange paths for training or prediction.
        Supports both absolute and relative paths.
        """
        # image folder
        if os.path.isabs(image_folder):
            train_images_path = image_folder
        else:
            if data_folder:
                train_images_path = os.path.join(self.root_folder, data_folder, image_folder)
            else:
                train_images_path = os.path.join(self.root_folder, image_folder)

        # mask folder (can be empty for predict)
        if not mask_folder:
            train_labels_path = ""
        elif os.path.isabs(mask_folder):
            train_labels_path = mask_folder
        else:
            if data_folder:
                train_labels_path = os.path.join(self.root_folder, data_folder, mask_folder)
            else:
                train_labels_path = os.path.join(self.root_folder, mask_folder)

        return train_images_path, train_labels_path

    def arrangeTestPath(self, data_folder, image_folder):
        """Arrange path for test images only."""
        if os.path.isabs(image_folder):
            return image_folder
        if data_folder:
            return os.path.join(self.root_folder, data_folder, image_folder)
        return os.path.join(self.root_folder, image_folder)

    # -------------------- IO & loaders --------------------
    def loadTrainingData(self, images_path, labels_path, each):
        images = nb.load(os.path.join(images_path, each)).get_fdata()
        labels = nb.load(os.path.join(labels_path, each)).get_fdata()
        affine = nb.load(os.path.join(images_path, each)).affine

        if self.dataAugmentation:
            images, labels = self.datagen.generate(images, labels, self.afold, affine)

        [self.img_rows, self.img_cols, self.numImgs] = images.shape
        images = images.transpose(2, 0, 1).reshape(self.numImgs, self.img_rows, self.img_cols, 1).astype(self.dtype)
        labels = labels.transpose(2, 0, 1).reshape(self.numImgs, self.img_rows, self.img_cols, 1).astype(self.dtypeL)
        return images, labels, affine

    def loadPredData(self, images_path, each):
        images = nb.load(os.path.join(images_path, each)).get_fdata()
        affine = nb.load(os.path.join(images_path, each)).affine
        [self.img_rows, self.img_cols, self.numImgs] = images.shape
        images = images.transpose(2, 0, 1).reshape(self.numImgs, self.img_rows, self.img_cols, 1).astype(self.dtype)
        return images, affine

    def load3DtrainingData(self, images_path, labels_path, each):
        images = nb.load(os.path.join(images_path, each)).get_fdata().astype('float32')
        labels = nb.load(os.path.join(labels_path, each)).get_fdata().astype('uint8')
        affine = nb.load(os.path.join(images_path, each)).affine
        ind = np.where(labels > 0)
        labels[ind] = 1

        if len(images.shape) > 3:
            images = images[:, :, :, 0]

        if self.dataAugmentation:
            images, labels = self.datagen.generate(images, labels, self.afold)
        [self.img_rows, self.img_cols, self.numImgs] = images.shape
        return images, labels, affine

    def load3DtestData(self, images_path, labels_path, each):
        images = nb.load(os.path.join(images_path, each)).get_fdata().astype('float32')
        affine = nb.load(os.path.join(images_path, each)).affine
        if len(images.shape) > 3:
            images = images[:, :, :, 0]

        if self.testLabelFlag and labels_path:
            labels = nb.load(os.path.join(labels_path, each)).get_fdata().astype('uint8')
            ind = np.where(labels > 0)
            labels[ind] = 1
        else:
            labels = []
        [self.img_rows, self.img_cols, self.numImgs] = images.shape
        return images, labels, affine

    def loadTestData(self, images_path, labels_path, each):
        images = nb.load(os.path.join(images_path, each)).get_fdata()
        affine = nb.load(os.path.join(images_path, each)).affine
        [self.img_rows, self.img_cols, self.numImgs] = images.shape
        images = images.transpose(2, 0, 1).reshape(self.numImgs, self.img_rows, self.img_cols, 1).astype(self.dtype)
        if self.testLabelFlag and labels_path:
            labels = nb.load(os.path.join(labels_path, each)).get_fdata()
            labels = labels.transpose(2, 0, 1).reshape(self.numImgs, self.img_rows, self.img_cols, 1).astype(self.dtypeL)
        else:
            labels = []
        return images, labels, affine

    # -------------------- Loss & metrics --------------------
    def dice(self, trueL, predL):
        smooth = 1
        trueLF = K.flatten(trueL)
        predLF = K.flatten(predL)
        intersection = K.sum(trueLF * predLF)
        dc = K.eval((2.0 * intersection + smooth) / (K.sum(trueLF) + K.sum(predLF) + smooth))
        print('dice index: ', dc)
        return dc

    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(self, trueL, predL):
        return -self.dice_coef(trueL, predL)

    # -------------------- Models --------------------
    def createModel(self):
        lr = self.lr
        decay = self.decay
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=lr)
            self.opt = optimizer
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=lr, decay=decay)
            self.opt = optimizer
        else:
            # default to Adam if not specified
            optimizer = Adam(learning_rate=lr)
            self.opt = optimizer

        base_model = Unet((self.img_row, self.img_col, self.channel), nb_classes=self.nb_classes)
        act1 = Activation(self.classifier)(base_model.output)
        new_output = Reshape((self.img_row * self.img_col, 1, self.nb_classes))(act1)
        top_model = Model(base_model.input, new_output)

        if self.loss == 'dice_loss':
            top_model.compile(optimizer=optimizer, loss=self.dice_loss, metrics=[self.dice_coef])
        else:
            top_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return top_model

    def createModel3D(self, size):
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr, decay=self.decay)
        else:
            optimizer = Adam(learning_rate=self.lr)

        base_model = Unet3D((size[0], size[1], size[2], self.channel))
        act1 = Activation(self.classifier)(base_model.output)
        new_output = Reshape((size[0] * size[1] * size[2], 1, self.nb_classes))(act1)
        top_model = Model(base_model.input, new_output)
        top_model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])
        return top_model

    # -------------------- Class weights --------------------
    def volumeBasedWeighting(self, trainLabels):
        ind2 = np.where(trainLabels == 1)
        ind1 = np.where(trainLabels == 0)
        ind3 = np.where(trainLabels == 2)
        ind4 = np.where(trainLabels == 3)

        wImg = np.zeros(trainLabels.shape, dtype=self.dtypeL)
        if self.nb_classes == 2:
            w1 = ((len(ind1[0])) * 1.0 / (self.numImgs * self.img_rows * self.img_cols)) + np.finfo('float').eps
            w2 = ((len(ind2[0])) * 1.0 / (self.numImgs * self.img_rows * self.img_cols)) + np.finfo('float').eps
            w1 = np.max([w1, w2]) / (w1 + np.finfo('float').eps)
            w2 = np.max([w1, w2]) / (w2 + np.finfo('float').eps)
            if len(ind1) != 0:
                wImg[ind1] = w1
            if len(ind2) != 0:
                wImg[ind2] = w2
        elif self.nb_classes == 3:
            w1 = ((len(ind1[0])) * 1.0 / (self.numImgs * self.img_rows * self.img_cols)) + np.finfo('float').eps
            w2 = ((len(ind2[0])) * 1.0 / (self.numImgs * self.img_rows * self.img_cols)) + np.finfo('float').eps
            w3 = ((len(ind3[0])) * 1.0 / (self.numImgs * self.img_rows * self.img_cols)) + np.finfo('float').eps
            w1 = np.max([w1, w2, w3]) / (w1 + np.finfo('float').eps)
            w2 = np.max([w1, w2, w3]) / (w2 + np.finfo('float').eps)
            w3 = np.max([w1, w2, w3]) / (w3 + np.finfo('float').eps)
            if len(ind1) != 0:
                wImg[ind1] = w1
            if len(ind2) != 0:
                wImg[ind2] = w2
            if len(ind3) != 0:
                wImg[ind3] = w3
        elif self.nb_classes == 4:
            w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
            w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
            w3 = ((len(ind3[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
            w4 = ((len(ind4[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
            w1 = np.max([w1, w2, w3, w4]) / (w1 + np.finfo('float').eps)
            w2 = np.max([w1, w2, w3, w4]) / (w2 + np.finfo('float').eps)
            w3 = np.max([w1, w2, w3, w4]) / (w3 + np.finfo('float').eps)
            w4 = np.max([w1, w2, w3, w4]) / (w4 + np.finfo('float').eps)
            if len(ind1) != 0:
                wImg[ind1] = w1
            if len(ind2) != 0:
                wImg[ind2] = w2
            if len(ind3) != 0:
                wImg[ind3] = w3
            if len(ind4) != 0:
                wImg[ind4] = w4
        return wImg

    def sliceBasedWeighting(self, trainLabels):
        wImg = np.zeros(trainLabels.shape, dtype=self.dtype)
        for i in range(wImg.shape[0]):
            if self.nb_classes == 2:
                ind2 = np.where(trainLabels[i, :] > 0)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w1 = np.max([w1, w2]) / w1
                w2 = np.max([w1, w2]) / w2
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
            elif self.nb_classes == 3:
                ind3 = np.where(trainLabels[i, :] == 2)
                ind2 = np.where(trainLabels[i, :] == 1)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w3 = ((len(ind3[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w1 = np.max([w1, w2, w3]) / (w1 + np.finfo('float').eps)
                w2 = np.max([w1, w2, w3]) / (w2 + np.finfo('float').eps)
                w3 = np.max([w1, w2, w3]) / (w3 + np.finfo('float').eps)
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
                if len(ind3) != 0:
                    wImg[i, ind3] = w3
            elif self.nb_classes == 4:
                ind4 = np.where(trainLabels[i, :] == 3)
                ind3 = np.where(trainLabels[i, :] == 2)
                ind2 = np.where(trainLabels[i, :] == 1)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w3 = ((len(ind3[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w4 = ((len(ind4[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w1 = np.max([w1, w2, w3, w4]) / (w1 + np.finfo('float').eps)
                w2 = np.max([w1, w2, w3, w4]) / (w2 + np.finfo('float').eps)
                w3 = np.max([w1, w2, w3, w4]) / (w3 + np.finfo('float').eps)
                w4 = np.max([w1, w2, w3, w4]) / (w4 + np.finfo('float').eps)
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
                if len(ind3) != 0:
                    wImg[i, ind3] = w3
                if len(ind4) != 0:
                    wImg[i, ind4] = w4
        return wImg

    def sliceBasedWeighting3D(self, trainLabels):
        wImg = np.zeros(trainLabels.shape, dtype=self.dtype)
        for i in range(wImg.shape[0]):
            if self.nb_classes == 2:
                ind2 = np.where(trainLabels[i, :] > 0)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (trainLabels.shape[1]) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (trainLabels.shape[1]) + np.finfo('float').eps)
                w1 = np.max([w1, w2]) / w1
                w2 = np.max([w1, w2]) / w2
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
            elif self.nb_classes == 3:
                ind3 = np.where(trainLabels[i, :] == 2)
                ind2 = np.where(trainLabels[i, :] == 1)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w3 = ((len(ind3[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w1 = np.max([w1, w2, w3]) / (w1 + np.finfo('float').eps)
                w2 = np.max([w1, w2, w3]) / (w2 + np.finfo('float').eps)
                w3 = np.max([w1, w2, w3]) / (w3 + np.finfo('float').eps)
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
                if len(ind3) != 0:
                    wImg[i, ind3] = w3
            elif self.nb_classes == 4:
                ind4 = np.where(trainLabels[i, :] == 3)
                ind3 = np.where(trainLabels[i, :] == 2)
                ind2 = np.where(trainLabels[i, :] == 1)
                ind1 = np.where(trainLabels[i, :] == 0)
                w1 = ((len(ind1[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w2 = ((len(ind2[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w3 = ((len(ind3[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w4 = ((len(ind4[0])) * 1.0 / (self.img_rows * self.img_cols) + np.finfo('float').eps)
                w1 = np.max([w1, w2, w3, w4]) / (w1 + np.finfo('float').eps)
                w2 = np.max([w1, w2, w3, w4]) / (w2 + np.finfo('float').eps)
                w3 = np.max([w1, w2, w3, w4]) / (w3 + np.finfo('float').eps)
                w4 = np.max([w1, w2, w3, w4]) / (w4 + np.finfo('float').eps)
                if len(ind1) != 0:
                    wImg[i, ind1] = w1
                if len(ind2) != 0:
                    wImg[i, ind2] = w2
                if len(ind3) != 0:
                    wImg[i, ind3] = w3
                if len(ind4) != 0:
                    wImg[i, ind4] = w4
        return wImg

    # -------------------- Train loops --------------------
    def train(self):
        model = self.createModel()

        print('-' * 30)
        print('Loading training data...')
        print('-' * 30)

        train_images_path, train_labels_path = self.arrangeDataPath(self.root_folder, self.image_folder, self.mask_folder)
        hist = {'acc': [], 'loss': []}
        for epochs in range(self.numEpochs):
            print('epochs: ', epochs)
            acc = 0
            loss = 0
            for each in os.listdir(train_images_path):
                if not (each.endswith('.nii') or each.endswith('.nii.gz')):
                    continue
                print('case: ', each)
                trainImages, trainLabels, affine = self.loadTrainingData(train_images_path, train_labels_path, each)
                print('training image shape:', trainImages.shape)
                trainLabels = trainLabels.reshape((self.numImgs, self.img_rows * self.img_cols))
                if self.wType == 'slice':
                    wImg = self.sliceBasedWeighting(trainLabels)
                else:
                    wImg = self.volumeBasedWeighting(trainLabels)

                trainLabels = to_categorical(trainLabels, self.nb_classes)
                trainLabels = trainLabels.reshape((self.numImgs, self.img_rows * self.img_cols, 1, self.nb_classes))
                trainLabels = trainLabels.astype(self.dtypeL)

                print('-' * 30)
                print('Training model...')
                print('-' * 30)

                history = model.fit(trainImages, trainLabels, batch_size=self.bs, epochs=1, verbose=1, sample_weight=wImg)

                # 'accuracy' key can be 'acc' in old keras; handle both
                acc_key = 'acc' if 'acc' in history.history else 'accuracy'
                acc = acc + history.history[acc_key][0]
                loss = loss + history.history['loss'][0]
            if ((epochs > 0) and ((epochs + 1) % 25) == 0):
                model.save_weights(os.path.join(self.save_folder, str(epochs + 1) + '_' + self.checkWeightFileName))
            n_cases = max(1, len([f for f in os.listdir(train_images_path) if f.endswith('.nii') or f.endswith('.nii.gz')]))
            hist['acc'].append(acc / n_cases)
            hist['loss'].append(loss / n_cases)
        np.save(self.save_folder + 'history.npy', hist)
        return

    def train3D(self):
        model = self.createModel3D([128, 128, 48])

        print('-' * 30)
        print('Loading training data...')
        print('-' * 30)

        train_images_path, train_labels_path = self.arrangeDataPath(self.root_folder, self.image_folder, self.mask_folder)
        hist = {'acc': [], 'loss': []}
        for epochs in range(self.numEpochs):
            print('epochs: ', epochs)
            acc = 0
            loss = 0
            for each in os.listdir(train_images_path):
                if not (each.endswith('.nii') or each.endswith('.nii.gz')):
                    continue
                print('case: ', each)
                trainImages, trainLabels, affine = self.load3DtrainingData(train_images_path, train_labels_path, each)
                trainImages = interp3D(trainImages, [0.25, 0.25, 1], cval=-1024)
                trainLabels = interp3D(trainLabels, [0.25, 0.25, 1], cval=0)
                trainImages, trainLabels = arrange3Ddata(trainImages, trainLabels, 48, self.dtype)

                [numImgs, img_rows, img_cols, img_dep, ch] = trainImages.shape
                print('training image shape:', trainImages.shape)
                trainLabels = trainLabels.reshape((numImgs, img_rows * img_cols * img_dep))
                if self.wType == 'slice':
                    wImg = self.sliceBasedWeighting3D(trainLabels)
                else:
                    wImg = np.ones(trainLabels.shape)

                trainLabels = to_categorical(trainLabels, self.nb_classes)
                trainLabels = trainLabels.reshape((numImgs, img_rows * img_cols * img_dep, 1, self.nb_classes))
                trainLabels = trainLabels.astype(self.dtype)

                print('-' * 30)
                print('Training model...')
                print('-' * 30)

                history = model.fit(trainImages, trainLabels, batch_size=self.bs, epochs=1, verbose=1, sample_weight=wImg)
                acc_key = 'acc' if 'acc' in history.history else 'accuracy'
                acc = acc + history.history[acc_key][0]
                loss = loss + history.history['loss'][0]
            if ((epochs > 0) and ((epochs + 1) % 25) == 0):
                model.save_weights(os.path.join(self.save_folder, str(epochs + 1) + '_' + self.checkWeightFileName))
            n_cases = max(1, len([f for f in os.listdir(train_images_path) if f.endswith('.nii') or f.endswith('.nii.gz')]))
            hist['acc'].append(acc / n_cases)
            hist['loss'].append(loss / n_cases)
        np.save(self.save_folder + 'history.npy', hist)
        return

    # -------------------- Metrics I/O --------------------
    def saveTestMetrics(self, saveFolder, testLabels, predImage, each):
        testDataMetrics = {}; DIL = []; caseList = []; accL = []; recallL = []; roc_aucL = []; cmL = []; precisionL = []; f1_scoreL = []
        y_trueL = testLabels.ravel()
        y_predL = predImage.ravel()
        acc = metrics.accuracy_score(y_trueL, y_predL)
        try:
            recall = metrics.recall_score(y_trueL, y_predL)
            roc_auc = metrics.roc_auc_score(y_trueL, y_predL)
            cm = metrics.confusion_matrix(y_trueL, y_predL)
            f1_score = metrics.f1_score(y_trueL, y_predL)
        except:
            recall = ''
            roc_auc = ''
            cm = ''
            f1_score = ''
        precision = metrics.precision_score(y_trueL, y_predL)
        DI = self.dice(y_trueL.astype(self.dtype), y_predL)

        caseList.append(each); accL.append(acc)
        recallL.append(recall); roc_aucL.append(roc_auc)
        cmL.append(cm); precisionL.append(precision)
        f1_scoreL.append(f1_score); DIL.append(DI)
        testDataMetrics['caseList'] = caseList
        testDataMetrics['acc'] = accL
        testDataMetrics['recall'] = recallL
        testDataMetrics['roc_auc'] = roc_aucL
        testDataMetrics['cm'] = cmL
        testDataMetrics['precision'] = precisionL
        testDataMetrics['f1_score'] = f1_scoreL
        testDataMetrics['DI'] = DIL
        if not os.path.lexists(saveFolder):
            os.mkdir(saveFolder)
        np.save(saveFolder + '/' + each, testDataMetrics)
        return

    def computeTestMetrics(self, testLabels, predImage):
        self.dice(testLabels.astype(self.dtype), predImage)
        return

    # -------------------- Utility helpers for slices --------------------
    def _pad_or_crop_to(self, vol, out_h, out_w, pad_value=0.0):
        """
        Center pad/crop each slice to (out_h, out_w).
        vol: (N, H, W, 1)
        """
        N, H, W, C = vol.shape
        assert C == 1
        oh, ow = int(out_h), int(out_w)
        out = np.full((N, oh, ow, 1), pad_value, dtype=vol.dtype)

        # source crop (center crop if src larger)
        h0 = max(0, (H - oh) // 2)
        w0 = max(0, (W - ow) // 2)
        h1 = h0 + min(H, oh)
        w1 = w0 + min(W, ow)

        # dest paste window
        dh0 = max(0, (oh - H) // 2)
        dw0 = max(0, (ow - W) // 2)
        dh1 = dh0 + (h1 - h0)
        dw1 = dw0 + (w1 - w0)

        out[:, dh0:dh1, dw0:dw1, 0] = vol[:, h0:h1, w0:w1, 0]
        return out

    def _ensure_dir(self, p):
        os.makedirs(p, exist_ok=True)

    def _window_and_normalize(self, img2d, hu_center=40.0, hu_width=80.0, out_dtype=np.uint8):
        """
        Windowing for CT HU and normalize to 0..255 (uint8) or keep float.
        """
        lo = hu_center - hu_width / 2.0
        hi = hu_center + hu_width / 2.0
        x = np.clip(img2d, lo, hi)
        x = (x - lo) / max(hi - lo, 1e-6)  # 0..1
        if out_dtype == np.uint8:
            x = (x * 255.0 + 0.5).astype(np.uint8)
        elif out_dtype == np.uint16:
            x = (x * 65535.0 + 0.5).astype(np.uint16)
        else:
            x = x.astype(out_dtype)
        return x

    def _save_slices(self,
                     case_id,
                     image_slices,   # (N, H, W, 1) float
                     mask_slices,    # (N, H, W, 1) {0,1}
                     out_root,
                     img_dirname="slices_img",
                     msk_dirname="slices_mask",
                     fmt="png",
                     hu_center=40.0,
                     hu_width=80.0):
        """
        Save per-slice image & mask.
        Output:
          <out_root>/<img_dirname>/<case_id>/<case_id>_zXXX.png (or .npy)
          <out_root>/<msk_dirname>/<case_id>/<case_id>_zXXX.png (or .npy)
        """
        img_out_dir = os.path.join(out_root, img_dirname, case_id)
        msk_out_dir = os.path.join(out_root, msk_dirname, case_id)
        self._ensure_dir(img_out_dir)
        self._ensure_dir(msk_out_dir)

        if fmt.lower() == "png":
            import imageio.v2 as imageio

        n = image_slices.shape[0]
        for i in range(n):
            img2d = image_slices[i, :, :, 0]
            msk2d = mask_slices[i, :, :, 0].astype(np.uint8)

            stem = f"{case_id}_z{i:03d}"
            if fmt.lower() == "png":
                img2d_vis = self._window_and_normalize(img2d, hu_center=hu_center, hu_width=hu_width, out_dtype=np.uint8)
                imageio.imwrite(os.path.join(img_out_dir, stem + ".png"), img2d_vis)
                imageio.imwrite(os.path.join(msk_out_dir, stem + ".png"), (msk2d * 255).astype(np.uint8))
            elif fmt.lower() == "npy":
                np.save(os.path.join(img_out_dir, stem + ".npy"), img2d)   # raw float
                np.save(os.path.join(msk_out_dir, stem + ".npy"), msk2d)   # 0/1
            else:
                raise ValueError(f"Unsupported slice format: {fmt}")
    def _pad_or_crop_3d_hw(self, vol, out_h, out_w, pad_value=-1024.0):
        """
        Center pad/crop a 3D volume (H, W, D) on spatial dims to (out_h, out_w, D).
        """
        import numpy as np
        assert vol.ndim == 3, f"Expected (H, W, D), got {vol.shape}"
        H, W, D = vol.shape
        oh, ow = int(out_h), int(out_w)
        out = np.full((oh, ow, D), pad_value, dtype=vol.dtype)

        # source crop window (center crop if src larger)
        h0 = max(0, (H - oh) // 2); h1 = h0 + min(H, oh)
        w0 = max(0, (W - ow) // 2); w1 = w0 + min(W, ow)

        # dest paste window
        dh0 = max(0, (oh - H) // 2); dh1 = dh0 + (h1 - h0)
        dw0 = max(0, (ow - W) // 2); dw1 = dw0 + (w1 - w0)

        out[dh0:dh1, dw0:dw1, :] = vol[h0:h1, w0:w1, :]
        return out

    def _ensure_channel_last_5d(self, x):
        """
        Ensure array shape is (N, H, W, D, 1).
        Accepts (N,H,W,D) and appends channel axis if missing.
        """
        import numpy as np
        if x.ndim == 5:
            return x
        if x.ndim == 4:
            return x[..., np.newaxis]
        raise ValueError(f"Expected 4D/5D array, got shape {x.shape}")


    # -------------------- Inference (2D) with optional per-slice export --------------------
    def Predict(self, weights, in_dir=None, out_dir=None):
        """
        2D inference with optional override for input/output directories.
        """
        # Allow CLI to override input/output
        if in_dir is not None:
            self.image_folder = in_dir
        if out_dir is not None:
            self.save_folder = out_dir

        # Resolve data paths
        test_images_path, test_labels_path = self.arrangeDataPath(
            self.root_folder, self.image_folder, self.mask_folder
        )

        print('-' * 30)
        print('Loading saved weights...')
        print('-' * 30)

        model = self.createModel()
        model.load_weights(weights)

        print("Input images path:", test_images_path)
        print("Output (save_folder):", self.save_folder)

        # Ensure slice root exists and tell user where slices go
        slice_root = os.path.join(self.save_folder, "slices")
        self._ensure_dir(slice_root)
        print("Per-slice export root:", slice_root)

        for each in sorted(os.listdir(test_images_path)):
            if not (each.endswith(".nii") or each.endswith(".nii.gz")):
                continue

            print('case: ', each)
            testImages, testLabels, affine = self.loadTestData(test_images_path, test_labels_path, each)

            # auto pad/crop to expected network input size
            if (testImages.shape[1] != self.img_row) or (testImages.shape[2] != self.img_col):
                testImages = self._pad_or_crop_to(testImages, self.img_row, self.img_col, pad_value=0.0)

            probs = model.predict(testImages, batch_size=8, verbose=1)

            print('-' * 30)
            print('Predicting masks on test data...')
            print('-' * 30)

            probs = probs.reshape((testImages.shape[0], self.img_row, self.img_col, self.nb_classes))
            pred_bin = (probs[:, :, :, self.sC - 1:self.sC] > 0.5).astype(np.uint8)  # (N,H,W,1)

            # save per-slice (png or npy)
            case_id = each[:-7] if each.endswith(".nii.gz") else each[:-4]
            self._save_slices(
                case_id=case_id,
                image_slices=testImages,           # original per-slice images (float)
                mask_slices=pred_bin,              # 0/1
                out_root=slice_root,
                img_dirname="slices_img",
                msk_dirname="slices_mask",
                fmt=getattr(self, "_slice_fmt", "npy"),
                hu_center=40.0, hu_width=80.0
            )

            print('pred slices shape: ', pred_bin.shape)

            saveFolder = os.path.join(self.save_folder)
            if self.testLabelFlag and isinstance(testLabels, np.ndarray) and testLabels != []:
                self.computeTestMetrics(testLabels, pred_bin)
            if self.testMetricFlag:
                self.saveTestMetrics(saveFolder, testLabels, pred_bin, each)
            if self.savePredMask:
                pred_nii = nb.Nifti1Image(pred_bin[:, :, :, 0].transpose(1, 2, 0), affine)
                nb.save(pred_nii, os.path.join(saveFolder, each))
        return

    # -------------------- Inference (3D) --------------------
    # -------------------- Inference (3D) for 48¡Á256¡Á256 --------------------
    def Predict3D(self, weights, in_dir=None, out_dir=None):

        # ---- override Â·¾¶ ----
        if in_dir is not None:
            self.image_folder = in_dir
        if out_dir is not None:
            self.save_folder = out_dir

        test_images_path, test_labels_path = self.arrangeDataPath(
            self.root_folder, self.image_folder, self.mask_folder
        )

        print('-' * 30)
        print('Loading saved weights...')
        print('-' * 30)


        vol_size = [self.img_row, self.img_col, 48]
        model = self.createModel3D(vol_size)

        if os.path.isabs(weights):
            model.load_weights(weights)
        else:
            model.load_weights(os.path.join(self.save_folder, weights))

        from datetime import datetime


        for each in sorted(os.listdir(test_images_path)):
            if not (each.endswith(".nii") or each.endswith(".nii.gz")):
                continue

            print('case: ', each)
            startTime = datetime.now()

            testImages, otestLabels, affine = self.load3DtestData(
                test_images_path, test_labels_path, each
            )
  
            testImages = self._pad_or_crop_3d_hw(
                testImages,
                self.img_row,
                self.img_col,
                pad_value=-1024.0
            )

            from load3Ddata import arrange3DtestImage  
            testImages = arrange3DtestImage(
                testImages,
                fold=48,
                dtype=self.dtype,
                pad_value=-1024
            )  # (N=1, H, W, 48, 1)
            testImages = self._ensure_channel_last_5d(testImages)

            numImgs, img_rows, img_cols, img_dep, ch = testImages.shape
            assert numImgs == 1, f"Expect single block, got {numImgs}"
            print('testing image shape:', testImages.shape)


            pred = model.predict(testImages, batch_size=1, verbose=1)
            print('-' * 30)
            print('Predicting 3D masks...')
            print('-' * 30)

            # pred: (1, img_rows*img_cols*img_dep, 1, nb_classes)
            pred = pred.reshape(
                (numImgs, img_rows, img_cols, img_dep, self.nb_classes)
            )


            pred = pred[0, :, :, :, self.sC - 1]   # shape: (H,W,D)
            pred_bin = (pred > 0.5).astype('uint8')

            print('pred volume shape: ', pred_bin.shape)
            print("Elapsed:", datetime.now() - startTime)

            saveFolder = os.path.join(self.save_folder, self.pred_folder)
            self._ensure_dir(saveFolder)


            if self.testLabelFlag and isinstance(otestLabels, np.ndarray) and otestLabels != []:

                self.computeTestMetrics(otestLabels, pred_bin)

            if self.testMetricFlag:
                self.saveTestMetrics(saveFolder, otestLabels, pred_bin, each)


            if self.savePredMask:
                pred_nii = nb.Nifti1Image(pred_bin.astype('uint8'), affine)
                nb.save(pred_nii, os.path.join(saveFolder, each))

        return
