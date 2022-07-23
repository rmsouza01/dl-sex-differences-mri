import numpy as np
import tensorflow as tf
import os
import nibabel as nib
import glob

# Loads and scales Nifti images
def load_nifti(folder_path, files, H, W, Z):
    X = np.zeros((files.size, H, W, Z), dtype=np.float32)
    Y = np.zeros(files.size, dtype=int)

    for ii in range(files.size):
        aux = glob.glob(os.path.join(folder_path, "*" + files[ii] + "*"))[0]
        vol = nib.load(aux).get_fdata()
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        X[ii] = vol
        Y[ii] = files[ii].split("_")[-1] == "M"
    return X, Y


# Loads train, validation and test sets
def load_data_3d(folder_path, train, val, test):
    train_files = np.loadtxt(train, dtype=str)
    val_files = np.loadtxt(val, dtype=str)
    test_files = np.loadtxt(test, dtype=str)

    # Get data dimensions
    aux = glob.glob(os.path.join(folder_path, "*" + train_files[0] + "*"))[0]
    H, W, Z = nib.load(aux).shape

    # Load train/val/ test sets
    Xtrain, Ytrain = load_nifti(folder_path, train_files, H, W, Z)
    Xval, Yval = load_nifti(folder_path, val_files, H, W, Z)
    Xtest, Ytest = load_nifti(folder_path, test_files, H, W, Z)

    return (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest)


# Loads train, validation and test sets
def load_data_3d_age(folder_path, train, val, test, norm=100.0):
    train_files = np.loadtxt(train, dtype=str)
    val_files = np.loadtxt(val, dtype=str)
    test_files = np.loadtxt(test, dtype=str)

    # Get data dimensions
    aux = glob.glob(os.path.join(folder_path, "*" + train_files[0] + "*"))[0]
    H, W, Z = nib.load(aux).shape

    # Load train/val/ test sets
    Xtrain, Ytrain = load_nifti(folder_path, train_files, H, W, Z)
    age_train = [int(f.split("_")[-2]) for f in train_files]
    age_train = np.array(age_train) / norm

    Xval, Yval = load_nifti(folder_path, val_files, H, W, Z)
    age_val = [int(f.split("_")[-2]) for f in val_files]
    age_val = np.array(age_val) / norm

    Xtest, Ytest = load_nifti(folder_path, test_files, H, W, Z)
    age_test = [int(f.split("_")[-2]) for f in test_files]
    age_test = np.array(age_test) / norm

    return (Xtrain, age_train, Ytrain), (Xval, age_val, Yval), (Xtest, age_test, Ytest)


def vgg_like_3d(ishape=(193, 229, 193)):
    """
    VGG like 3D model for image classification.
    :param ishape: Input shape
    :return: Tensorflow model
    """
    input_image = tf.keras.layers.Input(shape=(ishape[0], ishape[1], ishape[2], 1))
    conv1 = tf.keras.layers.Conv3D(30, kernel_size=(3, 3, 3), activation="relu")(
        input_image
    )
    conv3 = tf.keras.layers.Conv3D(
        30, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv1)
    conv3_drop = tf.keras.layers.Dropout(0.1)(conv3)
    conv4 = tf.keras.layers.Conv3D(60, kernel_size=(3, 3, 3), activation="relu")(
        conv3_drop
    )
    conv5 = tf.keras.layers.Conv3D(
        60, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv4)
    conv5_drop = tf.keras.layers.Dropout(0.1)(conv5)

    conv6 = tf.keras.layers.Conv3D(120, kernel_size=(3, 3, 3), activation="relu")(
        conv5_drop
    )
    conv7 = tf.keras.layers.Conv3D(
        120, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv6)
    conv7_drop = tf.keras.layers.Dropout(0.1)(conv7)
    conv8 = tf.keras.layers.Conv3D(240, kernel_size=(3, 3, 3), activation="relu")(
        conv7_drop
    )
    conv9 = tf.keras.layers.Conv3D(
        240, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv8)
    conv9_drop = tf.keras.layers.Dropout(0.1)(conv9)

    flat = tf.keras.layers.Flatten()(conv9_drop)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.models.Model(inputs=input_image, outputs=out)

    return model


def vgg_like_3d_age(ishape=(193, 229, 193)):
    """
    VGG like 3D model for image classification.
    :param ishape: Input shape
    :return: Tensorflow model
    """
    input_image = tf.keras.layers.Input(shape=(ishape[0], ishape[1], ishape[2], 1))
    age = tf.keras.layers.Input(shape=(1,))

    conv1 = tf.keras.layers.Conv3D(30, kernel_size=(3, 3, 3), activation="relu")(
        input_image
    )
    conv3 = tf.keras.layers.Conv3D(
        30, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv1)
    conv3_drop = tf.keras.layers.Dropout(0.1)(conv3)
    conv4 = tf.keras.layers.Conv3D(60, kernel_size=(3, 3, 3), activation="relu")(
        conv3_drop
    )
    conv5 = tf.keras.layers.Conv3D(
        60, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv4)
    conv5_drop = tf.keras.layers.Dropout(0.1)(conv5)

    conv6 = tf.keras.layers.Conv3D(120, kernel_size=(3, 3, 3), activation="relu")(
        conv5_drop
    )
    conv7 = tf.keras.layers.Conv3D(
        120, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv6)
    conv7_drop = tf.keras.layers.Dropout(0.1)(conv7)
    conv8 = tf.keras.layers.Conv3D(240, kernel_size=(3, 3, 3), activation="relu")(
        conv7_drop
    )
    conv9 = tf.keras.layers.Conv3D(
        240, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
    )(conv8)
    conv9_drop = tf.keras.layers.Dropout(0.1)(conv9)

    flat = tf.keras.layers.Flatten()(conv9_drop)
    concat = tf.keras.layers.Concatenate()([flat, age])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(concat)

    model = tf.keras.models.Model(inputs=[input_image, age], outputs=out)

    return model


def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr


def train(model, Xt, Yt, Xv, Yv, nepochs, batch_size, model_file, log_file):

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    monitor = tf.keras.callbacks.ModelCheckpoint(
        model_file,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    csv_logger = tf.keras.callbacks.CSVLogger(log_file, separator=",", append=True)

    model.fit(
        Xt,
        Yt,
        batch_size=batch_size,
        epochs=nepochs,
        verbose=1,
        callbacks=[early_stop, monitor, lr_schedule, csv_logger],
        validation_data=(Xv, Yv),
        shuffle=True,
    )

    return


def train_age(
    model, Xt, aget, Yt, Xv, agev, Yv, nepochs, batch_size, model_file, log_file
):

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    monitor = tf.keras.callbacks.ModelCheckpoint(
        model_file,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    csv_logger = tf.keras.callbacks.CSVLogger(log_file, separator=",", append=True)

    model.fit(
        [Xt, aget],
        Yt,
        batch_size=batch_size,
        epochs=nepochs,
        verbose=1,
        callbacks=[early_stop, monitor, lr_schedule, csv_logger],
        validation_data=([Xv, agev], Yv),
        shuffle=True,
    )

    return


def test(model, Xtest, Ytest, model_name):
    model.load_weights(model_name)
    results = model.evaluate(Xtest, Ytest, batch_size=1)
    return results


def test_age(model, Xtest, aget, Ytest, model_name):
    model.load_weights(model_name)
    results = model.evaluate([Xtest, aget], Ytest, batch_size=1)
    return results


def run_cross_validation(
    images_path, folds_path, out_path, train_log, best_model, lr, epochs, batch_size
):
    train_folds = [
        "train_1.txt",
        "train_2.txt",
        "train_3.txt",
        "train_4.txt",
        "train_5.txt",
        "train_6.txt",
        "train_7.txt",
        "train_8.txt",
        "train_9.txt",
        "train_10.txt",
    ]
    train_folds = [os.path.join(folds_path, f) for f in train_folds]
    val_folds = [
        "val_1.txt",
        "val_2.txt",
        "val_3.txt",
        "val_4.txt",
        "val_5.txt",
        "val_6.txt",
        "val_7.txt",
        "val_8.txt",
        "val_9.txt",
        "val_10.txt",
    ]

    val_folds = [os.path.join(folds_path, f) for f in val_folds]

    test_folds = [
        "test_1.txt",
        "test_2.txt",
        "test_3.txt",
        "test_4.txt",
        "test_5.txt",
        "test_6.txt",
        "test_7.txt",
        "test_8.txt",
        "test_9.txt",
        "test_10.txt",
    ]

    test_folds = [os.path.join(folds_path, f) for f in test_folds]

    log_file = os.path.join(out_path, train_log)

    for ii in range(len(train_folds)):
        model_name = os.path.join(out_path, best_model + str(ii + 1) + ".h5")
        (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = load_data_3d(
            images_path, train_folds[ii], val_folds[ii], test_folds[ii]
        )
        indexes = np.arange(Xtrain.shape[0], dtype=int)
        np.random.shuffle(indexes)
        Xtrain = Xtrain[indexes]
        Ytrain = Ytrain[indexes]

        model = vgg_like_3d(Xtrain.shape[1:])

        if ii == 0:
            print(model.summary())
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        train(
            model, Xtrain, Ytrain, Xval, Yval, epochs, batch_size, model_name, train_log
        )
        results = test(model, Xtest, Ytest, model_name)
        print(results)
    return


def run_cross_validation_age(
    images_path, folds_path, out_path, train_log, best_model, lr, epochs, batch_size
):
    train_folds = [
        "train_1.txt",
        "train_2.txt",
        "train_3.txt",
        "train_4.txt",
        "train_5.txt",
        "train_6.txt",
        "train_7.txt",
        "train_8.txt",
        "train_9.txt",
        "train_10.txt",
    ]
    train_folds = [os.path.join(folds_path, f) for f in train_folds]
    val_folds = [
        "val_1.txt",
        "val_2.txt",
        "val_3.txt",
        "val_4.txt",
        "val_5.txt",
        "val_6.txt",
        "val_7.txt",
        "val_8.txt",
        "val_9.txt",
        "val_10.txt",
    ]

    val_folds = [os.path.join(folds_path, f) for f in val_folds]

    test_folds = [
        "test_1.txt",
        "test_2.txt",
        "test_3.txt",
        "test_4.txt",
        "test_5.txt",
        "test_6.txt",
        "test_7.txt",
        "test_8.txt",
        "test_9.txt",
        "test_10.txt",
    ]

    test_folds = [os.path.join(folds_path, f) for f in test_folds]

    log_file = os.path.join(out_path, train_log)

    for ii in range(len(train_folds)):
        model_name = os.path.join(out_path, best_model + str(ii + 1) + ".h5")
        (
            (Xtrain, age_train, Ytrain),
            (Xval, age_val, Yval),
            (Xtest, age_test, Ytest),
        ) = load_data_3d_age(
            images_path, train_folds[ii], val_folds[ii], test_folds[ii]
        )
        indexes = np.arange(Xtrain.shape[0], dtype=int)
        np.random.shuffle(indexes)
        Xtrain = Xtrain[indexes]
        Ytrain = Ytrain[indexes]
        age_train = age_train[indexes]
        model = vgg_like_3d_age(Xtrain.shape[1:])

        if ii == 0:
            print(model.summary())
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        train_age(
            model,
            Xtrain,
            age_train,
            Ytrain,
            Xval,
            age_val,
            Yval,
            epochs,
            batch_size,
            model_name,
            train_log,
        )
        results = test_age(model, Xtest, age_test, Ytest, model_name)
        print(results)
    return
