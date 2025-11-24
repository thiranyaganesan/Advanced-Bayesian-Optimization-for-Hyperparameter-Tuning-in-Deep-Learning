from tensorflow.keras import layers, models, optimizers

def build_cnn(input_shape=(32,32,3), num_classes=10, params=None):
    """
    Build a small configurable CNN. `params` is a dict that may contain:
      - conv_filters: list of int, number of filters per conv block
      - kernel_size: int
      - dense_units: int
      - dropout_rate: float
      - lr: float (learning rate)
      - optimizer: 'adam' or 'sgd'
    """
    if params is None:
        params = {}
    conv_filters = params.get('conv_filters', [32, 64])
    kernel_size = params.get('kernel_size', 3)
    dense_units = params.get('dense_units', 128)
    dropout_rate = params.get('dropout_rate', 0.5)
    lr = params.get('lr', 1e-3)
    opt_name = params.get('optimizer', 'adam')

    inp = layers.Input(shape=input_shape)
    x = inp
    for f in conv_filters:
        x = layers.Conv2D(f, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)

    if opt_name == 'adam':
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
