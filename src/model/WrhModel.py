import torch
path = "C:/Users/90430/Desktop/crnn-audio-classification-master/crnn.cfg"
file = open(path)
print(file.read())
exit()

'''
def cnn(x_train, y_train, x_test, y_test):
    learning_rate = 0.01
    batch_size = 100

    inputs = tf.keras.Input(shape=(len(x_train[0]), len(x_train[0][0]), 1))

    x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    models = tf.keras.Model(inputs=inputs, outputs=predictions)

    models.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = models.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_test, y_test))
    return history
'''
