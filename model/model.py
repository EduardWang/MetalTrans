import tensorflow as tf
from Encoder import Encoder

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (
                tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return binary_focal_loss_fixed


def get_model():

    inputESM = tf.keras.layers.Input(shape=(161, 1280))
    inputProt = tf.keras.layers.Input(shape=(161, 1024))
    sequence = tf.keras.layers.Dense(512)(inputESM)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence = sequence[:,80,:]
    sequence_prot = tf.keras.layers.Dense(512)(inputProt)
    sequence_prot=tf.keras.layers.Dense(256)(sequence_prot)
    sequence_prot = Encoder(2, 256, 4, 1024, rate=0.3)(sequence_prot)
    Prot=sequence_prot[:,80,:]
    sequenceconcat = tf.keras.layers.Concatenate()([sequence, Prot])
    feature = tf.keras.layers.Dense(512, activation='relu')(sequenceconcat)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)
    qa_model = tf.keras.models.Model(inputs=[inputESM, inputProt], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    qa_model.compile(loss=[binary_focal_loss(alpha=.26, gamma=2)], optimizer=adam, metrics=['accuracy'])
    qa_model.summary()
    return qa_model
