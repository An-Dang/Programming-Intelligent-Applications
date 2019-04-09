from tensorflow.keras.datasets.mnist import load_data
from tensorflow.layers import Flatten
import tensorflow as tf

(train_X, train_y),(test_X, test_y) = load_data()

# mlp for Multi Layer Perceptron
with tf.variable_scope("mlp") as mlp_scope:
    x = tf.placeholder(tf.float32, shape=(None, 28, 28), name="x")
    y_target = tf.placeholder(tf.uint8, shape=(None), name="y_target")

    # Preprocess x
    x_flattened = Flatten()(x)
    x_normalized = x_flattened / 255.0

    w1 = tf.Variable(tf.random_normal(shape=(784, 100)), name="w1")
    b1 = tf.Variable(tf.zeros(shape=(100)), name="b1")

    # h1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
    h1 = tf.nn.relu((x_normalized @ w1) + b1)

    w2 = tf.Variable(tf.random_normal(shape=(100, 10)), name="w2")
    b2 = tf.Variable(tf.zeros(shape=(10)), name="b2")

    # logits=tf.add(tf.matmul(h1,w2),b2)
    logits = (h1 @ w2) + b2
    y_pred = tf.nn.softmax(logits)

    learn_rate = tf.placeholder(tf.float32, shape=None, name="learn_rate")

# One-hot encoding
y_target_onehot = tf.one_hot(y_target, 10)

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target_onehot, logits=logits))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_target_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000
batch_size = 100

dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
dataset = dataset.repeat().shuffle(len(train_X)).batch(batch_size)

iterator = dataset.make_one_shot_iterator()
nextbatch = iterator.get_next()

tf.set_random_seed(1234)

for i in range(epochs):
    image_batch, label_batch = sess.run(nextbatch)
    sess.run(optimizer, feed_dict={x: image_batch, y_target: label_batch, learn_rate: 0.5})

    if ((i % 1000) == 0):
        # .eval(session=sess) same as sess.run()
        test_acc = accuracy.eval(feed_dict={x: test_X, y_target: test_y}, session=sess)
        print(test_acc)

sess.close()
