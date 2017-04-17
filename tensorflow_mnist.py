import tensorflow as tf

# Get the input data which would be required for training and testing
from tensorflow.examples.tutorials.mnist import input_data

data_set = input_data.read_data_sets('.', one_hot=True, reshape=False)

# selecting the initial hyperparameters
epochs = 100
batch_size = 128
learning_rate = 0.001
display_step = 1

# parameters with respect to data
n_input = 784
n_hidden = 256
n_labels = 10

weights = {
    'i_to_h': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'h_to_o': tf.Variable(tf.random_normal([n_hidden, n_labels]))
}
biases = {
    'i_to_h': tf.Variable(tf.random_normal([n_hidden])),
    'h_to_o': tf.Variable(tf.random_normal([n_labels]))
}

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_labels])

# Flattens the matrix into rows of 784 elements(column) for each picture
x_flat = tf.reshape(x, [-1, n_input])
hidden_layer = tf.add(tf.matmul(x_flat, weights['i_to_h']), biases['i_to_h'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.add(
    tf.matmul(hidden_layer, weights['h_to_o']), biases['h_to_o']
)

# Finding the cost and reducing it using the gradient descent
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost)

prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_batch = int(data_set.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = data_set.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if epoch % 10 == 0:
            valid_accuracy = sess.run(accuracy, feed_dict={
                x: data_set.validation.images,
                y: data_set.validation.labels
            })
            print('Epoch: {} with accuracy {}'.format(
                epoch, valid_accuracy * 100)
            )

    test_accuracy = sess.run(accuracy, feed_dict={
                x: data_set.test.images, y: data_set.test.labels
    })
    print('Test accuracy is {}'.format(test_accuracy * 100))
