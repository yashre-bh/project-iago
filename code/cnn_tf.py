import tensorflow as tf
import numpy as np
import pickle, os, cv2


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

def cnn_model_fn(features, labels, mode):
	input_layer = tf.compat.v1.reshape(features["x"], [-1, image_x, image_y, 1], name="input")

	conv1 = tf.compat.v1.layers.conv2d(
	  inputs=input_layer,
	  filters=16,
	  kernel_size=[2, 2],
	  padding="same",
	  activation=tf.compat.v1.nn.relu,
	  name="conv1")
	print("conv1",conv1.shape)
	pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
	print("pool1",pool1.shape)

	conv2 = tf.compat.v1.layers.conv2d(
	  inputs=pool1,
	  filters=32,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.compat.v1.nn.relu,
	  name="conv2")
	print("conv2",conv2.shape)
	pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5, name="pool2")
	print("pool2",pool2.shape)

	conv3 = tf.compat.v1.layers.conv2d(
	  inputs=pool2,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.compat.v1.nn.relu,
	  name="conv3")
	print("conv3",conv3.shape)

	# Dense Layer
	flat = tf.compat.v1.reshape(conv3, [-1, 5*5*64], name="flat")
	print(flat.shape)
	dense = tf.compat.v1.layers.dense(inputs=flat, units=128, activation=tf.compat.v1.nn.relu, name="dense")
	print(dense.shape)
	dropout = tf.compat.v1.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.compat.v1.estimator.ModeKeys.TRAIN, name="dropout")

	# Logits Layer
	num_of_classes = get_num_of_classes()
	logits = tf.compat.v1.layers.dense(inputs=dropout, units=num_of_classes, name="logits")

	output_class = tf.compat.v1.argmax(input=logits, axis=1, name="output_class")
	output_probab = tf.compat.v1.nn.softmax(logits, name="softmax_tensor")
	predictions = {"classes": tf.compat.v1.argmax(input=logits, axis=1), "probabilities": tf.compat.v1.nn.softmax(logits, name="softmax_tensor")}
	if mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
		return tf.compat.v1.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.compat.v1.one_hot(indices=tf.compat.v1.cast(labels, tf.compat.v1.int32), depth=num_of_classes)
	loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
		train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
		return tf.compat.v1.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.compat.v1.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)
	#print(len(train_images[1]), len(train_labels))

	classifier = tf.compat.v1.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/cnn_model3")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.compat.v1.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": train_images}, y=train_labels, batch_size=500, num_epochs=10, shuffle=True)
	classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
	  x={"x": test_images},
	  y=test_labels,
	  num_epochs=1,
	  shuffle=False)
	test_results = classifier.evaluate(input_fn=eval_input_fn)
	print(test_results)


if __name__ == "__main__":
	tf.compat.v1.app.run()