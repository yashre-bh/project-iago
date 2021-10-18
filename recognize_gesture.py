import cv2, pickle
import numpy as np
import tensorflow as tf
from PIL import Image as im
import os
import sqlite3
from cnn_tf import cnn_model_fn
from keras.models import load_model
from keras import backend 
from keras.backend import set_session 
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from dotenv import load_dotenv

load_dotenv()
azure_key = os.getenv('AZURE_KEY')
azure_region = os.getenv('AZURE_REGION')



session = backend.get_session()
init = tf.global_variables_initializer()
session.run(init)
speech_config = SpeechConfig(subscription=azure_key, region=azure_region)
speech_config.speech_synthesis_language = "en-GB"
speech_config.speech_synthesis_voice_name ="en-GB-SoniaNeural"
audio_config = AudioOutputConfig(use_default_speaker=True)
synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
classifier = tf.estimator.Estimator(model_dir="tmp/cnn_model2", model_fn=cnn_model_fn)
prediction = None


config=tf.compat.v1.ConfigProto()

# # tf_config = config
session = tf.compat.v1.Session()

set_session(session)
graph = tf.get_default_graph()
model = tf.python.keras.models.load_model('cnn_model_keras2.h5')
model._make_predict_function()
# tf.reset_default_graph()
# graph = tf.get_default_graph()	

tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def tf_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	np_array = np.array(img)
	return np_array

def tf_predict(classifier, image):
	'''
	need help with prediction using tensorflow
	'''
	global prediction
	processed_array = tf_process_image(image)
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":processed_array}, shuffle=False)
	pred = classifier.predict(input_fn=pred_input_fn)
	prediction = next(pred)
	print(prediction)

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	global graph
	global session
	

	with graph.as_default():
		set_session(session)
		pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize(frame):
	global prediction
	hist = get_hand_hist()
	x, y, w, h = 300, 100, 300, 300
	text = ""
	img = frame
	img = cv2.flip(img, 1)
	img = cv2.resize(img, (640, 480))
	imgCrop = img[y:y+h, x:x+w]
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	(openCV_ver,_,__) = cv2.__version__.split(".")
	if openCV_ver=='3':
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
	elif openCV_ver=='4':
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	if len(contours) > 0:
		contour = max(contours, key = cv2.contourArea)
		if cv2.contourArea(contour) > 10000:	
			x1, y1, w1, h1 = cv2.boundingRect(contour)
			save_img = thresh[y1:y1+h1, x1:x1+w1]
			
			if w1 > h1:
				save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
			elif h1 > w1:
				save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
			
			pred_probab, pred_class = keras_predict(model, save_img)
			
			if pred_probab*100 > 80:
				text = get_pred_text_from_db(pred_class)
				synthesizer.speak_text_async("2")
				print(text)
		