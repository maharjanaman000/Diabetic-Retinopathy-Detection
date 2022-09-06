# Modules
import streamlit as st
from pyrebase import pyrebase
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py

# Configuration Key Database ma store garna

firebaseConfig = {
  'apiKey': "AIzaSyCo1qF0CyIFGH9EYS_zYah02FcuecE3MRU",
  'authDomain': "webapp-bb7d9.firebaseapp.com",
  'projectId': "webapp-bb7d9",
  'databaseURL': "https://webapp-bb7d9-default-rtdb.asia-southeast1.firebasedatabase.app/",
  'storageBucket':"webapp-bb7d9.appspot.com",
  'messagingSenderId': "671720314696",
  'appId': "1:671720314696:web:43f2a2c86e059a1445457d"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.title('Diabetic Retinopathy System')
st.sidebar.title('User Authentication')
st.image('1.png')

# Authentication

choice = st.sidebar.selectbox('Login/Signup',['Login','Sign Up'])

# User ko Input 

email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

# Sign Up Block

if choice == 'Sign Up':
	handle = st.sidebar.text_input('Please enter your Username')
	submit = st.sidebar.button('Create My Account')

	if submit:
		user = auth.create_user_with_email_and_password(email,password)
		st.success('Your account has been created successfully')

		# Sign in
		user = auth.sign_in_with_email_and_password(email,password)
		db.child(user['localId']).child("Name").set(handle) 
		db.child(user['localId']).child("ID").set(user['localId']) 
		st.title('Welcome ' +  handle)
		st.info('Login via dropdown Login Option')

#Login Block
if choice == 'Login':
	login = st.sidebar.checkbox('Login')
	if login:
		user = auth.sign_in_with_email_and_password(email,password)
		st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
		bio = st.radio('Pages',['Home','Detection'])
#Home Page
		if bio == 'Home':
			st.header('What is Diabetic Retinopathy?')
			st.image("dr.png")
			st.write('''Diabetic Retinopathy is a diabetes complication that affects the eyes and is caused by damage to the blood vessels of the 
			light-sensitive tissue at the back of the eye, especially the retina. Manual detection of diabetic retinopathy takes a long time, 
			and patients must endure discomfort during this period. An automated system can aid in the rapid detection of diabetic retinopathy, 
			allowing the patient to easily follow-up treatment to avoid further eye damage. The main aim of this project is to develop a 
			web application embedding trained machine learning that can be used to detect diabetic retinopathy by analyzing the captured retina images of the patient.
			''')
		# Detection Page
		elif bio == 'Detection':
			st.title('Detect Diabetic Condition')
			def main():
				file_uploaded = st.file_uploader("Choose the file", type=['jpg','png','jpeg'])
				if file_uploaded is not None:
					image = Image.open(file_uploaded)
					figure = plt.figure()
					plt.imshow(image)
					plt.axis('off')
					result = predict_class(image)
					st.write(result)
					st.pyplot(figure)
			def predict_class(image):
				classifier_model = tf.keras.models.load_model(r'/Users/nishabaruwal/Desktop/Aman Folder Final/saved_model/dr_detection.hdf5')
				shape = ((224,224,3))
				model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
				test_image = image.resize((224,224))
				test_image = preprocessing.image.img_to_array(test_image)
				test_image = test_image/255.0
				test_image = np.expand_dims(test_image, axis = 0)
				class_names = ['Dr',
				               'Mild',
							   'Moderate',
							   'No_Dr',
							   'Severe'
				]
				predictions = model.predict(test_image)
				scores = tf.nn.softmax(predictions[0])
				scores = scores.numpy()
				image_class = class_names[np.argmax(scores)]
				result = "The image uploaded is in the condition of: {}".format(image_class)
				return result
			if __name__ =="__main__":
				main()
    
    

		
		
		

		
	


								
								