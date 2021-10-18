# Project IAGO
A Sign-Language interpreter developed for the Microsoft Student's Hackathon 2021.

### Intsructions
1. Create a vitual environment using `Python 3.7`.
2. Install the required dependencies using `pip3 install -r requirements.txt`.
3. Set your hand histogram by running ```python3 set_hand_hist.py```
4. You may use `create_gestures.py` to add create new custom gestures.
5. After this, images need to be loaded using ```python load_images.py```
6. The model can be trained using ```python cnn_keras.py```
7. All set! Run `server.py` and log in to `http://localhost:5001/` to test the server!

### Model Details
A convolutional neural network has been used to train the model and obtain predictions. `Keras` library of python has been used to train it, on a live video feed. More details are present in the `model details` folder.

### Team Members
1. [Ayan Choudhary](https://github.com/ayanchoudhary)
2. [Gaurav Genani](https://github.com/h3llix)
3. [Nishanth R](https://github.com/AllMight2099)
4. [Stuti Lilani](https://github.com/stuts23)
5. [Reeshita Paul](https://github.com/reeshi-18)
6. [Shreya Bhagat](https://github.com/yashre-bh)



