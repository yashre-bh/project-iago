import random
from flask import Flask, render_template, Response, request
from flask import stream_with_context
from flask_bootstrap import Bootstrap
from camera import *
from recognize_gesture import * 


app = Flask(__name__)
Bootstrap(app)


@app.route("/")
def home_page():
    TITLE = 'Project IAGO'
    return render_template("home_page.html",TITLE=TITLE)

def gen(camera):
    while True:
        frame = camera.get_frame()
        frame_processed = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n')
        keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
        recognize(frame)


@app.route('/video_feed')
def video_feed():
    return Response((gen(
        Camera())
    ),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictive_text', methods = ["GET"])
def predictive_text():
    if request.method == "GET":
        return Response((gen(
        Camera())
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame')
        



if __name__ == '__main__':
    app.run(port= 5001, debug=True)
