import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, render_template, jsonify
from kafka import KafkaConsumer
# Kafka Topic
topic = "test"
W = 288
H = 512

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka:9093'])

# Set the consumer in a Flask App
app = Flask(__name__)

@app.route('/', methods=['GET'])
def Index():
    """
    This is the heart of our video display. Notice we set the mimetype to
    multipart/x-mixed-replace. This tells Flask to replace any old images with
    new values streaming through the pipeline.
    """
    return render_template('index.html')
   # return Response(
    #    get_stream(),
     #   mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame(0)
        yield (b'--frame\r\n'
               b'Content-Type: images/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(get_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

def get_stream():
    print('Listening...')

    for msg in consumer:

        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)

        # Convert the images from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_frame = frame[:, :, ::-1]
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.Canny(frame, 100,200)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg.tobytes()+ b'\r\n\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5001', debug=False)
