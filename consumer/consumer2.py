import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, render_template, jsonify
from kafka import KafkaConsumer


import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import pandas as pd
import PIL
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

import time

sess = K.get_session()
discovery_logo = cv2.imread("loggosto.jpg")



# Kafka Topic
topic = "test"
W = 288
H = 512

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka:9093'])

consumer1 = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka:9093'])

consumer2 = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka:9093'])


consumer3 = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka:9093'])

# Set the consumer in a Flask App
app = Flask(__name__)


@app.route('/', methods=['GET'])
def Index():
    return render_template('index.html')

@app.route('/demo1', methods=['GET'])
def demo1():
    return render_template('demo1.html')

@app.route('/demo2', methods=['GET'])
def demo2():
    return render_template('demo2.html')

@app.route('/demo3', methods=['GET'])
def demo3():
    return render_template('demo3.html')

def gen(camera):
    while True:
        frame = camera.get_frame(0)
        yield (b'--frame\r\n'
               b'Content-Type: images/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(get_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_canny')
def video_canny():
    return Response(video_2(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_dl')
def video_dl():
    return Response(video_1(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_advertiser')
def video_advertiser():
    return Response(video_3(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mobile', methods=['GET'])
def mobile():
    return Response(get_mobile(),mimetype='multipart/x-mixed-replace; boundary=frame')

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .4):
    box_scores = np.multiply(box_confidence, box_class_probs)
    box_classes = K.argmax(box_scores, axis=-1)  # index
    box_class_scores = K.max(box_scores, axis=-1)  # score of corresponding index
    filtering_mask = K.greater_equal(box_class_scores, threshold)  # create a bool mask like [0,1,1,0,0,0,0,1,....]
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.50):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (288.0, 512.0), max_boxes=20, score_threshold=.4, iou_threshold=.50):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)

    return scores, boxes, classes

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (288.0, 512.0)

yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def preprocess_image(img_itself, model_image_size):
    image = Image.fromarray(img_itself)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def predict(sess, image_file):
    
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data})#, K.learning_phase():0})
    
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    return np.asarray(image)

def predict_replace(sess, image_file, replacer):
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data})
    image = np.asarray(image)
    replacer = np.asarray(replacer)
    indices = (np.array(out_classes > 49).nonzero()[0])
    image_copy = image.copy()
    if(indices.size > 0):
        x_offset = int(out_boxes[indices[0]][1])
        y_offset = int(out_boxes[indices[0]][0])
        y_offset_ex = int(out_boxes[indices[0]][2])
        x_offset_ex = int(out_boxes[indices[0]][3])        
        box_width = abs(x_offset - x_offset_ex)
        box_height = abs(y_offset_ex - y_offset)
        replacer = cv2.resize(replacer,(box_width , box_height))        
        image_copy[y_offset:y_offset+replacer.shape[0], x_offset:x_offset+replacer.shape[1]] = replacer.copy()
    return np.asarray(image_copy)

def get_stream():
    print('Listening...')
    for msg in consumer:
        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg.tobytes()+ b'\r\n\r\n')

def video_2():
    print('Listening...')
    for msg in consumer2:
        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 100,200)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg.tobytes()+ b'\r\n\r\n')


def video_1():
    print('Listening...')
    
    counter = 1
   
    for msg in consumer1:
        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)
   
        #if (counter % 3 == 1):
        jpeg = predict(sess, frame)
        ret, jpeg = cv2.imencode('.jpg', jpeg)
        #else:
        #    ret, jpeg = cv2.imencode('.jpg', frame)
    
        #counter = counter + 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg.tobytes()+ b'\r\n\r\n')

def video_3():
    for msg in consumer1:
        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)
        jpeg = predict_replace(sess, frame, replacer=discovery_logo)
        ret, jpeg = cv2.imencode('.jpg', jpeg)
    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + jpeg.tobytes()+ b'\r\n\r\n')

def get_mobile():
    print('Listening...')
    for msg in consumer3:
        frame = cv2.imdecode(np.frombuffer(msg.value, np.uint8), -1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5001', debug=True, threaded=True)
