import argparse
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from object_tracker_utils.centroidtracker import CentroidTracker

# MQTT setup (Creating a Client Instance)
broker = 'test.mosquitto.org'
client = mqtt.Client('python_pub')


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f'MQTT client is CONNECTED to MQTT broker')
        client.subscribe(topic='video/+/peoplecount', qos=0)
    else:
        print(f'Connection to MQTT broker failed with result code {rc}')


def on_log(client, userdata, level, buf):
    print(f'MQTT Log: {buf}')


# Connect callbacks
client.on_connect = on_connect
client.on_log = on_log

# Connecting to a Broker
try:
    client.connect(broker, port=1883, keepalive=60)
except ConnectionError as e:
    print(f'Error connecting to MQTT broker: {e}')
    exit()


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Object Detection and Tracking with MQTT')
parser.add_argument('-c', '--camera', action='store_true', help='Set to true if you want to use the camera.')
parser.add_argument('-v', '--video', default=None, help='Path to video file.')
parser.add_argument('-m', '--model', default=None, help='Path to PB file with a trained model.')
args = parser.parse_args()


# Load the model
if args.model is not None:
    try:
        model = tf.saved_model.load(args.model)
    except OSError as e:
        print(f'Error loading the model: {e}')
        exit()
else:
    # Set the default model path if not provided
    default_model_path = 'default_model_path.pb'
    try:
        model = tf.saved_model.load(default_model_path)
    except OSError as e:
        print(f'Error loading the default model: {e}')
        exit()


# OpenCV video capture
if args.camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f'Error opening the camera.')
        exit()
else:
    try:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise IOError(f'Cannot open video: {args.video}')
    except OSError as e:
        print(f'Error opening the video: {e}')
        exit()


prev_count = 0
IN = 0
OUT = 0
total_count = 0  # Cumulative count

# CREATE OBJECT TRACKER
ct = CentroidTracker()
tracking_body = {}

# Initialize variables for calculating FPS
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("ended")
        break

    for_body_frame = frame.copy()

    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    found_body_detections = []
    detections = model(input_tensor)

    # Get detection boxes
    boxes = detections['detection_boxes'][0].numpy()

    # Get detection classes
    classes = detections['detection_classes'][0].numpy()

    # Get detection scores
    scores = detections['detection_scores'][0].numpy()

    # Count the number of people (class 1 in COCO dataset)
    curr_count = np.sum((classes == 1) & (scores > 0.5))

    # Draw bounding boxes
    for box in boxes[(classes == 1) & (scores > 0.5)]:
        padded_body_left = int(box[1] * frame.shape[1])
        padded_body_top = int(box[0] * frame.shape[0])
        padded_body_right = int(box[3] * frame.shape[1])
        padded_body_bottom = int(box[2] * frame.shape[0])
        cv2.rectangle(frame, (padded_body_left, padded_body_top), (padded_body_right, padded_body_bottom), (0, 255, 0), 2)
        cropped_body_points = [padded_body_left, padded_body_top, padded_body_right, padded_body_bottom]
        box = cropped_body_points
        found_body_detections.append(box)

    # Body tracking
    objects = ct.update(found_body_detections)
    for (objectID, centroid_bbox) in objects.items():
        centroid = centroid_bbox[0]
        bbox = centroid_bbox[1]

        if objectID not in tracking_body:
            tracking_body[objectID] = [time.time(), 0, 1]
            print(f'Welcome new person')
            print(f"Come-IN #{objectID} => {bbox}")
            padded_body_left, padded_body_top, padded_body_right, padded_body_bottom = bbox
            this_new_body = for_body_frame[padded_body_top:padded_body_bottom,
                            padded_body_left:padded_body_right]
            IN += 1

        else:
            tracking_body[objectID][1] = time.time() - tracking_body[objectID][0]
            if tracking_body[objectID][2] > 0:
                print(f"Went-OUT #{objectID} => {bbox}")
                padded_body_left, padded_body_top, padded_body_right, padded_body_bottom = bbox
                this_new_body = for_body_frame[padded_body_top:padded_body_bottom,
                                padded_body_left:padded_body_right]

    total_count = len(objects.items())
    OUT = IN - total_count

    # Display the resulting count on the frame
    cv2.putText(frame, f'Come-IN: {IN}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 4, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Went-OUT: {OUT}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 4, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Person on Frame : {total_count}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 4, 0), 2, cv2.LINE_AA)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 4, 0), 1, cv2.LINE_AA)

    if curr_count > prev_count:
        try:
            client.publish('video/in/peoplecount', payload=f"{curr_count - prev_count}", qos=0, retain=False)
        except Exception as e:
            print(f'Error publishing MQTT message: {e}')
    elif curr_count < prev_count:
        try:
            client.publish('video/out/peoplecount', payload=f"{prev_count - curr_count}", qos=0, retain=False)
        except Exception as e:
            print(f'Error publishing MQTT message: {e}')

    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
