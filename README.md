# People-Counter-Python

This project demonstrates real-time object detection and tracking using a pre-trained deep learning model. The system is designed to count the number of people entering and exiting a frame in a video stream. It utilizes MQTT (Message Queuing Telemetry Transport) for communication and publishes count updates using MQTT messages.

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.x
- OpenCV (cv2) library
- NumPy library
- TensorFlow library
- Paho MQTT library

You can install these dependencies using pip:

<pre> pip install opencv-python numpy tensorflow paho-mqtt </pre>

## Detection Model Zoo (Pre-Trained Deep Learning Model)
 The TensorFlow Model Garden is a repository : [https://github.com/tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

## Getting Started
To use this project, follow the steps below:

1. Clone the repository or download the source code files.
2. Open a terminal or command prompt and navigate to the project directory.
3. Install the required dependencies as mentioned in the Prerequisites section.
4. Set up an MQTT broker to receive the count updates. You can use popular MQTT brokers like Mosquitto or HiveMQ, or 
   any other MQTT broker of your choice.
5. Run the following command to start the object detection and tracking system:

   - camera with model
     <pre> python object_detection.py -c -m path/to/model_directory </pre> 
     This command enables the `camera` and `replace path/to/model_directory` with the actual path to the model directory as 
     mentioned in the detection model zoo section. 
     starts the system. Press the q key to stop the system. 
      
   - video file with model
     <pre> python object_detection.py -v path/to/video_file.mp4 -m path/to/model_directory </pre> 
     This command enables the video file, replace `path/to/video_file.mp4` with the actual path to the video file and
     replace `path/to/model_directory` with the actual path to the model directory as  
     mentioned in the detection model zoo section. 
     starts the system. Press the q key to stop the system. 
     
6. The system will display the video stream with bounding boxes around detected people. 
   The count of people entering and exiting the frame, as well as the frames per second (FPS), will be shown on the screen.


     
     
   
