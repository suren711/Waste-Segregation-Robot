
# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from Delta_Arm_Controller import pickup,ReleaseServo,sleep
import threading


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Create a lock for synchronization
lock = threading.Lock()

# Variable to indicate if the threads should continue running
running = True
labels = None
frame = None
res = None 
cap = None
capture_stat = False

# Initialize the frame buffer
frame_buffer = None

def load_labels(path='labels.txt'):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            labels[int(pair[0])] = pair[1].strip()
        else:
            labels[row_number] = pair[0].strip()
    return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def capture_frame():
    global cap
    global frame_buffer
    global running
    global frame_buffer2
    global capture_stat
    try:
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if ret:
                with lock:
                    frame_buffer = frame.copy()
                    capture_stat = True
            else:
                break
    except Exception as e:
        print(f"Error in capture_frame: {e}")
    finally:
        if cap is not None:
            cap.release()
  

def perf_obj_det():
    global frame_buffer
    global running
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    #print(type(frame_buffer))
    while running and frame_buffer is not None:
        with lock:
            
            img = cv2.resize(cv2.cvtColor(frame_buffer, cv2.COLOR_BGR2RGB), (320,320))
            res = detect_objects(interpreter, img, 0.8)
        
def robot_arm():
    global res
    global running
    # var to save the last position
    prev_xmid = 0
    prev_ymid = 0

    while running:
        with lock:
            if res is not None and len(res) > 0:
                #print(res[0]['bounding_box'])
                ymin, xmin, ymax, xmax = res[0]['bounding_box']
                xmid = (xmin + xmax) / 2
                ymid = (ymin + ymax) / 2
                prev_xmid, prev_ymid = pickup(xmid,ymid,prev_xmid,prev_ymid,res[0]['class_id'])

def display_feed():
    global res
    global labels
    global running
    global frame_buffer
    frame_count = 0
    frame_rate = 0.00
    start_time = cv2.getTickCount()
    #print(type(frame_buffer))

    while running :
        with lock:
            if frame_buffer is not None:
                frame_to_display = frame_buffer.copy()
            else:
                frame_to_display = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            if res is not None:
                print("there is a result")
                for result in res:
                    ymin, xmin, ymax, xmax = result['bounding_box']
                    xmin = int(max(1,xmin * CAMERA_WIDTH))
                    xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                    ymin = int(max(1, ymin * CAMERA_HEIGHT))
                    ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                
                    cv2.rectangle(frame_to_display,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                    print("box printing")
                    cv2.putText(frame_to_display,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)
                    print("item name printing")
                    
        # Display the frame with visualizations
        cv2.imshow('Pi Feed', frame_to_display)

        # Calculate and display the FPS
        frame_count += 1
        if frame_count >= 10:  # Calculate FPS every 10 frames
            end_time = cv2.getTickCount()
            time_diff = (end_time - start_time) / cv2.getTickFrequency()
            frame_rate = frame_count / time_diff
            print(f'FPS: {frame_rate:.2f}')
            # Draw framerate in corner of frame
            cv2.putText(frame_to_display,'FPS: {0:.2f}'.format(frame_rate),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            frame_count = 0
            start_time = end_time
        
        cv2.putText(frame_to_display,'FPS: {0:.2f}'.format(frame_rate),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            running = False
            break



def main():
    try:
        # Create and start the threads
        capture_thread = threading.Thread(target=capture_frame)
        detection_thread = threading.Thread(target=perf_obj_det)
        robot_thread = threading.Thread(target=robot_arm)
        display_thread = threading.Thread(target=display_feed)

        capture_thread.start()
        while(capture_stat == False):
            sleep(0.0001)
        
        if capture_stat == True:
            display_thread.start()
            detection_thread.start()
            robot_thread.start()
          

        # Wait for threads to finish
        capture_thread.join()
        detection_thread.join()
        display_thread.join()
        robot_thread.join()
    except Exception as e:
        # Handle exceptions at the highest level (e.g., log the error)
        print(f"Error in main: {e}")
    finally:

        global running
        running = False  # Set the running flag to False to stop all threads gracefully

        if cap is not None:
            cap.release()  # Release the video capture resource

        # Release the robot arm and resources (e.g., camera) when finished
        ReleaseServo()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()