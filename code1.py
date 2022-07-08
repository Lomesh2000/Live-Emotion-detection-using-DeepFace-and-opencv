import deepface
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Loading the video
video= cv2.VideoCapture('Video 1.mp4')

#getting frame_per_second , width, height of the video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# specify the four code for video codec 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# creating output variable to save every modified frame to get the result video 
output = cv2.VideoWriter('output3.mp4', fourcc, fps, (frame_width,frame_height))

while True:
    result,frame = video.read()
    #print(result)
    if not result:
        break
    
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    try:
        emotion=DeepFace.analyze(frame, actions = ['emotion'])
        cordinates=emotion['region']
        print(cordinates)
        cv2.rectangle(frame,(cordinates['x'],cordinates['y']),(cordinates['w']+cordinates['x'],cordinates['h']+cordinates['y']),(0,255,0),2)
        cv2.putText(frame,emotion['dominant_emotion'],(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    except:
        #print(1)
        pass
    #cv2.imshow('gray',gray)\
    output.write(frame)
    cv2.imshow('frame',frame)    
    if cv2.waitKey(1) & 0xFF==ord('q'): 
        break
    
video.release()
output.release()
cv2.destroyAllWindows()