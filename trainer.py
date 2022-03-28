#The detection of faces in the image is done using MTCNN model. Before running this file make sure that you have installed MTCNN
#package in your local PC using the command 'pip install mtcnn'.
#MTCNN is the state of the art technique for detecting the faces present in an image

#Directory of the dataset.
#base_dir=r'Location to the direcotory that contain directories with images of persons to be trained.'
base_dir=r'dataset'

#importing all the required packages.
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

#Loading Facenet model
model_emb=load_model('facenet.h5')
detector=MTCNN()

def get_pixels(img):
  img=cv2.imread(img)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  results=detector.detect_faces(img)
  if not(results):
      return (0,np.array([1,2,3]))
  x1,y1,width,height=results[0]['box']
  x1,y1=abs(x1),abs(y1)
  x2,y2=x1+width,y1+height
  face=img[y1:y2,x1:x2]
  face=cv2.resize(face,(160,160))
  return ([x1,x2,y1,y2],face)

#cropping the faces present in the image and storing the location of the face present in the image.
removed=[]
#The images in which the face was not detected will be removed. All such images are stored in this list.
def load(directory):
    images=glob.glob(directory+'/**/*')
    x,y=[],[]
    for img in images:
        print(img)
        _,face_array=get_pixels(img)
        if face_array.shape[0]!=160:
          print("No Face Detected in ",img)
          removed.append(img)
          continue
        x.append(face_array)
        #y.append(img.split('/')[-2]) # Linux
        y.append(img.split('\\')[-2]) # Windows
    return (np.asarray(x),np.asarray(y))

#2
trainx,trainy=load(base_dir)


def get_embeddings(face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=model_emb.predict(samples)
    return yhat[0]

#Finding the embeddings of the cropped faces using Facenet model
def load_embeddings(pixel_array):
  embed=[]
  for face in pixel_array:
    embed.append(get_embeddings(face))
  return np.asarray(embed)

#trainx_embed=load_embeddings(trainx_pix)
trainx_embed=load_embeddings(trainx)

#Loading the embeddings into a numpy file for further use.
np.savez_compressed('data.npz',a=trainx_embed,b=trainy)
