import streamlit as st
st.set_page_config(layout="centered")
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial.distance import canberra
from sklearn.preprocessing import Normalizer

#loading the facenet model
model_emb=load_model('facenet.h5')

#Loading the mtcnn model
detector=MTCNN()

l2_encoder=Normalizer(norm='l2')

#Modify the argument of np.load() function.
#data=np.load('location to emebeddings numpy file that got generated using emebeddings_generator.py')
data=np.load('data.npz')
trainx_embed,trainy=data['a'],data['b']

def get_pixels(img):
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

def get_embeddings(face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=model_emb.predict(samples)
    return yhat[0]

def calculate_distance(embedding,known_faces,known_labels):
  store=dict()
  for i in known_labels:
    if i not in store:
      store[i]=[]
  for i in range(known_faces.shape[0]):
    store[known_labels[i]].append(canberra(embedding,known_faces[i]))
  for i in store.keys():
    store[i]=sum(store[i])/len(store[i])
  dist=min(store.values())
  for i in store:
    if store[i]==dist:
      return (dist,i)


st.markdown('# <p style="background:#336699"  align=center><font face="微软雅黑" color="#FFFFFF" size="6"><b><br>兰州大学国家级大学生创新创业训练计划<br><br></b></font></p>',True)
st.markdown("# <center> <u> 口罩人脸识别系统 </u> </center> <br/> <br/>",True)


st.markdown("## <center> 上传戴口罩的人脸图片 </center>",True)
a=st.file_uploader("")
# cnt1,cnt2=st.beta_columns(2)
cnt1,cnt2=st.columns(2) # 修改了
if a:
    with cnt1:
      st.markdown("### <center> 上传图片 </center>",True)
      st.image(a)
    image=Image.open(a)
    im=image.save('a.jpg')
    image=cv2.imread('a.jpg')
    cord,face_pixels=get_pixels(image)
    if cord==0:
        st.error("未识别到人脸。")
    else:
        x1,x2,y1,y2=cord[0],cord[1],cord[2],cord[3]
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),10)
        face_embeddings=get_embeddings(face_pixels)
        face_embeddings_norm=l2_encoder.fit_transform([face_embeddings])
        trainx_norm=l2_encoder.fit_transform(trainx_embed)
        distance,label=calculate_distance(face_embeddings_norm,trainx_norm,trainy)
        print(label,distance)
        if distance>75:
            label="UNKNOWN"
        cv2.putText(image,label,(x1+80,y2+70),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,255),8)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with cnt2:
          st.markdown("### <center> 识别图片 </center>",True)
          st.image(image)
          st.write(label,distance)
        st.success("图片中的人脸为："+label)


st.markdown("## <center> 实时检测戴口罩的人脸 </center>",True)
run = st.button('开始 / 停止')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)
while run:
    ret, frame = cam.read()
    if ret==True:
      cv2.imwrite('a.jpg', frame)   # 保存图像
      image=cv2.imread('a.jpg')
      cord,face_pixels=get_pixels(image)
      if not cord==0:
          x1,x2,y1,y2=cord[0],cord[1],cord[2],cord[3]
          cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),10)
          face_embeddings=get_embeddings(face_pixels)
          face_embeddings_norm=l2_encoder.fit_transform([face_embeddings])
          trainx_norm=l2_encoder.fit_transform(trainx_embed)
          distance,label=calculate_distance(face_embeddings_norm,trainx_norm,trainy)
          print(label,distance)
          if distance>75:
              label="UNKNOWN"
          cv2.putText(image,label,(x1+80,y2+50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),8)
          image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          FRAME_WINDOW.image(image)
          #st.success("图片中的人脸为："+label)
else:
    cam.release()
    st.write('')