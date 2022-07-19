###############################################
##### input output directory 생성 및 삭제 #####
###############################################

import os
import shutil

# 이전 기록이 남아있으면 디렉토리 통째로 삭제 후 재생성
# input_directory
dir_path = "/content/StyleCariGAN/user_image"
if os.path.exists(dir_path): 
    shutil.rmtree(dir_path)

dir = 'user_image'
parent_dir = '/content/StyleCariGAN'
path = os.path.join(parent_dir, dir)
os.mkdir(path)


# 이전 기록이 남아있으면 디렉토리 통째로 삭제 후 재생성
# output_directory
dir_path = "/content/StyleCariGAN/user_result"
if os.path.exists(dir_path): 
    shutil.rmtree(dir_path)

dir = 'user_result'
parent_dir = '/content/StyleCariGAN'
path = os.path.join(parent_dir, dir)
os.mkdir(path)



########################
##### 이미지 넣기  #####
########################

#url>jpg
import os
import time
from PIL import Image

os.chdir("/content/StyleCariGAN/user_image")

url = "https://img.hankyung.com/photo/202111/p1065590921493731_758_thum.jpg"
os.system("curl " + url + " > photo.jpg") # curl 요청 [curl "이미지 주소" > "저장 될 이미지 파일 이름" ]

Image.open("./photo.jpg")

'''
import shutil

if emotion == 0 : #func1
  input_img = "/content/assets/user_image/{user_id}/{user.user_image.name}.jpg" #사용자가 입력한 이미지
else : #func2
  input_img = "/content/assets/user_image/user_{user_id}.jpg" #user_{user_id}.jpg

shutil.move(input_img, '/content/StyleCariGAN/user_image') 
'''



################
##### test #####
################

os.chdir("/content/StyleCariGAN")

#python test.py --ckpt [CHECKPOINT_PATH]              --input_dir [INPUT_IMAGE_PATH] --output_dir [OUTPUT_CARICATURE_PATH] --invert_images
!python test.py --ckpt ./checkpoint/StyleCariGAN/001000.pt --input_dir user_image --output_dir user_result --invert_images




#######################
##### output 조정 #####
#######################

#final_result폴더
dir_path = "/content/final_result"
if os.path.exists(dir_path): 
    shutil.rmtree(dir_path)
dir = 'final_result'
parent_dir = '/content'
path = os.path.join(parent_dir, dir)
os.mkdir(path)

#8개 스타일 고정
style_list = [0, 12, 15, 17, 22, 47, 58, 61]

for i in style_list:
    user_result_dir = '/content/StyleCariGAN/user_result/photo/' + str(i) + '.png'
    final_result_dir = '/content/final_result'
    shutil.move(user_result_dir, final_result_dir)

#테스트 위해서 하나만 출력
Image.open("/content/final_result/58.png")
