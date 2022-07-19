import os
import shutil
import time ##어디에 쓰이는거지?
from PIL import Image


def make_input_directory():
	#StyleCariGAN/user_image
	# 이전 기록이 남아있으면 디렉토리 통째로 삭제 후 재생성
	dir_path = "./StyleCariGAN/user_image"
	if os.path.exists(dir_path): 
	    shutil.rmtree(dir_path)

	dir = 'user_image'
	parent_dir = './StyleCariGAN'
	path = os.path.join(parent_dir, dir)
	os.mkdir(path)
	

def make_output_directory():
	#StyleCariGAN/user_result
	# 이전 기록이 남아있으면 디렉토리 통째로 삭제 후 재생성
	dir_path = "./StyleCariGAN/user_result"
	if os.path.exists(dir_path): 
	    shutil.rmtree(dir_path)
	
	dir = 'user_result'
	parent_dir = './StyleCariGAN'
	path = os.path.join(parent_dir, dir)
	os.mkdir(path)


def make_final_output_directory():
	#StyleCariGAN/final_result
	# 이전 기록이 남아있으면 디렉토리 통째로 삭제 후 재생성
	dir_path = "./StyleCariGAN/final_result"
	if os.path.exists(dir_path): 
	    shutil.rmtree(dir_path)

	dir = 'final_result'
	parent_dir = './StyleCariGAN'
	path = os.path.join(parent_dir, dir)
	os.mkdir(path)


def find_inputImg_1():
	os.chdir("./StyleCariGAN/user_image")

	url = "https://img.hankyung.com/photo/202111/p1065590921493731_758_thum.jpg"
	os.system("curl " + url + " > photo.jpg") # curl 요청 [curl "이미지 주소" > "저장 될 이미지 파일 이름" ]
	
	Image.open("./photo.jpg")


def find_inputImg_2(user, user_id, emotion): ##user받아오면 user.user_image.name 이거 되는거 맞는지?
	if emotion == 0 : #func1
	  input_img = "/CarryCARI/assets/user_image/{user_id}/{user.user_image.name}.jpg" #사용자가 입력한 이미지
	else : #func2
	  input_img = "/CarryCARI/assets/user_image/user_{user_id}.jpg" #user_{user_id}.jpg
	
	shutil.move(input_img, './StyleCariGAN/user_image')


def test():
	os.chdir("./StyleCariGAN")

	#python test.py --ckpt [CHECKPOINT_PATH]              --input_dir [INPUT_IMAGE_PATH] --output_dir [OUTPUT_CARICATURE_PATH] --invert_images
	!python test.py --ckpt ./checkpoint/StyleCariGAN/001000.pt --input_dir user_image --output_dir user_result --invert_images


def choose_8styles():
	#8개 스타일 고정
	style_list = [0, 12, 15, 17, 22, 47, 58, 61]

	for i in style_list:
	    user_result_dir = './StyleCariGAN/user_result/photo/' + str(i) + '.png'
	    final_result_dir = './StyleCariGAN/final_result'
	    shutil.move(user_result_dir, final_result_dir)
	
	#테스트 위해서 하나만 출력
	Image.open("./StyleCariGAN/final_result/58.png")


def run_StyleCariGAN(user, user_id, emotion):
	make_input_directory()
	make_output_directory()
	make_final_output_directory()

	find_inputImg_1()
	# find_inputImg_2(user, user_id, emotion)

	test()
	
	choose_8styles()
