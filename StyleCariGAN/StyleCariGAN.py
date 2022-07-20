import os
import shutil
from PIL import Image


def make_input_directory():
        #StyleCariGAN/user_image
        dir_path = "./StyleCariGAN/user_image"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        dir = 'user_image'
        parent_dir = './StyleCariGAN'
        path = os.path.join(parent_dir, dir)
        os.mkdir(path)


def make_output_directory():
        #StyleCariGAN/user_result
        dir_path = "./StyleCariGAN/user_result"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        dir = 'user_result'
        parent_dir = './StyleCariGAN'
        path = os.path.join(parent_dir, dir)
        os.mkdir(path)


def make_final_output_directory():
        #StyleCariGAN/final_result
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
        os.system("wget " + url + " -P /root/yeonwoo/StyleCariGAN/user_image" +" -O photo.jpg")
        Image.open("./photo.jpg")  


def find_inputImg_2(user, user_id, emotion):
        if emotion == 0 : #func1
	  file_name = user.user_img.name.split('/')[1] #ex) cat.png
          input_img = "/CarryCARI/assets/user_image/{user_id}/{file_name}"
        else : #func2
          input_img = "/CarryCARI/assets/user_image/user_{user_id}.jpg" #user_{user_id}.jpg

        shutil.move(input_img, './StyleCariGAN/user_image')


def test():
        os.chdir("/root/yeonwoo/StyleCariGAN")

        #python test.py --ckpt [CHECKPOINT_PATH]              --input_dir [INPUT_IMAGE_PATH] --output_dir [OUTPUT_CARICATURE_PATH] --invert_images
        os.system("python test.py --ckpt ./checkpoint/StyleCariGAN/001000.pt --input_dir user_image --output_dir user_result --invert_images")


def choose_8styles():
        style_list = [0, 12, 15, 17, 22, 47, 58, 61]

        for i in style_list:
            user_result_dir = './user_result/photo/' + str(i) + '.png'
            final_result_dir = './final_result'
            shutil.move(user_result_dir, final_result_dir)

        Image.open("./final_result/58.png")


def run_StyleCariGAN():
        make_input_directory()
        make_output_directory()
        make_final_output_directory()

        find_inputImg_1()
        # find_inputImg_2(user, user_id, emotion)

        test()
        choose_8styles()


run_StyleCariGAN()
