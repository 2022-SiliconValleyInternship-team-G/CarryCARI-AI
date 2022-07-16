import torch
import clip

import torchvision.transforms as transforms
import torch
from argparse import Namespace
from models.psp import pSp
from global_directions.manipulate import Manipulator
import numpy as np


resize_dims = (256, 256)

from PIL import Image
from common import tensor2im
from MapTS import GetFs,GetBoundary,GetDt
import matplotlib.pyplot as plt

import dlib
from alignment import align_face
import numpy as np
from PIL import Image


def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 


def display_alongside_source_image(result_image, source_image):
  res = np.concatenate([np.array(source_image.resize(resize_dims)),
                        np.array(result_image.resize(resize_dims))], axis=1)
  return Image.fromarray(res)


def run_on_batch(inputs, net):
  images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
  return images, latents
  
#main
def generate_imageclip(user_id, image_path, emotion):

  #init
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device) 

  EXPERIMENT_ARGS = {
        "model_path": "/content/ml/encoder4editing/e4e_ffhq_encode.pt"
    }
  EXPERIMENT_ARGS['transform'] = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

  model_path = EXPERIMENT_ARGS['model_path']
  ckpt = torch.load(model_path, map_location='cpu')
  opts = ckpt['opts']

  opts['checkpoint_path'] = model_path
  opts= Namespace(**opts)
  net = pSp(opts)
  net.eval()
  net.cuda()
  print('Model successfully loaded!')

  fs3=np.load('./npy/ffhq/fs3.npy')
  M=Manipulator(dataset_name='ffhq') 
  np.set_printoptions(suppress=True)

  #Align image : 입력된 이미지를 보고 얼굴이 있는 부분을 자르기
  #함수 parameter로 image_path받음
  original_image = Image.open(image_path)
  original_image = original_image.convert("RGB")

  input_image = run_alignment(image_path)
  input_image.resize(resize_dims) #(256,256)

  #Invert the image : 잘라진 부분을 latent vector로 변경
  img_transforms = EXPERIMENT_ARGS['transform']
  transformed_image = img_transforms(input_image)

  with torch.no_grad():
      images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
      result_image, latent = images[0], latents[0]
  torch.save(latents, f'/content/assets/user_image_latent/user_{user_id}_latents.pt')

  # Display inversion:
  display_alongside_source_image(tensor2im(result_image), input_image)

  img_index = 0
  latents=torch.load(f'/content/assets/user_image_latent/user_{user_id}_latents.pt')
  w_plus=latents.cpu().detach().numpy()
  dlatents_loaded=M.W2S(w_plus)

  img_indexs=[img_index]
  dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]

  M.num_images=len(img_indexs)

  M.alpha=[0]
  M.manipulate_layers=[0]
  codes,out=M.EditOneC(0,dlatent_tmp) 
  original=Image.fromarray(out[0,0]).resize((512,512))
  M.manipulate_layers=None
  original

  # 4가지 emotion과 integer mapping
  emotion_mapping = ['smile', 'sad', 'surprised', 'angry']
  input_emotion = emotion_mapping[emotion]
  neutral='face' #ex) face
  target= input_emotion + ' face' #ex) smile face

  classnames=[target,neutral]
  dt=GetDt(classnames,model)

  # trial&error결과 alpha=4.1, beta=0.15인게 전반적으로 좋았다.
  beta = 0.15 
  alpha = 4.1

  # 결과 이미지 도출 
  M.alpha=[alpha]
  boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
  codes=M.MSCode(dlatent_tmp,boundary_tmp2)
  out=M.GenerateImg(codes)
  generated=Image.fromarray(out[0,0]) #.resize((512,512))
  generated.save(f'/content/assets/clip_result/clipresult_{user_id}.jpg', 'JPEG') #생성된 이미지 저장

  plt.figure(figsize=(20,7), dpi= 100)
  plt.subplot(1,2,1)
  plt.imshow(original)
  plt.title('original')
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.imshow(generated)
  plt.title('manipulated')
  plt.axis('off')