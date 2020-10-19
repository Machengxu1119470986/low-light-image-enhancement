import numpy as np
import skimage 
import os
import cv2
from skimage.measure import compare_psnr
def compute_ssim(imgpath_1, imgpath_2):
    img1=cv2.imread(imgpath_1)    
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imgpath_2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)    
    img2=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)    
    return skimage.measure.compare_ssim(img1,img2,data_range=255)
def compute_psnr(imgpath_1, imgpath_2):
    img1=cv2.imread(imgpath_1)    
    (h,w)=img1.shape[:2]    
    img2=cv2.imread(imgpath_2)    
    resized=cv2.resize(img2,(w,h))    
    (h1,w1)=resized.shape[:2]    
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)    
    img2=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)    
    return skimage.measure.compare_psnr(img1,img2,255)

img_s_path = 'D:/mcxhaha/low_light/kindzhengshi/lowlighthh11/xz-attention/LOLdataset/eval15/high/'              
img_out_path = 'D:/mcxhaha/low_light/kindzhengshi/lowlighthh11/xz-attention/results/LOLdataset_eval15/'   

img_s = os.listdir(img_s_path)
img_out = os.listdir(img_out_path)
img_s.sort(key = lambda x: int(x[:-4]))
img_out.sort(key = lambda x: int(x[:-11]))

#print(img_s,img_out)

num=0
ssim_sum=0
psnr_sum=0
for i,j in zip(img_s,img_out):
    temp_ssim = compute_ssim(img_s_path+i,img_out_path+j)
    temp_psnr = compute_psnr(img_s_path+i,img_out_path+j)
    ssim_sum = ssim_sum + temp_ssim
    psnr_sum = psnr_sum + temp_psnr
    num = num + 1

print("number=",num,"     ssim=",ssim_sum/num,"     psnr=",psnr_sum/num )



