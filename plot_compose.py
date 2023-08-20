import os
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

 
def resize_blank_img(srcimg, dstpath):
    img = cv2.imread(srcimg)
    blankimg = np.zeros(np.shape(img), np.uint8)
    blankimg[:, :, 0] = 255
    blankimg[:, :, 1] = 255
    blankimg[:, :, 2] = 255
    num = [os.path.join(dstpath,imgpath) for imgpath in os.listdir(dstpath) if os.path.splitext(imgpath)[1] in img_type]
    cv2.imwrite(dstpath + "\\" + str(len(num)+1) + ".jpg", blankimg)
 
def image_compose(image_path):
    if not os.path.isdir(image_path):
        return -1
    imgpath_vec = [os.path.join(image_path,imgpath) for imgpath in os.listdir(image_path) if os.path.splitext(imgpath)[1] in img_type]
 
    #   Use the avaerage width and height 
    avg_width = 0
    avg_heigth = 0
    if avg_width == 0 or avg_heigth == 0:
        size = []
        for item in imgpath_vec:
            size.append((Image.open(item)).size)
        sum_width = sum_heigth = 0
        for item in size:
            sum_width += item[0]
            sum_heigth += item[1]
        avg_width = int(sum_width/(len(size)))
        avg_heigth = int(sum_heigth/(len(size)))
    avg_size = (avg_width,avg_heigth)
 
    #   Resize the size of images
    vec = [os.path.join(image_path, imgpath) for imgpath in os.listdir(image_path) if
           os.path.splitext(imgpath)[1] in img_type]
    while (len(vec)) < COL * ROW:
        vec = [os.path.join(image_path, imgpath) for imgpath in os.listdir(image_path) if
                       os.path.splitext(imgpath)[1] in img_type]
        resize_blank_img(vec[0],image_path)
 
    imgs = []
    for item in vec:
        imgs.append((Image.open(item)).resize(avg_size,Image.BILINEAR))
 
    #  Compose to one picture
    result_img = Image.new(imgs[0].mode,(avg_width * COL,avg_heigth * ROW))
    index = 0
    for i in range(COL):
        for j in range(ROW):
            result_img.paste(imgs[index],(i * avg_width, j * avg_heigth))
            index+=1
 
    #  Show and save the composed image
    plt.imshow(result_img)
    # plt.show()
    result_img.save(path+'\composed.png')
 

if __name__ == "__main__":
    img_type = ['.jpg','.JPG','.png','.PNG','.bmp','.BMP']
    ROW = 2
    COL = 3
    # the images directory path
    filenames=os.listdir('Images/')
    for filename in filenames:
        path = 'Images/' + filename
        image_compose(path)