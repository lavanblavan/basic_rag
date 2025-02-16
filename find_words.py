# Importing required functions for inference and visualization.
import os
from paddleocr import PaddleOCR ,draw_ocr
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
# images = convert_from_path('lavan_electrical_cv.pdf')
from pdf2image import convert_from_path

boxes,txts,scores,x_max,coord_txt = [],[],[],[],[]
ocr = PaddleOCR(use_angle_cls=True)


def save_ocr(img_path, out_path, result, font):
    save_path = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0] + '_output.png') # Better file naming
    image = cv2.imread(img_path)
    
    
    for i in result[0]:
        
        boxes.append(i[0])
        txts.append(i[1][0])
        scores.append(i[1][1])
        coord_txt.append(i[1]) 

    # im_show = draw_ocr(image, boxes, txts, scores, font_path=font)  # Use draw_ocr

    # cv2.imwrite(save_path, im_show) # Save the image with bounding boxes and text
    
    # # Optional: Display the image (for testing)
    # img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    for i in boxes:
        x_max.append(i[2][0])
    
    return txts,boxes,x_max,coord_txt

out_path = './output_images'
font = 'simfang.ttf'



def txt_find(images):
    txts1 =[]
    for i in range(len(images)):
  
      # Save pages as images in the pdf
        images[i].save('page'+ str(i) +'.jpg', 'JPEG')
        
        img_path =  'page'+ str(i) +'.jpg'
        result = ocr.ocr(img_path)
        txt,boxes,x_max,coord_txt =save_ocr(img_path, out_path, result, font)
        
        txts1.append(txt)
    
    return txt,boxes,x_max,result

txts,boxes,x_max,coord_txt = txt_find(convert_from_path('191025X-ASS2.pdf'))


    
	
