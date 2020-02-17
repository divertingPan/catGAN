import os
import cv2

img_id = 0
catface_cascade = cv2.CascadeClassifier('lbpcascade_frontalcatface.xml')
for maindir, subdir, file_name_list in os.walk('img/train/cat/'):
    for filename in file_name_list:
        apath = os.path.join(maindir, filename)
        
        img = cv2.imread(apath)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = catface_cascade.detectMultiScale(img, 1.1, 3, cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            face_img = img[x-w//2:x+w//2*3, y-h//2:y+h//2*3]
            # cv2.imshow('img', face_img)
            # cv2.waitKey(0)
            cv2.imwrite('img/train/catface/%d.jpg' %img_id, face_img)
            print(apath + 'saved: %d' %img_id)
            img_id = img_id + 1
            

