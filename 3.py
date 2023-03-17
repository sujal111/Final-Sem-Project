import cv2
import numpy as np
import os

# Perform iris recognition using the normalized correlation coefficient (NCC) method
def iris_recognition(test_image, database_images):
    scores = []
    for image in database_images:
        result = cv2.matchTemplate(test_image, image, cv2.TM_CCORR_NORMED)
        score = np.max(result)
        scores.append(score)
    return scores


def check_db(test_image_path, database_path) :
    
    test_image = cv2.imread(test_image_path, 0)
    test_image = cv2.equalizeHist(test_image)
    
    for pid in os.listdir(database_path) :
        database_images = []
        for img in os.listdir(os.path.join(database_path,pid)) : 
            database_images.append(cv2.imread(os.path.join(database_path,pid,img), 0))
            
        # Preprocess the test image and the database images
        database_images = [cv2.equalizeHist(image) for image in database_images]

        # Compare the test image to the database images and print the similarity scores 
        max_scores = max(iris_recognition(test_image, database_images))
        
        if max_scores > 0.98 :
            print("Iris image matched with user {}".format(pid))
            return
        
    print("Iris image didn't matched with anyone in the databse")

# Load the test image and the database images
test_image_path = '/Users/sujal/Desktop/SemEndPro/CASIA1/2/002_1_1.jpg'
database_path = '/Users/sujal/Desktop/SemEndPro/CASIA1'
check_db(test_image_path, database_path)