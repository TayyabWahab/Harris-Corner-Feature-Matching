import numpy as np
import cv2
import glob


def GetSobel(image, Sobel, width, height):

    I_d = np.zeros((width, height), np.float32)


    for rows in range(width):
        for cols in range(height):

            if rows >= 1 or rows <= width-2 and cols >= 1 or cols <= height-2:
                for ind in range(3):
                    for ite in range(3):
                        I_d[rows][cols] += Sobel[ind][ite] * image[rows - ind - 1][cols - ite - 1]
            else:
                I_d[rows][cols] = image[rows][cols]

    return I_d


# Harris Corner Detection algorithm
def HarrisCornerDetection(image):

    # The two Sobel operators - for x and y direction
    SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    w, h = image.shape

    # X and Y derivative of image using Sobel operator
    ImgX = GetSobel(image, SobelX, w, h)
    ImgY = GetSobel(image, SobelY, w, h)
   
    for ind1 in range(w):
        for ind2 in range(h):
            if ImgY[ind1][ind2] < 0:
                ImgY[ind1][ind2] *= -1

            if ImgX[ind1][ind2] < 0:
                ImgX[ind1][ind2] *= -1


    ImgX_2 = np.square(ImgX)
    ImgY_2 = np.square(ImgY)

    ImgXY = np.multiply(ImgX, ImgY)
    ImgYX = np.multiply(ImgY, ImgX)

    #Use Gaussian Blur
    Sigma = 0.5
    kernelsize = (5, 5)

    ImgX_2 = cv2.GaussianBlur(ImgX_2, kernelsize, Sigma)
    ImgY_2 = cv2.GaussianBlur(ImgY_2, kernelsize, Sigma)
    ImgXY = cv2.GaussianBlur(ImgXY, kernelsize, Sigma)
    ImgYX = cv2.GaussianBlur(ImgYX, kernelsize, Sigma)

    mbar = []
    alpha = 0.1
    R = np.zeros((w, h), np.float32)
    for row in range(w):
        for col in range(h):
            M_bar = np.array([[ImgX_2[row][col], ImgXY[row][col]], [ImgYX[row][col], ImgY_2[row][col]]])
            R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))
            mbar.append(M_bar)
    print(len(mbar))
    return R

def feature_matching(points1, points2):
    smallest = 0
    smallest2 = 0
    smallest_distance = []
    second_smallest = []
    location_points1 = []
    location_points2 = []
    cc = None
    cc1  = None

    x = 0
    i = 0
    j = 0
    for c in points1:
        i=0
        for c1 in points2:
            if i == 0:            
                d = np.sqrt(np.square(c[0]-c1[0])+np.square(c[1]-c1[1]))
                smallest = d
                smallest2 = d
                cc = c
                cc1 = c1
            else:
                d = np.sqrt(np.square(c[0]-c1[0])+np.square(c[1]-c1[1]))

            if d <= smallest:
                smallest = d
                cc=c
                cc1 = c1
            if d < smallest2 and d > smallest:
                smallest2 = d
            i = i+1
        if (smallest/smallest2) > 0.5:
            continue
            
        smallest_distance.append(smallest)
        second_smallest.append(smallest2)
        location_points1.append(cc)
        location_points2.append(cc1)

    return location_points1, location_points2, smallest_distance
            


def features_extract(image,threshold = 60000):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    R = HarrisCornerDetection(gray)

    # This parameter will need tuning based on the use-case
    CornerStrengthThreshold = threshold

    # Plot detected corners on image
    radius = 1
    color = (255, 0, 0)  # Green
    thickness = 1

    PointList = []

    for row in range(w):
        for col in range(h):
            if R[row][col] > CornerStrengthThreshold:
                max = R[row][col]

                skip = False
                for nrow in range(5):
                    for ncol in range(5):
                        if row + nrow - 2 < w and col + ncol - 2 < h:
                            if R[row + nrow - 2][col + ncol - 2] > max:
                                skip = True
                                break

                if not skip:
                    cv2.circle(image, (col, row), radius, color, thickness)
                    PointList.append((row, col))
    
    return image, PointList, R

def draw_matches(im1,im2,points1,points2,distances):
    
    h1,w1,c = im1.shape
    h2,w2,c = im2.shape
    img = []
    temp=distances


    img = np.zeros([h1,w1+w2,3],dtype=np.uint8)
    
    p1 = []
    p2 = []
    dist = []
    dist.append(0)
    
    img[0:h1,0:w1,0:3] = im1
    img[0:h2,w1:w1+w2,0:3] = im2
    
    counter1 = 0
    counter2 = 0
    
    for i in points1:
        counter1 = counter1+1
        
    for i in points2:
        counter2 = counter2+1
    
    min_length = int(np.min([counter1,counter2])/2)
    
    print('min length',min_length)
    
    for i in range(min_length):
        i = temp.index(np.min(temp))
        temp[temp.index(np.min(temp))] = i+1000
        
        #cv2.line(img,(points1[i]),(points2[i][1]+w1,points2[i][0]),(0,255,0),1) 
        cv2.line(img,(points1[i][1],points1[i][0]),(points2[i][1]+w1,points2[i][0]),(0,255,0),1) 
        p1.append(points1[i])
        p2.append((points2[i][1]+w1,points2[i][1]))

        
    return img
    
def display(image,caption = ''):
    plt.figure(figsize = (5,10))
    plt.title(caption)
    plt.imshow(image)
    plt.show()



path_query = '/Users/eapplestroe/Matching/*.jpg'
paths = glob.glob(path_query)


images = []
for path in paths:
    image = cv2.imread(path)
    images.append(image)

reference_image = cv2.imread('/Users/eapplestroe/Downloads/original.jpeg')
reference_image = cv2.resize(reference_image,(0,0), fx = 0.1, fy = 0.1)
feature_image1, points1,R = features_extract(reference_image,10000)

i = 0
for q_image in images:
    feature_image2, points2,R = features_extract(q_image,1000000)
    display(feature_image2,'Features found')
    display(R,'R Image')
    
    loc_points1, loc_points2, distances = feature_matching(points1,points2)
    
    final = draw_matches(reference_image,q_image,loc_points1,loc_points2,distances)
    display(final,'Matching')
    cv2.imwrite('/Users/eapplestroe/Matching/Results/'+str(i)+'.jpg',final)
    i = i+1
    
print('Done')

    

    

    

