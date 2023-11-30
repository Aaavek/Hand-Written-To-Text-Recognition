# %%
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt

from scipy.ndimage import zoom  

from keras.models import load_model

# %% [markdown]
# ### Loading the models

# %%
def load_models():
    loaded_model = []

    for i in range(5):
        # Loading model
        model = load_model("./models/model_"+str(i)+".h5")
        loaded_model.append(model)
    return loaded_model

def predict(models, X_t):
    
    Y_pred = []

    for model in models:
        y_pred = model.predict(X_t)
        Y_pred.append(y_pred)


    Y_pred = np.array(Y_pred)
    Y_pred = np.argmax(Y_pred, axis=2)


    Y_pred = np.transpose(Y_pred)

    Y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=Y_pred)
    return Y_pred

# %%
models = load_models()

# %%
models

# %%
Rev_Dict = {}

# for i in range(10):
#     Rev_Dict[i] = str(i)
    
for i in range(26):
    Rev_Dict[i] = chr(i+65)
    Rev_Dict[i+26] = chr(i+97)
    # Rev_Dict[i+10] = chr(i+65)
    # Rev_Dict[i+36] = chr(i+97)
    
Rev_Dict

# %% [markdown]
# ### Reading and segmenting the image into horizontal lines

# %%
def get_segments(image):

    if image is None:
        print("Could not read the image.")
        exit()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image (e.g., apply Gaussian blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    # Perform adaptive dilation
    height, width = thresh.shape[:2]

    # Customize the kernel size based on the image dimensions and line characteristics
    kernel_height = 1  # Adjust as needed
    kernel_width = int(width)    # Adjust as needed
    kernel = np.full((kernel_height, kernel_width),255, np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=10)

    # Find contours
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        exit()
    # Sort contours based on x-coordinate of the bounding box
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    heights = [cv2.boundingRect(ctr)[3] for ctr in contours]
    
    avg_height = sum(heights)/len(heights)

    ans = 0

    for i, ctr in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        
        if (h < avg_height/2):
            continue
        
        # Extract ROI
        roi = blurred[y:y + h, x:x + w]
        
        cv2.imwrite("segment_no_" + str(ans) + ".png", roi)
        ans += 1
        cv2.rectangle(blurred, (x, y), (x + w, y + h), (90, 0, 255), 2)

    # Save final image with bounding boxes
    cv2.imwrite('segmented_image.png', blurred)

    return ans

#take input from user
input = None

while(True):
    input = input("Enter the path of the image: ")

    if input == "exit":
        break
    
    #check if file exists
    if not os.path.exists(input):
        print("File does not exist. Try again.")
        continue

Image = cv2.imread(input)

No_of_segments = get_segments(Image)


# %% [markdown]
# ### Character segment pre-processing

# %%
# the height and width of our final image
fnl_hyt = 32
fnl_wdt = 32
pad_value = 1


def crop_cond(arr):
    if np.min(arr) == 0 :
        return True
    else:
        return False


def crop(image):
    plt.imshow(image)
    plt.show()
    
    
    img_wdt = len(image[0])
    img_hyt = len(image)
    non_white_pixels = np.argwhere(image != pad_value)

    top_mrgn = 0
    bot_mrgn = img_hyt-1
    left_mrgn = 0
    right_mrgn = img_wdt-1

    # method 1****************************************************

    # top_mrgn, left_mrgn = non_white_pixels.min(axis=0)
    # bot_mrgn, right_mrgn = non_white_pixels.max(axis=0)

    # method 2*****************************************************
    for i in range(img_hyt):
        if crop_cond(image[i]):
            top_mrgn = i
            break
    
    for i in range(img_hyt):
        if crop_cond(image[img_hyt - i - 1]):
            bot_mrgn = img_hyt - i - 1
            break
    
    t_image = image.T
    for i in range(img_wdt):
        if crop_cond(t_image[i]):
            left_mrgn = i
            break
    
    for i in range(img_wdt):
        if crop_cond(t_image[img_wdt - i - 1]):
            bot_mrgn = img_wdt - i - 1
            break
    
    return image[top_mrgn:bot_mrgn+1 , left_mrgn:right_mrgn+1]

# the value which we are padding

def pad(image):    
    
    img_hyt = len(image)
    img_wdt = len(image[0])

    pad_hyt = img_hyt
    pad_wdt = img_wdt

    while(pad_hyt%fnl_hyt != 0):
        pad_hyt += 1

    while(pad_wdt%fnl_wdt != 0):
        pad_wdt += 1

    if pad_wdt//fnl_wdt > pad_hyt//fnl_hyt:
        pad_hyt = fnl_hyt*(pad_wdt//fnl_wdt)
    else :
        pad_wdt = fnl_wdt*(pad_hyt//fnl_hyt)
    
    top_pad = np.zeros(((pad_hyt-img_hyt)//2, img_wdt)) + pad_value
    bot_pad = np.zeros(((pad_hyt-img_hyt+1)//2 , img_wdt)) + pad_value
    left_pad = np.zeros((pad_hyt, (pad_wdt-img_wdt)//2)) + pad_value
    right_pad = np.zeros((pad_hyt, (pad_wdt-img_wdt+1)//2)) + pad_value

    # Stack the padding arrays and the image
    return np.hstack((left_pad, np.vstack((top_pad, image,bot_pad)) ,right_pad))

def reshape(image):
    return zoom(image,(fnl_hyt/len(image), fnl_wdt/len(image[0])) , order = 1)

def trim(image, avg):
    int_fill = 1 - (np.sum(image) / (image.shape[0] * image.shape[1]))
    fill = int_fill

    if fill > avg :
        while fill > avg:
            change = np.argwhere((image - np.roll(image, 1, axis=0)) +
                                (image - np.roll(image, 1, axis=1)) +
                                (image - np.roll(image, -1, axis=0)) +
                                (image - np.roll(image, -1, axis=1)) < 0)

            image[change[:, 0], change[:, 1]] = pad_value
            tfill = 1 - (np.sum(image) / (image.shape[0] * image.shape[1]))

            if tfill >= fill:
                raise ValueError("Error: Unable to trim the image further.")

            fill = tfill
    
    else :
        image = 1 - image
        fill = 1 - fill
        while fill >= 1 - avg:
            change = np.argwhere((image - np.roll(image, 1, axis=0)) +
                                (image - np.roll(image, 1, axis=1)) +
                                (image - np.roll(image, -1, axis=0)) +
                                (image - np.roll(image, -1, axis=1)) < 0)

            image[change[:, 0], change[:, 1]] = pad_value
            tfill = 1 - (np.sum(image) / (image.shape[0] * image.shape[1]))

            if tfill >= fill:
                raise ValueError("Error: Unable to trim the image further.")

            fill = tfill
        image = 1 - image

    return image

def transform(image):
    return reshape(pad(image))
    # return crop(image)

    

# %%
def image_to_array(image):
    image = transform(image)
    
    image = np.where(image > 0.5, 1, 0)
    
    image = trim(image, 0.2)
    
    return image

# %% [markdown]
# ### Character segmenting, prediction from the horizontally segmented characters

# %%
def segment_characters(segment, n):
    # cv2.imshow("Segment2", segment)
    
    plt.imshow(segment)
    plt.show()
    
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)
    
    widths = [cv2.boundingRect(contour)[2] for contour in contours]
    avgwidth = sum(widths)/len(widths)

    num = 0
    total_width = 0
    total_height = 0

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < avgwidth/5 and h < avgheight/5:
            continue
        
        total_width += w
        total_height += h
        
        num += 1

    avgwidth = total_width/num
    avgheight = total_height/num

    num = 0

    factor = 1.5

    prev_end = None
    
    Ans = ""

    # Iterate through each contour (presumed to be a character) and save as individual images
    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore contours that are too small

        if w < avgwidth/5 and h < avgheight/5:
            continue

        if prev_end is None:
            prev_end = x + w
        elif x > prev_end + factor * avgwidth:
            Ans += " "
        
        prev_end = x + w

        character_image = segment[y:y + h, x:x + w]

        # color to Gray scale character_image
        
        character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY) 
        
        # cv2.imwrite(f'segment_{n}_character_{i + 1}.png', character_image)
        
        cv2.rectangle(segment, (x, y), (x + w, y + h), (90, 0, 255), 2)

        
        character_image = character_image / 255

        X = image_to_array(character_image)
        
        # cv2.imwrite(f'segment_{n}_character_{i + 1}_transformed.png', X*255)
        
        X = np.array(X,dtype='float32')
        X = X.astype('float32')
        
        # Predict and print the value of x
        y_pred = predict(models, X.reshape(1, 32, 32, 1))
        print("Y_pred: ", y_pred)
        print("Predicted char: ", Rev_Dict[y_pred[0]])

        Ans += Rev_Dict[y_pred[0]]

    cv2.imwrite(f'segment_{n}_bounded_box_image.png', segment)

        # Save the character image
        
    return Ans

# %%

Final_ans = ""

for i in range(No_of_segments):
    segment = cv2.imread("segment_no_" + str(i) + ".png")

    x = segment_characters(segment, i)
    
    if i == 0:
        Final_ans += x
    else:
        Final_ans += "\n" + x
    
while Final_ans[0] == ' ':
    Final_ans = Final_ans[1:]
    
print("Predicted Text: \n", Final_ans)



