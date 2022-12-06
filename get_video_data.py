import cv2
import math 
import numpy as np
from PIL import Image
from PIL import ImageStat

def get_video_data(filename, video_timespan=[]):
    # takes two parameters: the file path 
    # and the real life timespan of the video (if the video is spead up)

    def brightness(im_file):
        im = Image.open(im_file)
        stat = ImageStat.Stat(im)
        return stat.mean[0]
        #finds the brightness of each image by averaging out all the pixel values



    def getFirstFrame(videofile):
        vidcap = cv2.VideoCapture(videofile)
        success, image = vidcap.read()
        if success:
            cv2.imwrite("first_frame.jpg", image)
        # this function gets the first frame of the video and saves it as a file called first_frame




    def get_contour(img):
        # this function finds a string of x and y coordinates for the contour of the brightest object
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turns image to gray
        blurred = cv2.GaussianBlur(gray, (7, 7), 0) # blurs image by averaging out all neightboring values
        # this is necessary for removing any noise 

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred) # finds location of brightest point
        # if the image isn't blurred then the brightest spot might be a random really bright pixel 
        # rather than the actual brightest region
        
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        # image preprocessing

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # finds the contours of the image

        boxes = []
        for c in contours: # loop over the contours individually
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) 
            # creates box around each contour and 
            # finds series of coordinates for each of the box's corners
            box = np.array(box, dtype="int")
            # each box is array of 4 tuples
            boxes.append(box) # adds each "box" to an array
    
        my_box = []
        for i in boxes:
            if (i[0][0]-10 < maxLoc[0] < i[1][0]+10) and (i[1][1]-10 < maxLoc[1] < i[0][1]+10):
                my_box.append(i)
                # checks all the boxes to see if they fall in range of the loci of the brightest point
                # the x coordinates of the max loci has to be in the range of the x coordinates of the left and right bounds of the box
                # similarly the y coordinate is in the range of the y coordinates of the top and bottom of the box
        ver = []
        hor = []
        for i in range(len(my_box)):
            for num in my_box[i]:
                # separates the x and y coordinates of each box into two separate lists
                ver.append(num[0])
                hor.append(num[1])
        return ver, hor 
        # returns a list of the x and a list of the y coordinates that form the contour/s 
        # that surrounds the brightest spot on the image




    def length(filename, frame_count):
        time_stamp = []
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS) # fnds how many frames per second in the video
        
        duration = frame_count / fps # gets duration of the video by dividing the amount of frame by frames per seconds
        seconds_per_frame = duration / frame_count # fins how many seconds passes between each frame
        
        s = 0
        for i in range(frame_count):
            time_stamp.append(s)
            s += seconds_per_frame
            # creates a list of values with the amount of seconds that have passed per frame

        return time_stamp, duration

    def crop(img, vertical_bounds, horizontal_bounds):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0) # turns image gray and then blurs
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred) # finds location of brightest point
        if len(vertical_bounds) > 0: # checks to see if the program was able to find contours around brightest loci
            # if no contours around the brightest loci were found, then the list will turn turn up empty

            box_height = max(vertical_bounds) - min(vertical_bounds) # determines the vertical height for the cropped image
            box_width = max(horizontal_bounds) - min(horizontal_bounds) # determines the horizontal width for the cropped image
            if (maxLoc[1] - box_height > 0) and (maxLoc[0]-box_width > 0): 
                # checks to make sure that the crop is still within the bounds of the image
                crop = gray[maxLoc[1]-box_height:maxLoc[1]+box_height, maxLoc[0]-box_width:maxLoc[0]+box_width] 
                # crops orginial image around brightest point
            else:
                crop = gray # if image is unable to be cropped, function returns original image
        else:
            crop = gray
        return crop



    getFirstFrame(filename)
    first_frame = cv2.imread("first_frame.jpg") # creates and reads first frame of the video
    x_coordinates, y_coordinates = get_contour(first_frame) 
    # finds the x and y coordinates of the contour around the first image
    # the rest of the program uses these coordinates to find how much each frame should be cropped by

    count = 0
    cap = cv2.VideoCapture(filename)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    brightness_values = []
    frame_count = 0

    while(cap.isOpened()):
        frameId = cap.get(25) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frame = crop(frame, x_coordinates, y_coordinates) # crops each frame around the brightest spot
            filename ="frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame) # creates image file
            
            image_brightness = brightness(filename) # finds the brightness index for each image
            brightness_values.append(image_brightness) # adds the brightness index for each image to an array
            
            frame_count += 1 # keeps a tally of how many frames there are in the video
    cap.release()

    time_values, vid_length = length(filename, frame_count) 
    # gets an array with the time passed at each frame and the length of the video

    new_time_values = []
    if video_timespan: # checks to see if the video has been compressed
        for i in time_values:
            i = (video_timespan[0] / vid_length) * i # scales the length of the video to it's duration in real time
            new_time_values.append(i)
        time_values = new_time_values # replaces old array with time values with updated values

    return time_values, brightness_values, vid_length 
    # returns an array with time at each frame, an array with the brightness at each frame, and the length of the video



