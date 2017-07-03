import cv2
import imutils
import logging
import numpy as np
import settings

logger = logging.getLogger("supervasion")

def find_contours(mask):
    # Applying noise reduction and calculate a threshold image
    kernel = (3,3)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closing, kernel, 0)
    thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

    # finding contours on the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # Filtering by contour area applying threshold to discard false detections
    cnts = filter(lambda cnt: cv2.contourArea(cnt) >
                              settings.CONTOUR_THRESHOLD, cnts)

    # Sort the cnts from largest to smallest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    return cnts

# Extracted from:
# http://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def calculate_mass_centers(contours):
    mass_centers = []
    for c in contours:
        # Compute the center of the contour
        M = cv2.moments(c)

        # Avoid the ZeroDivision error
        if (M["m00"] == 0):
            M["m00"] = 1

        # Calculate the contour center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        mass_centers.append((cX, cY))

    return mass_centers


def create_mask(frame, lower_colors, upper_colors):
    # Some of techniques can alterate the original
    # frame, we create a copy to these cases
    frame_cpy = frame.copy()

    # Changing frame color to calculate the mask.
    frame_hsv = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2HSV)

    mask = []
    # Creating a mask with color specifications
    for lower_value, upper_value in zip(lower_colors,
                                        upper_colors):
        mask.append(cv2.inRange(frame_hsv, lower_value, upper_value))

    # Due the masks in opencv are numpy arrays they can
    # joined creating a composed mask
    return sum(mask)

    
"""
Calculate the Euclidean distance in pixels
between two positions
"""
def measure_distance_between_points(position_A, position_B):
    return cv2.norm(np.array(position_A) - np.array(position_B))

# Caveat -> These methods modify the frame appearance


def paint_text_in_frame(frame, point, text, text_color):
    cX, cY = point
    cv2.circle(frame, (cX, cY), 2, text_color, -1)
    cv2.putText(frame, text, (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


def paint_distance_in_frame(frame, point_A, point_B, value):
    cv2.line(frame, point_A[:2], point_B[:2], (60, 0, 60),
             thickness=1, lineType=8, shift=0)
    cv2.circle(frame,
               (int((point_A[0]+point_B[0])/2),
                int((point_A[1]+point_B[1])/2)),
               7, (60, 0, 60), -1)
    cv2.putText(frame, "%s" % str(value),
                (int((point_A[0]+point_B[0])/2)-20,
                 int((point_A[1]+point_B[1])/2)-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 0, 60), 1)


# ------------------------------------------------------


def create_ROI(frame, mask, ROI_size):
    closet_mark_to_clamp = {
        "contour": None,
        "mass_center": None
    }

    mass_centers = []

    # Some of techniques can alterate the original frame, we create a copy to
    # these cases
    frame_cpy = frame.copy()

    # First of all, we identify the color marks contours
    cnts = find_contours(mask)

    # If the contours of our color marks don't appear in the captured frame,
    # returns a black square image with the size of the ROI.
    if len(cnts) < 2:
        return np.zeros(ROI_size + (3,), dtype=np.uint8)

    # We take two most representative contours (our color marks)
    cnts = cnts[:2]

    # Then, we calculate the mass centers of these contours
    for c in cnts:

        # Compute the center of the contour
        M = cv2.moments(c)

        # Avoid the ZeroDivision error
        if (M["m00"] == 0):
            M["m00"] = 1

        # Calculate the contour center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Register the mass center of each contour
        mass_centers.append((cX, cY))

        # Due surgery instrumental always is leaning toward one of the sides,
        # we can estimate where is the instrumental clamp taking the lowest point
        # of the taken contours (highest Y-axis). With that in mind, we wil calculate
        # the lowest contour of image
        if closet_mark_to_clamp["contour"] is None:
            closet_mark_to_clamp["contour"] = c
            closet_mark_to_clamp["mass_center"] = (cX, cY)
        else:
            if closet_mark_to_clamp["mass_center"][1] < cY:
                closet_mark_to_clamp["contour"] = c
                closet_mark_to_clamp["mass_center"] = (cX, cY)

    # Then, we can calculate de distance between the two mass centers
    mass_centers_distance = np.linalg.norm(np.asarray(
        mass_centers[0]) - np.asarray(mass_centers[1]))

    # We calculate the angle of instrumental and print it on frame
    _, _, angle = cv2.fitEllipse(closet_mark_to_clamp["contour"])

    ### ROI generation part ###
    # Later, we use the fourth part of diference between mass centers as
    # reference for ROI size
    ROI_height = abs(mass_centers[1][1] - mass_centers[0][1]) / 2
    ROI_width = abs(mass_centers[1][0] - mass_centers[0][0]) / 2

    # Change the width direction depending the instrumental angle
    if angle < 90:
        ROI_width *= -1

    # Build mask for ROI
    ROI_mask = np.zeros(frame_cpy.shape, dtype=np.uint8)

    # Prepare the coordinates for a circle ROI
    ROI_center = (closet_mark_to_clamp["mass_center"][0] + ROI_width,
                  closet_mark_to_clamp["mass_center"][1] + ROI_height)
    ROI_radius = int(mass_centers_distance / 2)

    # Apply mask (using bitwise & operator) using original frame (free of
    # "putText" messages )
    ROI_frame = frame & ROI_mask

    # Apply ROI over original frame
    ROI_frame = frame[ROI_center[1] - ROI_radius:ROI_center[1] + ROI_radius,
                      ROI_center[0] - ROI_radius:ROI_center[0] + ROI_radius, :]

    # Resizing for classificator input
    ROI_frame = cv2.resize(ROI_frame, ROI_size)

    # Returns an image with the ROI
    return ROI_frame

