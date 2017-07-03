import logging.config
import os
import time
from argparse import ArgumentParser
from datetime import datetime

import cv2
import cv2.cv as cv
import numpy as np
from bs4 import BeautifulSoup
from keras import backend as K
from keras.models import model_from_json
from stereovision.calibration import StereoCalibration

import settings
from cv_utils import (paint_distance_in_frame)
from elements import Pan, Instrument, Bean

# Logging configuration extracted from:
# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
if settings.LOGGING_CONFIG:
    logging.config.dictConfig(settings.LOGGING_CONFIG)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("supervasion")

# Command line parameters for collecting input data


def get_program_arguments():
    parser = ArgumentParser(description="Detection of low-level events "
                                        "in exercise 1 execution. This "
                                        "prototype can works both "
                                        "online and video inputs.")
    parser.add_argument("--output",
                        help="Enable the video output with some "
                             "information related with frame position "
                             "and time-frame (in seconds).",
                        action="store_true")
    online_parser = parser.add_mutually_exclusive_group(required=True)
    online_parser.add_argument("-o", "--online",
                        dest='online_feature',
                        nargs=2,
                        help="Enable the online version. Requires cameras ids.",
                        type=int)
    online_parser.add_argument("-n", "--no-online",
                        dest='online_feature',
                        nargs=2,
                        help="Enable the video inputs instead of the online "
                             "version. Requires the paths to each camera "
                             "video.")
    online_parser.set_defaults(online_feature=True)
    return parser.parse_args()

# Creates the color mark objects using the XML specifications
def process_color_marks(marks_config):
    color_marks = []

    # Loading the xml structure
    xml = BeautifulSoup(marks_config, "lxml")

    # Proccesing the XML file, extracting the info for our color mark objects
    for color_mark_tag in xml.findAll("color_mark"):
        color_mark = {"id": color_mark_tag.attrs["id"],
                      "name": color_mark_tag.attrs["name"],
                      "meta": color_mark_tag.attrs["meta"],
                      "type": color_mark_tag.attrs["type"],
                      "num_of_marks": color_mark_tag.attrs["num_of_marks"]}
        lower_colors = []
        upper_colors = []
        for (lower_color_tag,
             upper_color_tag) in zip(color_mark_tag.findAll("lower_color"),
                                     color_mark_tag.findAll("upper_color")):
            lower_color = np.array([lower_color_tag.attrs["h"],
                                    lower_color_tag.attrs["s"],
                                    lower_color_tag.attrs["v"]],
                                   dtype="uint8")
            upper_color = np.array([upper_color_tag.attrs["h"],
                                    upper_color_tag.attrs["s"],
                                    upper_color_tag.attrs["v"]],
                                   dtype="uint8")
            lower_colors.append(lower_color)
            upper_colors.append(upper_color)
        color_mark["lower_colors"] = lower_colors
        color_mark["upper_colors"] = upper_colors
        color_marks.append(color_mark)

    return color_marks


def init_elements(class_name, color_marks):
    elements = []
    for color_mark in color_marks:
        element = class_name(**color_mark)
        elements.append(element)
    return elements


def create_image_classifier():
    # In case that classifier was trained with grayscale images
    if settings.COLOR_MODE_TYPE == "grayscale":
        settings.NUM_CHANNELS = 1

    # dimensions of our images.
    img_width, img_height = settings.INPUT_IMAGE_SIZE

    if K.image_data_format() == 'channels_first':
        settings.INPUT_SHAPE = (1,
                                settings.NUM_CHANNELS,
                                img_width,
                                img_height)
    else:
        settings.INPUT_SHAPE = (1,
                                img_width,
                                img_height,
                                settings.NUM_CHANNELS)

    # load json and create model
    json_file = open(settings.MODEL_CONFIG_FILE, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(settings.MODEL_WEIGHTS_FILE)
    print("Loaded model from disk")

    model.compile(loss='%s_crossentropy' % (settings.LOSS_TYPE),
                  optimizer=settings.OPTIMIZER,
                  metrics=['accuracy'])
    return model


# Disparity map calculation method
def get_disparity(left_img, right_img, method="BM"):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    c, r = gray_left.shape
    if method == "BM":
        sbm = cv.CreateStereoBMState()
        disparity = cv.CreateMat(c, r, cv.CV_32F)
        sbm.SADWindowSize = 9
        sbm.preFilterType = 1
        sbm.preFilterSize = 5
        sbm.preFilterCap = 61
        sbm.minDisparity = -39
        sbm.numberOfDisparities = 112
        sbm.textureThreshold = 507
        sbm.uniquenessRatio = 0
        sbm.speckleRange = 8
        sbm.speckleWindowSize = 0

        gray_left = cv.fromarray(gray_left)
        gray_right = cv.fromarray(gray_right)

        cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
        disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
        cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
        disparity_visual = np.array(disparity_visual)

    elif method == "SGBM":
        sbm = cv2.StereoSGBM()
        sbm.SADWindowSize = 9
        sbm.numberOfDisparities = 96
        sbm.preFilterCap = 63
        sbm.minDisparity = -21
        sbm.uniquenessRatio = 7
        sbm.speckleWindowSize = 0
        sbm.speckleRange = 8
        sbm.disp12MaxDiff = 1
        sbm.fullDP = False

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(
            disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.cv.CV_MINMAX,
            dtype=cv2.cv.CV_8U)

    return disparity_visual


def main():
    # Taking the arguments passed by user
    args = get_program_arguments()

    # Loading configuration file with color marks specifications
    with open(settings.CONFIG_FILE, 'r') as f:
        config_file = f.read()

    # Taking the program parameters
    left_cam = args.online_feature[0]
    right_cam = args.online_feature[1]

    # Open the connection with cams
    cap_L = cv2.VideoCapture(left_cam)
    cap_R = cv2.VideoCapture(right_cam)
    cap_L.set(3, settings.VIDEO_WIDTH), cap_L.set(4, settings.VIDEO_HEIGHT)
    cap_R.set(3, settings.VIDEO_WIDTH), cap_R.set(4, settings.VIDEO_HEIGHT)

    # Preparing the video output
    if args.output:
        video_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_cam_1 = "output_{}_cam_1.mp4".format(video_name)
        mp4v = cv2.cv.CV_FOURCC(*'mp4v')
        output_1 = cv2.VideoWriter(
            os.path.join("videos", video_cam_1),
            mp4v,
            10,
            (int(cap_L.get(3)), int(cap_L.get(4))))
        video_cam_2 = "output_{}_cam_2.mp4".format(video_name)
        mp4v = cv2.cv.CV_FOURCC(*'mp4v')
        output_2 = cv2.VideoWriter(
            os.path.join("videos", video_cam_2),
            mp4v,
            10,
            (int(cap_R.get(3)), int(cap_R.get(4))))

    # Creating an array with color marks configuration
    color_marks = process_color_marks(config_file)

    # Filtering the marks associated to instruments
    instruments_color_marks = filter(lambda x: x["type"] == "instrument",
                                     color_marks)

    # Filtering the marks associated to pans position
    pans_color_marks = filter(lambda x: x["type"] == "pan",
                              color_marks)

    # Filtering the marks associated to beans color
    beans_color_marks = filter(lambda x: x["type"] == "object",
                               color_marks)

    # Creating objects of pans from info
    pans = init_elements(Pan, pans_color_marks)

    # Creating the instrument objects from info
    instruments = init_elements(Instrument, instruments_color_marks)

    # Creating bean object from info
    beans = init_elements(Bean, beans_color_marks)

    # Creating an image classifier to detect the instrument state
    image_classifier = create_image_classifier()

    # Disparity map with z-index information
    disparity_map = None

    # Loading the StereoVision resources to calibrate the cameras
    stereo_calibration = StereoCalibration(None,
                                           input_folder=os.path.join(
                                               "calibration_results"))

    # Taking the time-frame (in seconds) and position of current frame
    frame_pos = 1
    frame_sec = 0
    start_time = time.time()

    # Some initialisations
    for pan in pans:
        pan.set_position_in_scene(cap_L.read()[1])

    for instrument in instruments:
        instrument.set_initial_proximity_to_targets(pans)

    while(True):

        # Capturing frame-by-frame
        ret_L, frame_L = cap_L.read()
        ret_R, frame_R = cap_R.read()

        # Rectifying frames with the camera calibration
        (rf_frame_L, rf_frame_R) = stereo_calibration.rectify((frame_L,
                                                               frame_R))

        # Creating disparity map
        disparity_map = get_disparity(rf_frame_L, rf_frame_R, method="BM")

        # Some of techniques can alterate the original frame, we create a copy
        # to these cases
        frame_cpy = frame_L.copy()

        # Taking bean position in the scene
        for bean in beans:
            bean.set_position_in_scene(frame_L)


        # Detecting in_scene / out_scene events
        for instrument in instruments:
            instrument.detect_is_in_scene(frame_L, frame_pos, frame_sec)

            if instrument.is_in_scene is True:
                # Setting the x and y position of each instrument
                instrument.set_position_in_scene(frame_L)

                # Detecting whether instrumental is moving or not
                instrument.detect_is_moving(frame_pos, frame_sec)

                # Painting distance between instruments and pans
                for pan in pans:
                    position = pan.position_in_scene
                    if position is not None:
                        distance = instrument.detect_proximity_to_target(
                            frame_L, disparity_map, pan, frame_pos, frame_sec)
                        paint_distance_in_frame(frame_cpy,
                                                instrument.position_in_scene,
                                                pan.position_in_scene,
                                                distance)

                # Detecting events related with state changes of instruments
                instrument.detect_clamp_state(frame_L,
                                              image_classifier,
                                              frame_pos,
                                              frame_sec)

        # Write the time-frame over main frame
        cv2.putText(frame_cpy, "Time-Frame: %s " % str(frame_sec), (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 2)
        cv2.putText(frame_cpy, "Frame Pos: %s " % str(frame_pos), (20, 60),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)

        # Incrementing the vars related to frame position and time
        frame_pos += 1
        #frame_sec = int(frame_pos / fps)
        frame_sec = int(time.time() - start_time)

        # Showing video on screen
        # cv2.imshow("video", frame)

        # Showing frame_cpy with elements position
        cv2.imshow("output", frame_cpy)
        cv2.imshow("DM", disparity_map)

        if args.output:
            output_1.write(frame_L)
            output_2.write(frame_R)

        # Press 'q' key to close the window with the video
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    # When everything done, release the capture
    cap_L.release()
    cap_R.release()

    # Destroying all opened windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
