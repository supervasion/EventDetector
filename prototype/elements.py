import abc
import logging
import pprint

import cv2
import numpy as np
from enum import Enum
from keras.preprocessing.image import img_to_array

import settings
from cv_utils import (find_contours, create_mask, create_ROI,
                      calculate_mass_centers, sort_contours,
                      measure_distance_between_points)
from events import Events

logger = logging.getLogger("supervasion")


class Element(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        self.position_in_scene = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def set_position_in_scene(self, frame):
        position = self.get_position_in_scene(frame)
        self.position_in_scene = position

    @abc.abstractmethod
    def get_position_in_scene(self, frame):
        return

    def print_instrument(self):
        pprint.pprint(self.__dict__)


class Pan(Element):

    def __init__(self, **kwargs):
        Element.__init__(self, **kwargs)

    def get_position_in_scene(self, frame):
        position = None
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        contours_detected = find_contours(mask)
        cnts, _ = sort_contours(contours_detected, method="left-to-right")
        mass_centers = calculate_mass_centers(cnts)
        if len(contours_detected) != 0:
            if self.name == "left_pan":
                cv2.imshow("pan_{}".format(self.name), mask)
                position = mass_centers[0]
            else:
                position = mass_centers[1]

        self.position_in_scene = position
        return position


class Bean(Element):

    def __init__(self, **kwargs):
        Element.__init__(self, **kwargs)

    def get_position_in_scene(self, frame):
        position = None
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        contours_detected = find_contours(mask)
        if len(contours_detected) != 0:
            cnts, rect = sort_contours(contours_detected,
                                       method="bottom-to-top")
            mass_centers = calculate_mass_centers(cnts)
            cX, cY = mass_centers[0]  # Taking the bottommost mark
            position = (cX, cY)
        return position



class States(Enum):
    CLOSED = 0
    OBJECT_CLAMPED = 1
    OPEN = 2


class Distances(Enum):
    MEDIUM = 0
    SHORT = 1
    LARGE = 2


class Instrument(Element):

    def __init__(self, **kwargs):
        Element.__init__(self, **kwargs)
        self.is_in_scene = False
        self.previous_positions = [(0, 0)] * 13
        self.last_position = (0, 0)
        self.is_moving = False
        # A buffer of size n with CLOSED state
        self.previous_predictions = [0] * 10
        self.previous_state = 0  #  CLOSED
        self.proximity_to_targets = None

    def set_position_in_scene(self, frame):
        position = self.get_position_in_scene(frame)
        self.position_in_scene = position
        self.update_last_position(position)

    def update_last_position(self, position):
        self.previous_positions.pop(0)
        self.previous_positions.append(position)
        if self.is_moving == True:
            self.last_position = position

    def get_position_in_scene(self, frame):
        position = None
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        contours_detected = find_contours(mask)
        if len(contours_detected) != 0:
            cnts, _ = sort_contours(contours_detected, method="bottom-to-top")
            mass_centers = calculate_mass_centers(cnts)
            position = mass_centers[0]  # Taking the bottommost mark
        return position

    def set_initial_proximity_to_targets(self, targets):
        target_names = [target.name for target in targets]
        self.proximity_to_targets = dict(zip(target_names, [Distances.LARGE] * len(
            target_names)))

    # Event detection
    def detect_is_moving(self, frame_pos, frame_sec):
        point_A = self.previous_positions[0]
        if self.is_moving == True:
            # Last position registered in buffer
            point_B = self.previous_positions[-1]
        else:
            point_B = self.last_position
        distance = measure_distance_between_points(point_A, point_B)
        if distance > settings.MOVEMENT_THRESHOLD and self.is_moving == False:
            Events.move(self, frame_pos, frame_sec)
            self.is_moving = True
        elif distance < settings.MOVEMENT_THRESHOLD and self.is_moving == True:
            Events.stop(self, frame_pos, frame_sec)
            self.is_moving = False

    def detect_is_in_scene(self, frame, frame_pos, frame_sec):
        # Detecting marks contours on instrument
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        cv2.imshow("instrument_{}".format(self.name), mask)
        contours_detected = find_contours(mask)
        if len(contours_detected) > 0 and self.is_in_scene == False:
            self.is_in_scene = True
            Events.in_scene(self, frame_pos, frame_sec)
        elif len(contours_detected) == 0 and self.is_in_scene == True:
            self.is_in_scene = False
            Events.out_scene(self, frame_pos, frame_sec)

    def detect_proximity_to_target(self,
                                   frame,
                                   disparity_map,
                                   target,
                                   frame_pos,
                                   frame_sec):
        # Thresholding the frame to extract the object contour and minimum
        # rectangle
        mask = create_mask(frame, target.lower_colors, target.upper_colors)
        contours_detected = find_contours(mask)
        cnts, rect = sort_contours(contours_detected, method="bottom-to-top")
        rect = rect[0]  # We choose the closest to the ground

        # Getting the 3D position of the given object
        obj_2d_pos = target.position_in_scene
        obj_3d_pos = Utils.get_3d_position(obj_2d_pos, disparity_map, rect)
        obj_3d_pos_norm = Utils.normalise_3d_position(obj_3d_pos)

        # Thresholding the frame to extract the instrument contour and minimum
        # rectangle
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        contours_detected = find_contours(mask)
        cnts, rect = sort_contours(contours_detected, method="bottom-to-top")
        rect = rect[0]  # We choose the closest to the ground

        # Getting the 3D position of this instrument
        self_3d_pos = Utils.get_3d_position(self.position_in_scene,
                                            disparity_map,
                                            rect)
        self_3d_pos_norm = Utils.normalise_3d_position(self_3d_pos)

        # Measuring the distance between the object and this instrument
        distance = measure_distance_between_points(obj_3d_pos_norm,
                                                   self_3d_pos_norm)

        # Using some vars for code shortening.
        margin = settings.DISTANCE_THRESHOLD_MARGIN
        short = settings.SHORT_DISTANCE_THRESHOLD
        medium = settings.MEDIUM_DISTANCE_THRESHOLD
        previous_proximity = self.proximity_to_targets[target.name]

        # Discretising the distance in a categorical value
        if distance < (short - margin):
            proximity = Distances.SHORT
        elif distance < (medium - margin) and distance > (short + margin):
            proximity = Distances.MEDIUM
        elif distance > (medium + margin):
            proximity = Distances.LARGE
        else:
            proximity = previous_proximity

        # Evaluating proximity to determine which event should be triggered.
        if (previous_proximity != Distances.SHORT and
                proximity == Distances.SHORT):
            Events.in_cross5(self, target, frame_pos, frame_sec)

        if (previous_proximity == Distances.SHORT and
                proximity == Distances.MEDIUM):
            Events.out_cross5(self, target, frame_pos, frame_sec)

        if (previous_proximity == Distances.LARGE and
                proximity == Distances.MEDIUM):
            Events.in_cross10(self, target, frame_pos, frame_sec)

        if (previous_proximity == Distances.MEDIUM and
                proximity == Distances.LARGE):
            Events.out_cross10(self, target, frame_pos, frame_sec)

        # Updating the distance from the object
        self.proximity_to_targets[target.name] = proximity

        return distance

    def detect_clamp_state(self,
                           frame,
                           image_classifier,
                           frame_pos,
                           frame_sec):
        mask = create_mask(frame, self.lower_colors, self.upper_colors)
        ROI = create_ROI(frame, mask, settings.INPUT_IMAGE_SIZE)

        # Insuring the image is not the dummy black image
        if np.count_nonzero(ROI) > 1:
            prediction = self.predict_clamp_state_with_model(ROI,
                                                             image_classifier)
            # Postprocess

            self.previous_predictions.pop(0)
            self.previous_predictions.append(prediction[0])

            # Using the mode to calculate the principal state
            postprocessed_prediction = max(
                self.previous_predictions, key=self.previous_predictions.count)

            if postprocessed_prediction != self.previous_state:
                new_state = postprocessed_prediction
                _object = "Bean"
                if new_state == 2:
                    Events.open_instrument(self, frame_pos, frame_sec)
                elif new_state == 0:
                    Events.close_instrument(self, frame_pos, frame_sec)
                elif new_state == 1:
                    Events.picked_object(self, _object, frame_pos, frame_sec)

                # In case that object was picked previously
                if self.previous_state == 1:
                    Events.dropped_object(self, _object, frame_pos, frame_sec)

                self.previous_state = postprocessed_prediction

        return ROI

    # Given an ROI image, checks the instrumental state with our model and
    # returns its prediction
    def predict_clamp_state_with_model(self, img, model):
        # Convert the input image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image normalisation process
        x = img_to_array(img)
        x -= x.min()
        x /= (x.max() - x.min())

        # Predicting the class of the image
        prediction = model.predict_classes(
            x.reshape(settings.INPUT_SHAPE), batch_size=32, verbose=0)

        return prediction


class Utils(object):

    @classmethod
    def get_3d_position(cls, _2d_position, disparity_map, rect):

        # Getting the z_index from disparity map
        z_index = cls.calculate_z_index(disparity_map, rect)

        # Merging the x, y and z values
        _3d_position = (_2d_position + (z_index, ))

        return _3d_position

    @classmethod
    def calculate_z_index(cls, disparity_map, rectangle):
        # Splitting the rectangle in its properties
        x, y, w, h = rectangle

        # Taking the intensity values applying the rectangle
        # position over disparity
        rect = disparity_map[y:y+h, x:x+w]

        # Dismissing the zero values
        rect_without_zeros = rect[np.nonzero(rect)]

        # Finally, we calculate the intensity mean of all
        z_index = rect_without_zeros.mean()

        # Because the pixel intensity in disparity map is inversely
        # proportional with the proximity to stereocams, we need to invert this
        # value to allow a good distance measure.
        z_index = round(255 - z_index, 2)

        return z_index

    @classmethod
    def normalise_3d_position(cls, _3d_position):

        # Normalising the position values
        _3d_position_norm = (cls.normalise_value(_3d_position[0],
                                                 settings.NORMALISATION_VALUES[
                                                     "x"]
                                                 ["min"],
                                                 settings.NORMALISATION_VALUES[
                                                     "x"]
                                                 ["max"]),
                             cls.normalise_value(_3d_position[1],
                                                 settings.NORMALISATION_VALUES[
                                                     "y"]
                                                 ["min"],
                                                 settings.NORMALISATION_VALUES[
                                                     "y"]
                                                 ["max"]),
                             cls.normalise_value(_3d_position[2],
                                                 settings.NORMALISATION_VALUES[
                                                     "z"]
                                                 ["min"],
                                                 settings.NORMALISATION_VALUES[
                                                     "z"]
                                                 ["max"]))
        return _3d_position_norm

    @classmethod
    def normalise_value(cls, value, _min, _max):
        value -= _min
        value /= float(_max - _min)
        return value
