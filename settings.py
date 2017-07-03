import os
from datetime import datetime
from keras import optimizers

# Configuration file with the global variables

CONFIG_FILE = "config.xml"

# Logger vars
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": "False",
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(name)s %(asctime)s %(module)s %(process)d %(thread)d %(pathname)s@%(lineno)s: %(message)s'
        },
        'simple': {
            'format': '%(filename)s: %(message)s'
        }
    },

    "handlers": {
        "console": {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },

        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": os.path.join("logfiles","info_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S"))),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": os.path.join("logfiles","errors.log"),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },

    "loggers": {
        "supervasion": {
            "level": "DEBUG",
            "handlers": ["console", "info_file_handler", "error_file_handler"],
            "propagate": "False"
        }
    }
}

# ROI vars
CONTOUR_THRESHOLD = 15  # Threshold to reject false detections

# Images classifier vars
MODELS_PATH = os.path.join("models")
INPUT_IMAGE_SIZE = (50, 50)
MODEL_ID = 8
MODEL_CONFIG_FILE = os.path.join(MODELS_PATH, "multiclass_model_%s.json" % MODEL_ID)
MODEL_WEIGHTS_FILE = os.path.join(MODELS_PATH, "multiclass_model_%s_weights.h5" % MODEL_ID)

## Keras model configuration vars
COLOR_MODE_TYPE = "rgb"  # "rgb" or "grayscale"
LOSS_TYPE = "categorical"
NUM_CHANNELS = 3
# My configuration is "channels_last" in ~/.keras/keras.json file
# Change it whether you have another one
INPUT_SHAPE = (1, INPUT_IMAGE_SIZE, NUM_CHANNELS)

OPTIMIZER = optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Video resolution vars (in pixels)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Distance and movement vars
# Distance in pixels to verify the element is really moving
MOVEMENT_THRESHOLD = 13  # move, stop events
DISTANCE_THRESHOLD_MARGIN = 0.04
MEDIUM_DISTANCE_THRESHOLD = 0.32 # in_cross10, out_cross10 events
#SHORT_DISTANCE_THRESHOLD =  MEDIUM_DISTANCE_THRESHOLD / 2 # in_cross5,
# out_cross5 events
SHORT_DISTANCE_THRESHOLD =  0.22 # in_cross5, out_cross5 events


# Stereovision constants
# Normalization constant values
NORMALISATION_VALUES = {
    "x": {
        "min": 0,
        "max": VIDEO_WIDTH
    },
    "y": {
        "min": 0,
        "max": VIDEO_HEIGHT
    },
    "z": {
        "min": 255,
        "max": 0
    }
}


