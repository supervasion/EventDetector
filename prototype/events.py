import logging

logger = logging.getLogger("supervasion")

class Events(object):

    @classmethod
    def print_event(cls, kwargs):
        logger.info("=======================================")
        for k, v in kwargs.iteritems():
            logger.info("%s: %s"%(str(k), str(v)))
        logger.info("=======================================")

    @classmethod
    def close_instrument(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "closes",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def dropped_object(cls, instrument, _object, num_frame, time_frame):
        cls.print_event({
            "Event": "drops",
            "Instrument": instrument.name,
            "Object": _object,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def in_cross10(cls, instrument, target, num_frame, time_frame):
        cls.print_event({
            "Event": "crosses_in_10",
            "Instrument": instrument.name,
            "Target": target.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def in_cross5(cls, instrument, target, num_frame, time_frame):
        cls.print_event({
            "Event": "crosses_in_5",
            "Instrument": instrument.name,
            "Target": target.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def in_scene(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "appears",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })
    
    @classmethod
    def move(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "starts_moving",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def on_towards(cls, instrument, target, num_frame, time_frame):
        cls.print_event({
            "Event": "on_towards", 
            "Instrument": instrument.name,
            "Target": target.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def open_instrument(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "opens",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def out_cross10(cls, instrument, target, num_frame, time_frame):
        cls.print_event({
            "Event": "crosses_out_10",
            "Instrument": instrument.name,
            "Target": target.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def out_cross5(cls, instrument, target, num_frame, time_frame):
        cls.print_event({
            "Event": "crosses_out_5",
            "Instrument": instrument.name,
            "Target": target.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def out_scene(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "disappears",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def picked_object(cls, instrument, _object, num_frame, time_frame):
        cls.print_event({
            "Event": "picks",
            "Instrument": instrument.name,
            "Object": _object,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })

    @classmethod
    def stop(cls, instrument, num_frame, time_frame):
        cls.print_event({
            "Event": "stops",
            "Instrument": instrument.name,
            "Time_Frame": time_frame,
            "Frame": num_frame
        })
