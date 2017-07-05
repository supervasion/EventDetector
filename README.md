# EventDetector 
We present an event detection system in a laparoscopic surgery domain, as part of a more ambitious supervasion by observation project. The system, which only requires the incorporation of two cameras in a laparoscopic training box, integrates several computer vision and machine learning techniques to detect the states and movements of the elements involved in the exercise. We compare the states detected by the system with the hand-labelled ground truth, using an exercise of the domain as example. We show that the system is able to detect the events accurately. 

## Prototype
The `Event Detector` prototype is our first approach for the software component that is in charge of processing the video recordings of the domain (minimally invasive surgery exercise), detecting the finest grain of information about the observable events. The program code is available in the [`prototype`](/prototype) folder.

## Results
The results obtained after prototype execution can be found in [`evaluation`] folder. The figure below shows the preliminary results that we have obtained. Concretely, this figure shows the comparison between the
output of our prototype and the hand-labelled ground truth, using a recording with an example of the domain. Specifically, rows represent the low-level events involved in this exercise. The columns state the order in which each event happens. In this phase of development, we focus on ensuring the detection of events and whether they are in the right order. Thus, the results show that the prototype fits the ground truth reasonably well during the example execution, although we found some mistakes in the order of events detected (swap pattern). Therefore, these results are promising to move from this low-level event detection and recognition to a more high-level interpretation of events such that the feedback can be of a higher quality.
![Evaluation plot](/evaluation/evaluation_plot.png)
