# ComputerVisionWorks

The following files are available:
* MotionDetector.py

## Motion Detector
This algorithm detects motion and whether there are people in the video stream.
Most of this algorithm was taken from Brandon Joffe's Motion Detection module in
his Home Surveillance project here [https://github.com/BrandonJoffe/home_surveillance].
Big thanks to Brandon for putting that out there.

I modified his Motion Detection module to work without needing the entire system so that
it could be more easily adapted to other uses, such as home automation.

Besides tweaking a few parameters in his algorithm, I added the Brightness Reset logic,
which will reset the background if a sudden light change occurs, like turning the lights
on or off in a room.

I also adapted the algorithm to take in a video source as an argument. The video source can
be a local cam or a folder that contains a list of sequence images, as often found in research projects.
I plan on adding support for webcam (via URL) soon.

## Resources
### Yet Another Computer Vision Index to Datasets
This site was very useful. It contains links to many different publicly available datasets
for computer vision. I was able to get everything I needed to experiment with the OpenCV features.

http://riemenschneider.hayko.at/vision/dataset/index.php?filter=+detection

## PIROPO database (People in Indoor Rooms with Perspective and Omnidirectional cameras)
This is the site where I grabbed most of the videos for motion, people detection, and tracking. One thing
to note is that the videos there are actually a sequence of jpegs, not a video file. Therefore, you must
iterate over these in your code. MotionDetector.py has a sample of that.

https://sites.google.com/site/piropodatabase/