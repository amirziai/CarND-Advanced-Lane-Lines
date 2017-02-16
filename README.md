#Advanced Lane Finding for a Self-Driving Cars

[//]: # (Image References)

[image1]: ./examples/undistorted.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

###Camera Calibration

The code for this step is in `calibration.py`. The `Calibration` class is initialized with image size and a calibration pickle file path. If this path exists the calibration points object and image is loaded, otherwise it is calculated.

In the calibration stage for each image we read and resize the image if necessary. We know that we're looking for the same cheeseboard (fixed number of rows and columns) which are passed to the `findChessboardCorners`. The output `corners` is appended to the `points_image` list and the same `obj` is attached to the `points_object` every time since the coordinates are not changing for the chess board. Once we have these values (pickled for later use) we can use `calibrateCamera` to get the camera matrix and distortion coefficients. Having those we can now undistort a new image using the `undistort` method in the class that calls `cv2.undistort`.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Distortion-correction
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Threshold binary image
I used yellow threshold (on HSV), Sobel gradient thresholds, and a 2D filter (using `cv2.filter2D`) to generate a binary image (thresholding steps at function `lane_mask` in `processing.py`). Here's an example of my output for this step:

![alt text][image3]

####3. Perspective transform

The `Perspective` class inside `perspective.py` takes care of this. My source and destination points are in `config.py` inside the `annotate` dictionary.

An example of perspective transform:

![alt text][image4]

####4. Polynomial fitting

Polynomials for left and right lanes are fit on a history of `x` and `y` points in the `update` function in the `Line` class in `lane_detection.py` (lines 35-61).

![alt text][image5]

####5. Radius and distance from center

I calculate the curvature in the `curvature` function (lines 126-131) in `processing.py`.

####6. Output

I'm annotating the video with curvature, distance from center, and lane overlays in functions `_draw_overlay` and `_draw_info` inside the `LaneDetector` class in `lane_detection.py` (lines 111-135). Here's an example:

![alt text][image6]

---

###Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

###Discussion
This approach is much more robust compared to my initial lane finding, using a history of information, better usage of gradients, incorporation of perspective transformation, the histogram method, and polynomial fitting. However it is still probably highly optimized for the test images and will not generalize to different lighting conditions, off-road situations, and maybe roads outside of California or the US.
