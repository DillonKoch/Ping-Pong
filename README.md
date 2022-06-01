# Ping Pong Referee and Training Games

GIF of a bunch of annotations over gameplay

## Contents
<!-- 
Intro
- loosely inspired by ttnet
- general flow of the project
Data Collection
- openttgames videos
- my own videos
Data Cleaning
- mov to mp4
- frame folders
Table Detection
- UNet architecture
- input/output
- augmentation/data cleaning
- training strategy/performance
Ball Detection
- go through the phases in the parent class
Event Detection
- bounces occur when ball goes down, up
- hits occur when ball goes left,right or right,left
Referee
- explain rules of ping pong and how parent info is enough to ref
Coffin Corner Challenge
- uses bounces to give players points
 -->

[1. Intro](#Intro)\
[2. Data Collection](#Data-Collection)\
[3. Data Cleaning](#Data-Cleaning)\
[4. Table Detection](#Table-Detection)\
[5. Ball Detection](#Ball-Detection)\
[6. Event Detection](#Event-Detection)\
[7. Referee](#Referee)\
[8. Coffin Corner Challenge](#Coffin-Corner-Challenge)


<hr>

## [1. Intro](/Intro/)
This project analyzes ping pong games from scratch by locating the table, tracking the ball, and detecting events.
This data can be used to referee a ping pong game and create other training games.

The rest of this README walks through the data collection and cleaning, the steps to locate the table/ball/events, and how that data is used by the referee and training games.

## [2. Data Collection](/Data_Collection/)
I used two data sources in this project:
- <a href="https://lab.osai.ai/">OpenTTGames Dataset</a>: an open online dataset of ping pong games shot at 120 frames per second
(script to download this is [download_data.py](/Data_Collection/download_data.py))
- Videos I have personally taken with my iPhone 13, also recorded at 120 frames per second

## [3. Data Cleaning](/Data_Cleaning/)
In order to train the table detection model faster, I use [frame_folders.py](/Data_Cleaning/frame_folders.py) to save off relevant frames from each video into separate folders.
This way, I can efficiently load only the frames I need for training.

Additionally, I use [mov_to_mp4.py](/Data_Cleaning/mov_to_mp4.py) to convert iPhone videos from MOV to MP4 format.


## [4. Table Detection](/Table_Detection/)
The first step in analyzing ping pong gameplay is to find the table.
We need to know where the table is in order to detect bounces and map them to a shot chart.

To do this, I used a UNet Semantic Segmentation model that identifies every pixel in an input image that belongs to the table.

((((Side by side images of a real frame and a table mask))))

Regular frames from the ping pong game are sent to the model, and the output is a black image with white pixels where the table is.

(((( details about the training process, the file to train, evaluation, and perhaps a screenshot of the weights and biases charts))))

Once this output is created, I use opencv to find the 4 contours, or corners of the table.

((( picture of a table mask with circles on the four corners )))

## [5. Ball Detection](/Ball_Detection/)
#### Experiments
Detecting the ball was the biggest challenge.
It's very small and is always moving.

The TTNet paper proposed a method of predicting the probability of the ball being present at each x-y pixel value, but that proved to be inaccurate and needlessly complicated.

I also considered standard object detection methods, but they also had a hard time finding a small and often blurry ball in a large frame.

#### Background Subtraction
I found the best method of ball tracking was to use a background subtraction method.
By looking at the difference between two consecutive frames, I could easily locate moving objects, and determine which ones are the ball.

These are two consecutive frames from one of the videos:
<!-- ! TWO FRAMES SIDE BY SIDE -->

They look almost identical, but there is some small movement.
Looking at the absolute difference between the two frames gives us this:
<!-- ! DIFFERENCE FRAME -->

This way, we can ignore the stationary background and only focus on the moving players and ball.

#### Ball Detection - Phase 1
<!-- classic, neighbor, backtracked -->
Now that we've used background subtraction, only the players and ball are visible.
The only task left is to distinguish the ball apart from the players.
Using OpenCV, I detect all the contours (shapes) in the background-subtracted image like this:
<!-- ! image of contours -->
The next task is programatically determining which contour is the ball, if any.

##### Find Ball - Classic
First, I locate the ball when its contour is above the middle of the table and far away from the two players.
This way, I am certain that the contour I detect is the ball, and not a moving player.

##### Find Ball - Neighbor
<!-- Once the ball is found using the classic method, I search the next frame for a similarly sized contour near the ball's contour from the previous frame. -->
Once the ball is found using the classic method, I search the next frame for 



##### Find Ball - Backtrack


#### Ball Detection - Phase 2
<!-- clean contours, find centers -->


#### Ball Detection - Phase 3
<!-- interpolate centers during events and within an arc, rejoin all the centers -->

<p align="center">
  <img src="./Misc/arc_dots_gif.gif" width=800 />
</p>


## [6. Event Detection](/Event_Detection/)


## [7. Referee](/Referee/)


## [8. Coffin Corner Challenge](/Coffin-Corner-Challenge/)
