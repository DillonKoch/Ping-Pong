# Ping Pong Referee and Training Games


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

[1. Intro](#Intro)
[2. Data Collection](#Data-Collection)
[3. Data Cleaning](#Data-Cleaning)
[4. Table Detection](#Table-Detection)
[5. Ball Detection](#Ball-Detection)
[6. Event Detection](#Event-Detection)
[7. Referee](#Referee)
[8. Coffin Corner Challenge](#Coffin-Corner-Challenge)


<hr>
<a name="Intro"></a>

# [1. Intro](/Intro/)


# [2. Data Collection](/Data_Collection/)


# [3. Data Cleaning](/Data_Cleaning/)


# [4. Table Detection](/Table_Detection/)


# [5. Ball Detection](/Ball_Detection/)
<!-- 
background subtraction

 -->
### Experiments
Detecting the ball was the biggest challenge.
It's very small and is always moving.

The TTNet paper proposed a method of predicting the probability of the ball being present at each x-y pixel value, but that proved to be inaccurate and needlessly complicated.

I also considered standard object detection methods, but they also had a hard time finding a small and often blurry ball in a large frame.

### Background Subtraction
I found the best method of ball tracking was to use a background subtraction method.
By looking at the difference between two consecutive frames, I could easily locate moving objects, and determine which ones are the ball.

These are two consecutive frames from one of the videos:
<!-- ! TWO FRAMES SIDE BY SIDE -->

They look almost identical, but there is some small movement.
Looking at the absolute difference between the two frames gives us this:
<!-- ! DIFFERENCE FRAME -->

This way, we can ignore the stationary background and only focus on the moving players and ball.

### Ball Detection - Phase 1
<!-- classic, neighbor, backtracked -->
Now that we've used background subtraction, only the players and ball are visible.
The only task left is to distinguish the ball apart from the players.
Using OpenCV, I detect all the contours (shapes) in the background-subtracted image like this:
<!-- ! image of contours -->
The next task is programatically determining which contour is the ball, if any.

#### Find Ball - Classic
First, I locate the ball when its contour is above the middle of the table and far away from the two players.
This way, I am certain that the contour I detect is the ball, and not a moving player.

#### Find Ball - Neighbor
<!-- Once the ball is found using the classic method, I search the next frame for a similarly sized contour near the ball's contour from the previous frame. -->
Once the ball is found using the classic method, I search the next frame for 



#### Find Ball - Backtrack


### Ball Detection - Phase 2
<!-- clean contours, find centers -->


### Ball Detection - Phase 3
<!-- interpolate centers during events and within an arc, rejoin all the centers -->


# [6. Event Detection](/Event_Detection/)


# [7. Referee](/Referee/)


# [8. Coffin Corner Challenge](/Coffin-Corner-Challenge/)
