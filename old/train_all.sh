#!/bin/sh

# train events, then ball, then table (so I can run frame_folders without erasing existing data)

# ! EVENTS
cd ../Data_Cleaning
python frame_folders.py --model_type=Events

cd ../Modeling
python event_detection.py


# ! BALL
cd ../Data_Cleaning
python frame_folders.py --model_type=Ball

cd ../Modeling
python ball_present.py
python ball_location.py


# ! TABLE
cd ../Data_Cleaning
python frame_folders.py --model_type=Table

cd ../Modeling
python table_detection.py
cd ..
