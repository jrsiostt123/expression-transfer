#!/bin/bash
# Downloads dlib's 68-point facial landmark model (~100MB)
echo "Downloading shape_predictor_68_face_landmarks.dat..."
wget -q --show-progress http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
echo "Extracting..."
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
echo "Done! Model saved to: shape_predictor_68_face_landmarks.dat"
