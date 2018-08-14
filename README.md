# face-recognition-assignment

This is the given task or assignment for the role of data science intern at absolute face

## Requirements

- Python 3.6
- OpenCV 3.3.0 or above
- Numpy 1.14.3 or above

## Main 3 steps of this project:

1. Gathering the dataset from webcam.(also you can skip it if you have the data already)
2.Train a model from the acquired dataset and save the model.
3.Use the trained model to classify faces in realtime.

## Steps to run this on your system.
This must work with OSX and Linux. I haven't tried windows yet.

**Step 1**. Create directories saved_model and training_data. Although it will be added automatically if you forget to create.
Run the face_datasets.py file on your terminal. Before running it set the **face_id** (on line 23) to some integer value for each time
you run it. This sets the labels.

**Step 2**.Start training the model with the obtained dataset by running the training.py.

**Step 3**: Run the face_recognition.py file to start detection. But before running that don't forget to change the names of ids from
line 59.(Some names are already set since I ran it)

See the results.

Reference: OpenCV Documentation


