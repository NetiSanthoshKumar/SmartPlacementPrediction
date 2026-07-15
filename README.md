"# SmartPlacementPrediction

A Flask-based web application that predicts whether a student is likely to get placed based on academic and skill-related features.

## Features
- Student placement prediction through a trained machine learning pipeline
- Simple web form for entering student details
- Result page that shows the prediction and improvement suggestions for low-probability cases

## Project Structure
- app.py: Flask application entry point
- templates/: HTML pages for the web interface
- static/: CSS assets
- src/: Data preprocessing, training, and prediction modules
- data/: Dataset used for training
- models/: Trained model files

## Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Run the Application
From the project root, run:

```bash
python app.py
```

Then open your browser and go to:

```text
http://127.0.0.1:5000/
```

## Usage
1. Enter the student details in the form.
2. Click Predict Placement.
3. Review the result and suggested improvement areas if the placement chance is low.

## Notes
- The project uses a saved scikit-learn pipeline for prediction.
- Make sure the model file exists in the models folder before running the app.
" 
