# Smart Placement Prediction

## Problem statement
This Flask application estimates whether a student is likely to be placed from academic, skill, experience, and activity information. It is an educational project for demonstrating an end-to-end machine-learning workflow, not a hiring decision system.

## Objectives
- Collect every feature used by the trained model.
- Apply the same preprocessing during training and prediction.
- Compare several classification algorithms with an untouched test set.
- Show probability, important model features, and practical improvement areas.

## Features
- Full validated student input form.
- Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting comparison.
- Accuracy, precision, recall, F1, ROC-AUC, classification report, and confusion matrix.
- Placement probability and model confidence.
- Feature importance for the selected model when supported.
- Recommendations phrased as potential improvement areas, not guaranteed causes.

## Dataset
`data/student_placement_prediction_dataset_2026.csv` contains 100,000 rows. The target is `placement_status`; `student_id` and `salary_package_lpa` are excluded from prediction. The input features are age, gender, CGPA, branch, college tier, internships, projects, certifications, coding, aptitude, communication, logical reasoning, hackathons, GitHub repositories, LinkedIn connections, mock interview score, attendance, backlogs, extracurricular score, leadership, volunteer experience, sleep, and study hours.

The training script writes `models/data_quality_report.csv`, including row count, duplicate count, missing-cell count, and class rates. It also compares the selected model with a majority-class baseline and runs three-fold stratified ROC-AUC validation. The dataset may contain weak or synthetic relationships, so a result near the baseline is possible and should be discussed honestly.

## Technologies
Python, pandas, scikit-learn, joblib, Flask, Jinja2, HTML, and CSS.

## Preprocessing and training
Categorical features use `OneHotEncoder(handle_unknown="ignore")`. Numeric features use `StandardScaler` inside the saved scikit-learn pipeline. A stratified 80/20 split is used for the comparison. The best model is selected by test ROC-AUC, then saved as `models/placement_pipeline.pkl`. The comparison and feature rankings are saved as CSV files.

## Project structure
```text
app.py                         Flask application and validation
data/                          Source CSV
models/                        Saved pipeline and generated reports
src/train_pipeline.py         Training, comparison, quality checks
templates/index.html           Full input form
templates/result.html          Prediction result
static/css/style.css           Simple responsive styling
INTERVIEW_QUESTIONS.md         Beginner-friendly interview preparation
```

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_pipeline.py
python app.py
```
Open `http://127.0.0.1:5000/`.

## Deploy on Render
1. Make sure `models/placement_pipeline.pkl` is committed to your Git repository. It is the small production model used by Flask. Do not commit `models/placement_model.pkl`; it is an obsolete 477 MB artifact.
2. Push the project to GitHub. Keep the CSV dataset private or excluded from the deployment repository because the running app does not need it.
3. In Render, choose **New +** -> **Web Service**, connect the GitHub repository, and select the project.
4. Set **Runtime** to `Python`, **Build Command** to `pip install -r requirements.txt`, and **Start Command** to `gunicorn app:app`.
5. Choose the free plan for a demonstration, create the service, and open the generated `.onrender.com` URL.

Render provides the `PORT` environment variable automatically. The application listens on `0.0.0.0` and uses that port in production. Do not use Flask's debug server for a public deployment.

## Deploy on Vercel
Vercel can run the Flask app as a serverless Python function using the included `vercel.json` file.

1. Commit and push the project to GitHub, including `vercel.json` and `models/placement_pipeline.pkl`.
2. Import the repository at [vercel.com](https://vercel.com/).
3. Keep the detected framework as `Other`.
4. Leave the build command empty. Vercel reads `requirements.txt` automatically.
5. Deploy and open the generated Vercel URL.

The app does not need the training CSV at runtime. Vercel has serverless execution limits, so it is appropriate for a portfolio demonstration rather than a long-running training service. Retrain models locally, commit only the small production pipeline, and redeploy.

### GitHub large-file warning
If GitHub refuses an old model because it is larger than 100 MB, remove it from Git tracking before pushing:
```bash
git rm --cached models/placement_model.pkl
git add .gitignore models/placement_pipeline.pkl
git commit -m "Prepare app for deployment"
git push
```
The old file can remain on your computer, but it is not needed by the deployed app.

## API
`GET /` displays the form. `POST /predict` accepts the form fields, validates them, sends a one-row DataFrame through the saved pipeline, and renders the result. Invalid or missing values return the form with HTTP 400 and friendly messages.

## Architecture
```text
Browser form -> Flask validation -> pandas DataFrame -> saved preprocessing/model pipeline
			-> class prediction + probability -> result page and recommendations
```

## Evaluation and limitations
The generated comparison table is `models/model_comparison.csv`. No accuracy should be assumed before running the trainer. Accuracy alone is insufficient; inspect recall, F1, ROC-AUC, class distribution, baseline, and confusion matrix. Feature importance indicates association used by the model, not causation. Probabilities are model estimates and are not automatically calibrated real-world chances.

## Screenshots
Run the application locally and add screenshots of the form and result page here for a portfolio submission.

## Future improvements
- Add probability calibration and a calibration curve.
- Tune thresholds using a validation set and explain the precision/recall tradeoff.
- Add automated tests for the schema and validation boundaries.
- Track model versions and dataset versions.
- Recheck fairness across demographic groups before any real-world use.
