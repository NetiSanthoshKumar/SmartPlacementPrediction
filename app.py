from pathlib import Path
import os

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "models" / "placement_pipeline.pkl")

NUMERIC_FIELDS = {
    "age": (18, 24, int),
    "cgpa": (4.5, 10, float),
    "internships_count": (0, 8, int),
    "projects_count": (0, 13, int),
    "certifications_count": (0, 11, int),
    "coding_skill_score": (20, 100, float),
    "aptitude_score": (20, 100, float),
    "communication_skill_score": (20, 100, float),
    "logical_reasoning_score": (20, 100, float),
    "hackathons_participated": (0, 8, int),
    "github_repos": (0, 16, int),
    "linkedin_connections": (50, 999, int),
    "mock_interview_score": (20, 100, float),
    "attendance_percentage": (50, 100, float),
    "backlogs": (0, 6, int),
    "extracurricular_score": (0, 100, float),
    "leadership_score": (0, 100, float),
    "sleep_hours": (3, 10, float),
    "study_hours_per_day": (0.5, 10, float),
}
CHOICES = {
    "gender": {"Male", "Female"},
    "branch": {"CSE", "AIML", "DS", "IT", "ECE", "ME"},
    "college_tier": {"Tier 1", "Tier 2", "Tier 3"},
    "volunteer_experience": {"Yes", "No"},
}


def get_improvement_suggestions(data):
    suggestions = []

    if data["cgpa"] < 7.5:
        suggestions.append("Improve your CGPA by focusing on core subjects and regular revision.")
    if data["attendance_percentage"] < 75:
        suggestions.append("Increase attendance, as consistent presence improves your academic profile.")
    if data["internships_count"] < 1:
        suggestions.append("Gain internships or practical experience to strengthen your resume.")
    if data["projects_count"] < 2:
        suggestions.append("Build more projects to show hands-on technical skills.")
    if data["backlogs"] > 0:
        suggestions.append("Clear backlogs to improve your academic standing and readiness.")
    if data["coding_skill_score"] < 60 or data["mock_interview_score"] < 60:
        suggestions.append("Practice coding problems and mock interviews to improve technical readiness.")
    if data["certifications_count"] < 2:
        suggestions.append("Complete relevant certifications aligned with your target roles.")

    if not suggestions:
        suggestions.append("Focus on strengthening technical projects and internship experience.")

    return suggestions


def parse_form():
    values = {}
    errors = []
    for field, (minimum, maximum, converter) in NUMERIC_FIELDS.items():
        raw_value = request.form.get(field, "").strip()
        try:
            value = converter(raw_value)
            if not minimum <= value <= maximum:
                raise ValueError
            values[field] = value
        except (TypeError, ValueError):
            errors.append(f"{field.replace('_', ' ').title()} must be between {minimum} and {maximum}.")

    for field, allowed in CHOICES.items():
        value = request.form.get(field, "")
        if value not in allowed:
            errors.append(f"Please select a valid {field.replace('_', ' ')}.")
        else:
            values[field] = value
    return values, errors


def get_feature_drivers(row):
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if not model or not preprocessor:
        return []
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        return []
    top_features = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:5]
    return [name.replace("num__", "").replace("cat__", "").replace("_", " ").title() for name, _ in top_features]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data, errors = parse_form()
    if errors:
        return render_template("index.html", errors=errors, form=request.form), 400

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]
    classes = list(pipeline.classes_)
    placement_probability = float(probabilities[classes.index("Placed")]) * 100
    confidence = max(probabilities) * 100
    drivers = get_feature_drivers(df)

    if prediction == "Placed":
        result = "High Chance of Placement"
        suggestions = []
    else:
        result = "Low Chance of Placement"
        suggestions = get_improvement_suggestions(data)

    return render_template(
        "result.html",
        result=result,
        suggestions=suggestions,
        probability=round(placement_probability, 1),
        confidence=round(confidence, 1),
        drivers=drivers,
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )