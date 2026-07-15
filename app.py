from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pipeline once when the app starts
pipeline = joblib.load("models/placement_pipeline.pkl")


def get_improvement_suggestions(data):
    cgpa = data["cgpa"][0]
    attendance = data["attendance_percentage"][0]
    internships = data["internships_count"][0]
    projects = data["projects_count"][0]
    backlogs = data["backlogs"][0]

    suggestions = []

    if cgpa < 7.5:
        suggestions.append("Improve your CGPA by focusing on core subjects and regular revision.")
    if attendance < 75:
        suggestions.append("Increase attendance, as consistent presence improves your academic profile.")
    if internships < 1:
        suggestions.append("Gain internships or practical experience to strengthen your resume.")
    if projects < 2:
        suggestions.append("Build more projects to show hands-on technical skills.")
    if backlogs > 0:
        suggestions.append("Clear backlogs to improve your academic standing and readiness.")

    if not suggestions:
        suggestions.append("Focus on strengthening technical projects and internship experience.")

    return suggestions


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Collect form data
    data = {
        "age": [21],   # Temporary fixed value
        "gender": [request.form["gender"]],
        "cgpa": [float(request.form["cgpa"])],
        "branch": [request.form["branch"]],
        "college_tier": ["Tier 1"],   # Temporary fixed value
        "internships_count": [int(request.form["internships_count"])],
        "projects_count": [int(request.form["projects_count"])],
        "certifications_count": [2],  # Temporary fixed value
        "coding_skill_score": [75],   # Defaulted value
        "aptitude_score": [65.0],    # Dataset-informed default
        "communication_skill_score": [68.0],  # Dataset-informed default
        "logical_reasoning_score": [70],  # Temporary fixed value
        "hackathons_participated": [1],   # Temporary fixed value
        "github_repos": [2],              # Temporary fixed value
        "linkedin_connections": [150],    # Temporary fixed value
        "mock_interview_score": [70],     # Temporary fixed value
        "attendance_percentage": [float(request.form["attendance_percentage"])],
        "backlogs": [int(request.form["backlogs"])],
        "extracurricular_score": [70],    # Temporary fixed value
        "leadership_score": [65],         # Temporary fixed value
        "volunteer_experience": ["Yes"],  # Temporary fixed value
        "sleep_hours": [7.0],             # Defaulted value
        "study_hours_per_day": [5.0]      # Defaulted value
    }

    df = pd.DataFrame(data)

    prediction = pipeline.predict(df)[0]

    if prediction == "Placed":
        result = "🎉 High Chance of Placement"
        suggestions = []
    else:
        result = "❌ Low Chance of Placement"
        suggestions = get_improvement_suggestions(data)

    return render_template("result.html", result=result, suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)