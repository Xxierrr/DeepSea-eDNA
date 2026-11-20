import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "mlops" / "retraining_demo"
REPORT = DEMO / "evaluation_report.json"
OUTPUT = DEMO / "MLOps_Practical_Assessment_Report.pdf"

data = json.loads(REPORT.read_text(encoding="utf-8"))
validation = data["task_a_data_validation"]
champion = data["task_b_champion_evaluation"]["metrics"]
training = data["task_c_candidate_training"]
gate = data["task_d_quality_gate"]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#123B4A")))
styles.add(ParagraphStyle(name="SubtitleCenter", parent=styles["Heading2"], alignment=TA_CENTER, textColor=colors.HexColor("#4A6470")))


def table(rows, widths=None):
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7BC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return result


doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
story = []
story.extend([
    Spacer(1, 1.5*cm),
    Paragraph("MLOps Practical Assessment Report", styles["TitleCenter"]),
    Paragraph("DeepSea eDNA Model Monitoring and Retraining", styles["SubtitleCenter"]),
    Spacer(1, 0.7*cm),
    Paragraph("Project: DeepSea-eDNA", styles["Heading3"]),
    Paragraph(f"Evidence generated: {data['generated_at']}", styles["Normal"]),
    Paragraph("MLflow run: <b>9cc0fc6a2f434a1a8ec99cc49afdd665</b>", styles["Normal"]),
    Spacer(1, 0.8*cm),
])

story.append(Paragraph("1. Executive Result", styles["Heading2"]))
story.append(Paragraph(
    "The complete monitoring workflow was executed with a controlled smoke dataset. The incoming batch was validated, "
    "the bundled champion was evaluated, a candidate was trained and saved, the quality gate was executed, and the local "
    "MLflow store recorded a run. The candidate did not meet the quality gate, so the champion was retained and was not overwritten.",
    styles["Normal"],
))

story.append(Spacer(1, 0.3*cm))
story.append(table([
    ["Assessment item", "Result"],
    ["A. Incoming data", "Complete: 16 sequences, 8 balanced class labels"],
    ["B. Champion evaluation", "Complete: all classification metrics calculated"],
    ["C. Candidate retraining", "Complete: 56 combined sequences; separate candidate artifact"],
    ["D. Quality gate", "Complete: RETAIN_CHAMPION"],
    ["E. Demonstration", "Complete: report, CSV, weights, config and MLflow evidence"],
], [5.2*cm, 10.2*cm]))

story.append(PageBreak())
story.append(Paragraph("2. Task A - Prepare Incoming Data", styles["Heading2"]))
story.append(table([
    ["Check", "Observed value"],
    ["Dataset size", str(validation["size"])],
    ["Missing values", str(validation["missing_values"])],
    ["Duplicate sequences", str(validation["duplicate_sequences"])],
    ["Duplicate IDs", str(validation["duplicate_ids"])],
    ["Target distribution", ", ".join(f"class {k}: {v}" for k, v in validation["target_distribution"].items())],
], [5.2*cm, 10.2*cm]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Validation outcome", styles["Heading3"]))
story.append(Paragraph("Required sequence/label fields were present. The batch contained no missing values or duplicate IDs/sequences.", styles["Normal"]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("3. Task B - Existing Champion Evaluation", styles["Heading2"]))
story.append(table([
    ["Metric", "Champion result"],
    ["Accuracy", f"{champion['accuracy']:.4f}"],
    ["Weighted precision", f"{champion['precision_weighted']:.4f}"],
    ["Weighted recall", f"{champion['recall_weighted']:.4f}"],
    ["Weighted F1", f"{champion['f1_weighted']:.4f}"],
], [5.2*cm, 10.2*cm]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("The champion was loaded from <b>ml_models/marine_model</b>, used to predict the incoming labelled sequences, and evaluated with scikit-learn classification metrics.", styles["Normal"]))

story.append(PageBreak())
story.append(Paragraph("4. Task C - Candidate Retraining", styles["Heading2"]))
story.append(table([
    ["Item", "Result"],
    ["Existing smoke sample", "40 sequences"],
    ["Incoming batch", "16 sequences"],
    ["Combined training size", str(training["combined_training_size"])],
    ["Candidate directory", training["candidate_directory"]],
    ["Candidate epoch", str(training["training_results"].get("best_epoch", 0))],
], [5.2*cm, 10.2*cm]))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "The incoming batch was appended to a reduced existing training sample. The candidate used the project DNA tensor preprocessing "
    "(<b>A=1, C=2, G=3, T=4, N/other=0</b>) and was saved separately from the bundled champion. This demonstration used 1 epoch and small "
    "data samples as a smoke run; it is not intended as a production retraining result.",
    styles["Normal"],
))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("5. Task D - Quality Gate", styles["Heading2"]))
story.append(table([
    ["Gate item", "Value"],
    ["Selected metric", gate["metric"]],
    ["Champion score", f"{gate['champion_score']:.6f}"],
    ["Candidate score", f"{gate['candidate_score']:.6f}"],
    ["Minimum improvement", str(gate["minimum_improvement"])],
    ["Decision", gate["decision"]],
    ["Champion overwritten", str(gate["champion_overwritten"])],
], [5.2*cm, 10.2*cm]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Weighted F1 was selected because the project is a multiclass problem with uneven class counts. It considers both precision and "
    "recall across classes and gives lower-weighted influence to less frequent classes than macro F1. The candidate did not meet or exceed "
    "the champion score, so RETAIN_CHAMPION was recorded and the champion was never overwritten.",
    styles["Normal"],
))

story.append(PageBreak())
story.append(Paragraph("6. Execution and MLflow Evidence", styles["Heading2"]))
story.append(Paragraph("Reproducible command", styles["Heading3"]))
story.append(Paragraph("<b>.venv312\\Scripts\\python.exe scripts\\model_monitoring_retraining.py --epochs 1 --incoming-per-class 2 --existing-per-class 5 --batch-size 8 --mlflow-tracking-uri sqlite:///mlops/retraining_demo/mlflow.db --mlflow-experiment deepsea-assessment-monitoring --mlflow-model-name deepsea-assessment-candidate</b>", styles["Code"] if "Code" in styles else styles["Normal"]))

story.append(Spacer(1, 0.4*cm))
story.append(table([
    ["Artifact", "Location", "Status"],
    ["Evaluation report", "mlops/retraining_demo/evaluation_report.json", "Created"],
    ["Incoming summary", "mlops/retraining_demo/incoming_batch_summary.csv", "Created"],
    ["Candidate weights", "mlops/retraining_demo/candidate_model/best_model.pth", "Created"],
    ["Candidate config", "mlops/retraining_demo/candidate_model/model_config.json", "Created"],
    ["Candidate taxonomy", "mlops/retraining_demo/candidate_model/taxonomy_map.json", "Created"],
    ["Local MLflow store", "mlops/retraining_demo/mlflow.db", "Created"],
], [4.3*cm, 8.8*cm, 2.3*cm]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("MLflow verification", styles["Heading3"]))
story.append(Paragraph("Run ID: <b>9cc0fc6a2f434a1a8ec99cc49afdd665</b><br/>Registered model version: <b>1</b><br/>Champion weighted F1: <b>0.933333</b><br/>Candidate weighted F1: <b>0.027778</b>", styles["Normal"]))

story.append(PageBreak())
story.append(Paragraph("7. Environment and Remaining Work", styles["Heading2"]))
story.append(Paragraph("Environment installed", styles["Heading3"]))
story.append(Paragraph("Python 3.12.10 was installed, a <b>.venv312</b> virtual environment was created, CPU PyTorch and project dependencies were installed, and ReportLab was installed for PDF generation.", styles["Normal"]))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Remaining items for a stricter production-grade submission", styles["Heading3"]))
story.append(table([
    ["Remaining item", "Why it matters", "Recommended next step"],
    ["Run at least 3 epochs", "The evidence run was a fast 1-epoch smoke test", "Run 3-10 epochs on a larger but reproducible sample"],
    ["Run a representative incoming batch", "Only 16 smoke sequences were available", "Use a larger external labelled batch from the same data schema"],
    ["Test an eligible candidate", "The executed candidate was rejected", "Only promote after weighted F1 exceeds champion + improvement"],
    ["Use full MLflow server", "Evidence uses local SQLite MLflow", "Start MLflow service and point MLFLOW_TRACKING_URI to it"],
    ["Improve candidate training", "The compatibility smoke trainer is intentionally simple", "Run the existing advanced trainer with sufficient CPU time and validate its Windows compatibility"],
], [4.5*cm, 5.5*cm, 5.4*cm]))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Conclusion", styles["Heading3"]))
story.append(Paragraph(
    "The practical assessment implementation and demonstration are complete for the requested workflow. The one remaining distinction is that "
    "the evidence is a small CPU smoke demonstration, not a production-quality candidate training run. This distinction is preserved in the report.",
    styles["Normal"],
))

doc.build(story)
print(OUTPUT)
