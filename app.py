# app.py
import os
import base64
import random
from datetime import datetime, date
from oct_checker import is_oct_image


from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, session, Response, abort, send_file, request as flask_request
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ===== إرسال الإيميل عبر Gmail SMTP للـ OTP =====
import smtplib
from email.mime.text import MIMEText

from oct_checker import is_oct_image

# ===============================
# Gmail SMTP Email config
# ===============================
GMAIL_ADDRESS = "deepsight.team@gmail.com"
GMAIL_APP_PASSWORD = "wdijxacjiabrcktk"  # App Password

def send_otp_email(to_email: str, otp_code: str) -> bool:
   
    try:
        subject = "DeepSight Login Verification Code"
        html_body = f"""
        <html>
        <body>
            <p>Dear Doctor,</p>
            <p>Your DeepSight verification code is:</p>
            <h2 style="letter-spacing:4px;">{otp_code}</h2>
            <p>This code is valid for a short time only. If you did not request this, please ignore this email.</p>
            <p>Best regards,<br>DeepSight System</p>
        </body>
        </html>
        """

        msg = MIMEText(html_body, "html")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        print("OTP email sent successfully to", to_email)
        return True
    except Exception as e:
        print("❌ OTP Email Error (Gmail):", repr(e))
        return False


# ===============================
# Flask & Database configuration
# ===============================
app = Flask(__name__)
app.secret_key = "X9v#4tLq8!pD2zR1mB7sH5wK0fU6yQ3j"  

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost:3306/deepsight_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ===============================
# Paths
# ===============================
STATIC_DIR  = os.path.join(app.root_path, "static")
UPLOAD_DIR  = os.path.join(STATIC_DIR, "uploads")
REPORTS_DIR = os.path.join(STATIC_DIR, "reports")
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}
def allowed_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ===============================
# Load DenseNet model
# ===============================
MODEL_PATH = os.path.join(app.root_path, "models", "best_model_fold_3.keras")
model = load_model(MODEL_PATH)

CLASSES_FOR_REPORT = ["DME", "Normal"]

# ===============================
# ORM Models
# ===============================
class Doctor(db.Model):
    __tablename__ = "Doctors"
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Doctor_ID = db.Column(db.String(20), unique=True, nullable=False)
    Doctor_Name = db.Column(db.String(100), nullable=False)
    Password = db.Column(db.String(100), nullable=False)
    Specialization = db.Column(db.String(50), nullable=False)
    Phone_Num = db.Column(db.String(15))
    Email = db.Column(db.String(100))
    Experience = db.Column(db.Integer)
    Hospital = db.Column(db.String(100))
    Profile_Image = db.Column(db.String(255))

class Patient(db.Model):
    __tablename__ = "Patients"
    ID            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Patient_ID    = db.Column(db.String(20), unique=True, nullable=False)
    Patient_Name  = db.Column(db.String(100), nullable=False)
    Gender        = db.Column(db.String(10), nullable=False)
    Date_Of_Birth = db.Column(db.Date, nullable=False)

class Diagnosis(db.Model):
    __tablename__ = "Diagnoses"
    ID               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Patient_Name     = db.Column(db.String(100), nullable=False)
    Patient_ID       = db.Column(db.String(20), db.ForeignKey("Patients.Patient_ID"), nullable=False)
    Doctor_Name      = db.Column(db.String(100), nullable=False)
    Date_Of_Scan     = db.Column(db.Date, nullable=False, default= datetime.now())
    Diagnosis_Result = db.Column(db.String(200))


# ===============================
# Grad-CAM helper
# ===============================
def build_heatmap(saved_path: str, x_batch: np.ndarray, cls_idx: int, label_text: str) -> str | None:
    """يبني Heatmap ويعيد اسم الملف داخل static/uploads، أو None عند الفشل."""
    try:
        last_conv = model.get_layer("conv5_block16_concat")
        heatmap_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = heatmap_model(x_batch)
            loss = preds[:, cls_idx]

        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat    = tf.reduce_mean(conv_out * pooled, axis=-1).numpy()[0]
        heat    = np.maximum(heat, 0)
        if heat.max() > 0:
            heat /= heat.max()

        orig = cv2.imread(saved_path)
        if orig is None:
            return None

        heat    = cv2.resize(heat, (orig.shape[1], orig.shape[0]))
        heat    = np.uint8(255 * heat)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR

        if (label_text or "").lower() == "normal":
            colored[:, :, 2] = (colored[:, :, 2] * 0.15).astype(np.uint8)

        overlay      = cv2.addWeighted(orig, 0.6, colored, 0.4, 0)
        heatmap_name = "heatmap_" + os.path.basename(saved_path)
        heatmap_path = os.path.join(UPLOAD_DIR, heatmap_name)
        ok = cv2.imwrite(heatmap_path, overlay)
        return heatmap_name if ok and os.path.exists(heatmap_path) else None
    except Exception as e:
        print("Grad-CAM error:", repr(e))
        return None


# ===============================
# Auth & basic pages
# ===============================
TEST_MODE =True
TEST_OTP="654321"
@app.get("/")
def index():
    # أول صفحة تظهر عند الدخول
    return redirect(url_for("starting"))

@app.get("/login")
def login():
    return render_template("Login.html")

@app.post("/login")
def login_post():
    doctor_id = (request.form.get("id") or "").strip()
    password = request.form.get("password") or ""
    if not doctor_id or not password:
        return "Missing ID or password", 400

    doc = Doctor.query.filter_by(Doctor_ID=doctor_id).first()
    if not doc or doc.Password != password:
        return "Invalid ID or password", 401

    session["doctor_id"] = doc.Doctor_ID
    session["doctor_name"] = doc.Doctor_Name
    session["doctor_specialty"] = doc.Specialization
    session["doctor_phone"] = doc.Phone_Num
    session["doctor_email"] = doc.Email
    session["doctor_experience"] = doc.Experience
    session["doctor_hospital"] = doc.Hospital
    if doc.Profile_Image:
        session["doctor_image"] = url_for(
            "static", filename=f"uploads/{doc.Profile_Image}"
        )
    else:
        session["doctor_image"] = None
    session["otp_verified"] = False

    otp = str(random.randint(100000, 999999))
    session["pending_otp"] = otp
    session["doctor_email"] = doc.Email 

    if doc.Email:
        sent_ok = send_otp_email(doc.Email, otp)
        if not sent_ok:
            print("DEBUG OTP (send failed):", otp)
    else:
        print("DEBUG OTP (no email in DB):", otp)

    print("DEBUG OTP (login, saved in session):", otp)

    return jsonify(ok=True, redirect=url_for("verify"))


@app.get("/verify")
def verify():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("VerifyOTP.html")


@app.post("/verify", endpoint="verify_otp_post")
def verify_post():
    raw_input = (request.form.get("otp") or "").strip()
    otp_input = "".join(ch for ch in raw_input if ch.isdigit())

    real_otp = session.get("pending_otp")
    real_otp_str = str(real_otp) if real_otp is not None else ""

    print(f"DEBUG VERIFY: otp_input='{otp_input}', real_otp='{real_otp_str}'")

    # --- وضع التجربة فقط ---
    if TEST_MODE:
        if otp_input == TEST_OTP:
            session["otp_verified"] = True
            # لا تزيل pending_otp هنا لكي لا يؤثر على النظام الحقيقي
            return jsonify(ok=True)
        else:
            return jsonify(ok=False, error="invalid_test_code"), 401
    # ------------------------

    # --- الوضع الحقيقي (بدون تغيير) ---
    if not real_otp_str:
        return jsonify(ok=False, error="expired"), 400

    if otp_input != real_otp_str:
        return jsonify(ok=False, error="invalid"), 401

    session["otp_verified"] = True
    session.pop("pending_otp", None)

    return jsonify(ok=True)

@app.get("/resend_otp")
def resend_otp():
    if "doctor_id" not in session:
        return jsonify(ok=False, error="Unauthorized"), 401

    new_otp = str(random.randint(100000, 999999))
    session["pending_otp"] = new_otp

    to_email = session.get("doctor_email")
    if to_email:
        sent_ok = send_otp_email(to_email, new_otp)
        if not sent_ok:
            print("DEBUG OTP (resend, send failed):", new_otp)
    else:
        print("DEBUG OTP (resend, no email in session):", new_otp)

    print("DEBUG OTP (resend, saved in session):", new_otp)

    return jsonify(ok=True)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("starting", msg="Logged out successfully ✅"))


# app.py

@app.get("/starting")
def starting():
    logged_in = "doctor_id" in session
    if logged_in:
        if not session.get("otp_verified"):
            return redirect(url_for("verify"))
            
    msg = flask_request.args.get("msg", "")
    return render_template("starting.html", logged_in=logged_in, message=msg)

# ===============================
# Upload → Predict → Save
# ===============================
@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        msg = flask_request.args.get("msg", "")
        return render_template("Upload.html", message=msg)

    # (1) بيانات المريض
    patient_name = (request.form.get("fname") or "").strip()
    patient_id   = (request.form.get("id") or "").strip()
    gender       = (request.form.get("gender") or "").strip()
    dob_raw      = (request.form.get("dateOfBirth") or "").strip()

    # (2) فحص وجود الصورة وصحة الامتداد
    f = request.files.get("uploadImage")
    if not f or f.filename == "":
        return render_template("Upload.html", message="OCT image is required ❌")

    if not allowed_ext(f.filename):
        return render_template("Upload.html", message="Only PNG/JPG images are allowed ❌")

    # (3) حفظ الصورة
    safe_name  = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(saved_path)

    # (4) التحقق من أن الصورة OCT 
    try:
        if not is_oct_image(saved_path):
            os.remove(saved_path)
            return render_template("Upload.html", message="This is not a valid OCT image ❌")
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return render_template("Upload.html", message="OCT verification failed ❌")

    # (5) تجهيز الصورة للموديل
    try:
        pil = Image.open(saved_path).convert("RGB").resize((224, 224))
    except:
        os.remove(saved_path)
        return render_template("Upload.html", message="Failed to process image ❌")

    x = np.asarray(pil, dtype="float32")
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)

    # (6) التوقع
    y       = model.predict(x, verbose=0)[0]
    cls_idx = int(np.argmax(y))
    label   = CLASSES_FOR_REPORT[cls_idx] if cls_idx < len(CLASSES_FOR_REPORT) else "None"
    score   = float(np.max(y))
    conf_percent = round(score * 100, 2)

    # Grad-CAM
    heatmap_name = build_heatmap(saved_path, x, cls_idx, label_text=label)

    # (7) حفظ/تحديث بيانات المريض مع تحقق صارم من الاسم والـ ID
    try:
        dob = datetime.strptime(dob_raw, "%Y-%m-%d").date()
    except ValueError:
        os.remove(saved_path)
        return render_template("Upload.html", message="Invalid date format ❌")

    patient = Patient.query.filter_by(Patient_ID=patient_id).first()
    if patient:
        if patient.Patient_Name != patient_name:
            os.remove(saved_path)
            return render_template(
                "Upload.html",
                message=f"Patient ID '{patient_id}' already exists with a different name ❌"
            )
        # يمكن تحديث بقية المعلومات مثل الجنس وتاريخ الميلاد
        patient.Gender        = gender
        patient.Date_Of_Birth = dob
    else:
        patient = Patient(
            Patient_ID=patient_id,
            Patient_Name=patient_name,
            Gender=gender,
            Date_Of_Birth=dob
        )
        db.session.add(patient)

    db.session.commit()

    # سجل التشخيص
    diag = Diagnosis(
        Patient_Name=patient_name,
        Patient_ID=patient_id,
        Doctor_Name=session.get("doctor_name", ""),
        Date_Of_Scan = datetime.now(),
        Diagnosis_Result=label
    )
    db.session.add(diag)
    db.session.commit()

    # تخزين بيانات آخر نتيجة
    session["last_diag_id"]       = diag.ID
    session["last_patient_id"]    = patient_id
    session["last_patient_name"]  = patient_name
    session["last_patient_gender"]= gender
    session["last_patient_dob"]   = dob_raw
    session["last_scan_time"]     = datetime.now().isoformat(timespec="minutes")
    session["last_image_name"]    = os.path.basename(saved_path)
    session["last_label"]         = label
    session["last_confidence"]    = conf_percent
    session["last_heatmap_name"]  = heatmap_name or ""

    return redirect(url_for("results"))

# ===============================
# Results & Report pages
# ===============================
@app.get("/results")
def results():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template(
        "Results.html",
        patient_id=session.get("last_patient_id"),
        patient_name=session.get("last_patient_name"),
        image_name=session.get("last_image_name"),
        label=session.get("last_label")
    )



@app.get("/report")
def report():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    doctor = Doctor.query.filter_by(Doctor_ID=session.get("doctor_id")).first()
    report_ctx = {
        "patientName": session.get("last_patient_name", "—"),
        "patientID":   session.get("last_patient_id", "—"),
        "gender":      session.get("last_patient_gender", "—"),
        "dob":         session.get("last_patient_dob", ""),
        "scan_time":   session.get("last_scan_time", "—"),
        "result":      session.get("last_label", "—"),
        "confidence":  session.get("last_confidence", "—"),
        "heatmapUrl":  url_for("static", filename=f"uploads/{session.get('last_heatmap_name','')}") if session.get("last_heatmap_name") else "",
        "octImage":    url_for("static", filename=f"uploads/{session.get('last_image_name','')}") if session.get("last_image_name") else ""
    }
    diag_id = session.get("last_diag_id")
    return render_template("ViewReport.html", report=report_ctx, doctor=doctor, diag_id=diag_id)


# ===============================
# Save report PDF
# ===============================
@app.post("/save_report_pdf")
def save_report_pdf():
    if "doctor_id" not in session:
        return "Unauthorized", 401

    diag_id  = (request.form.get("diag_id") or "").strip()
    pdf_file = request.files.get("pdf")
    if not diag_id or not pdf_file:
        return "Missing diag_id or pdf file", 400

    safe_name = f"report_{diag_id}.pdf"
    save_path = os.path.join(REPORTS_DIR, safe_name)
    pdf_file.save(save_path)
    return jsonify(ok=True, filename=safe_name)


# ===============================
# Download Report (PDF / HTML)
# ===============================
def _inline_css_into_html(html: str) -> str:
    try:
        css_path = os.path.join(STATIC_DIR, "Pages.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css_text = f.read()
        base_tag = f'<base href="{flask_request.url_root}">'
        injection = f"{base_tag}\n<style>{css_text}</style>"
        return html.replace("</head>", injection + "</head>", 1)
    except Exception as e:
        print("CSS inline error:", repr(e))
        return html

@app.get("/download_report/<int:diag_id>")
def download_report_by_id(diag_id: int):
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    pdf_name = f"report_{diag_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_name)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=pdf_name)

    d = Diagnosis.query.filter_by(ID=diag_id).first()
    if d is None:
        abort(404)

    p = Patient.query.filter_by(Patient_ID=d.Patient_ID).first()
    is_latest = (session.get("last_diag_id") == d.ID)
    heatmap_url = url_for("static", filename=f"uploads/{session.get('last_heatmap_name','')}") if is_latest and session.get("last_heatmap_name") else ""
    oct_url     = url_for("static", filename=f"uploads/{session.get('last_image_name','')}")     if is_latest and session.get("last_image_name")   else ""

    report_ctx = {
        "patientName": d.Patient_Name or (p.Patient_Name if p else "—"),
        "patientID":   d.Patient_ID,
        "gender":      (p.Gender if p else "—"),
        "dob":         (p.Date_Of_Birth.isoformat() if p else ""),
        "scan_time":   d.Date_Of_Scan.isoformat(),
        "result":      d.Diagnosis_Result or "—",
        "confidence":  session.get("last_confidence", "—") if is_latest else "—",
        "heatmapUrl":  heatmap_url,
        "octImage":    oct_url
    }

    html = render_template("ViewReport.html", report=report_ctx, doctor=None, diag_id=d.ID)
    html_inlined = _inline_css_into_html(html)

    download_html_name = f"DeepSight_Report_{d.Patient_ID}_{d.ID}.html"
    return Response(
        html_inlined,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{download_html_name}"'
        }
    )


# ===============================
# History page + API
# ===============================
@app.get("/history")
def history():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("History.html")

@app.get("/get_history")
def get_history():
    if "doctor_id" not in session:
        return jsonify([])

    rows = db.session.query(Diagnosis).order_by(Diagnosis.ID.desc()).all()
    out = []
    for r in rows:
        out.append({
            "diag_id":       r.ID,
            "patient_name":  r.Patient_Name,
            "Patient_ID":    r.Patient_ID,
            "doctor_name":   r.Doctor_Name,
            "scan_datetime": str(r.Date_Of_Scan),
            "result":        r.Diagnosis_Result or "—",
        })
    return jsonify(out)


# ===============================
# Support
# ===============================
@app.get("/support")
def support():
 logged_in = "doctor_id" in session
 return render_template("Support.html", logged_in=logged_in) 

# ===============================
# Profile
# ===============================
@app.get("/profile")
def profile():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    doctor = Doctor.query.filter_by(Doctor_ID=session["doctor_id"]).first()
    if not doctor:
        return "Doctor not found", 404
    doctor_image_url = url_for('static', filename=f"uploads/{doctor.Profile_Image}") if doctor.Profile_Image else None
    return render_template("Profile.html", doctor=doctor, doctor_image=doctor_image_url)
# ===============================
# Update Profile info
# ===============================
@app.post("/update_doctor_profile")
def update_doctor_profile():
    if "doctor_id" not in session:
        return jsonify(ok=False, error="Unauthorized"), 401
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")
    image_data = data.get("profile_image")
    doctor = Doctor.query.filter_by(Doctor_ID=session["doctor_id"]).first()
    if not doctor:
        return jsonify(ok=False, error="Doctor not found"), 404
    doctor.Email = email
    doctor.Phone_Num = phone
    # حفظ صورة بروفايل مصغّرة إذا أرسلت Base64
    if image_data and image_data.startswith("data:image"):
        import io
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB") 
        filename = f"doctor_{doctor.ID}.png"
        save_path = os.path.join(UPLOAD_DIR, filename)
        img.save(save_path, "PNG")
        doctor.Profile_Image = filename
        session["doctor_image"] = url_for("static", filename=f"uploads/{filename}")
    db.session.commit()
    return jsonify(ok=True)


# ===============================
# Run
# ===============================
