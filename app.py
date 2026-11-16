import json
import urllib.request
import base64
import io as _io
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(
    page_title="Moroccan Food Classifier — Hamza Oulbayyad",
    page_icon="🍲",
    layout="centered"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', sans-serif;
  }
  .stApp {
      background: radial-gradient(circle at 50% -10%, #172554 0%, #090d16 55%, #03050a 100%);
      color: #f8fafc;
  }
  .hero-banner {
      position: relative;
      background: linear-gradient(115deg, #431407 0%, #9a3412 35%, #ea580c 72%, #f59e0b 100%);
      border: 1px solid rgba(251, 146, 60, 0.45);
      border-radius: 26px;
      padding: 36px 28px;
      text-align: center;
      color: white;
      margin-bottom: 28px;
      box-shadow: 0 22px 45px -12px rgba(234, 88, 12, 0.4);
      overflow: hidden;
  }
  .hero-banner::before {
      content: "";
      position: absolute;
      inset: 0;
      background-image: radial-gradient(rgba(255, 255, 255, 0.12) 1px, transparent 1px),
                        radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
      background-size: 28px 28px;
      background-position: 0 0, 14px 14px;
      opacity: 0.45;
      pointer-events: none;
  }
  .hero-title-row {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      margin-bottom: 10px;
  }
  .hero-icon-box {
      background: rgba(255, 255, 255, 0.18);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.35);
      border-radius: 16px;
      width: 52px;
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.7rem;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  }
  .hero-title {
      margin: 0;
      font-size: 2.15rem;
      font-weight: 800;
      color: #ffffff;
      text-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
      letter-spacing: -0.5px;
  }
  .hero-sub {
      position: relative;
      margin: 0 0 18px 0;
      font-size: 1.05rem;
      color: #ffedd5;
      font-weight: 500;
  }
  .hero-creator-pill {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.16);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.45);
      border-radius: 999px;
      padding: 8px 24px;
      font-size: 0.96rem;
      color: #ffffff;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 6px;
      margin-bottom: 12px;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 10px 22px;
      color: #94a3b8;
      font-weight: 700;
      transition: all 0.25s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
      color: #ffffff;
      border-color: rgba(251, 146, 60, 0.4);
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)) !important;
      border-color: rgba(251, 146, 60, 0.5) !important;
      border-bottom: 2.5px solid #fb923c !important;
      color: #ffffff !important;
      box-shadow: 0 8px 20px -6px rgba(234, 88, 12, 0.35);
  }
  [data-testid="stFileUploader"] section {
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(9, 13, 22, 0.85) 100%) !important;
      border: 1.5px dashed rgba(148, 163, 184, 0.35) !important;
      border-radius: 22px !important;
      padding: 28px 20px !important;
      transition: all 0.25s ease !important;
  }
  [data-testid="stFileUploader"] section:hover {
      border-color: #fb923c !important;
      background: rgba(30, 27, 75, 0.45) !important;
      box-shadow: 0 12px 30px -10px rgba(234, 88, 12, 0.25) !important;
  }
  [data-testid="stFileUploader"] section button {
      background: rgba(234, 88, 12, 0.18) !important;
      border: 1px solid rgba(251, 146, 60, 0.5) !important;
      border-radius: 12px !important;
      color: #fdba74 !important;
      font-weight: 700 !important;
      padding: 6px 18px !important;
  }
  [data-testid="stFileUploader"] section button:hover {
      background: #ea580c !important;
      color: #ffffff !important;
  }
  .preview-card {
      position: relative;
      background: rgba(15, 23, 42, 0.8);
      border: 1.5px solid rgba(251, 146, 60, 0.35);
      border-radius: 22px;
      padding: 10px;
      margin: 14px 0 18px 0;
      box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.65);
  }
  .preview-card img {
      width: 100%;
      border-radius: 15px;
      display: block;
      max-height: 360px;
      object-fit: contain;
  }
  .bar-wrap {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 18px;
      padding: 14px 18px;
      margin: 10px 0;
      direction: rtl;
  }
  .bar-label {
      font-size: 1.12rem;
      font-weight: 700;
      margin-bottom: 8px;
      color: #fdba74;
  }
  .bar-track {
      background: rgba(2, 6, 23, 0.85);
      border-radius: 999px;
      height: 22px;
      width: 100%;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.06);
      padding: 2px;
  }
  .bar-fill {
      height: 16px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      padding-right: 12px;
      justify-content: flex-end;
      color: white;
      font-size: 0.85rem;
      font-weight: 800;
      min-width: 52px;
      transition: width 0.8s ease;
  }
  .rank-1 .bar-fill { background: linear-gradient(90deg, #c2410c, #ea580c, #fb923c); }
  .rank-2 .bar-fill { background: linear-gradient(90deg, #9a3412, #d97706, #fcd34d); }
  .rank-3 .bar-fill { background: linear-gradient(90deg, #334155, #475569, #94a3b8); }
  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(30, 41, 59, 0.85) !important;
      border: 1px solid rgba(255, 255, 255, 0.14) !important;
      color: #ffffff !important;
      padding: 11px 20px !important;
      transition: all 0.25s ease !important;
      width: 100% !important;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #16a34a, #15803d) !important;
      border-color: #4ade80 !important;
      transform: translateY(-2px) !important;
      box-shadow: 0 10px 20px -5px rgba(22, 163, 74, 0.45) !important;
  }
  .st-key-predict_dish_btn button {
      background: linear-gradient(90deg, #c2410c 0%, #ea580c 50%, #f97316 100%) !important;
      border: 1.5px solid #fdba74 !important;
      border-radius: 999px !important;
      font-size: 1.05rem !important;
      font-weight: 800 !important;
      padding: 14px 24px !important;
      box-shadow: 0 10px 25px -5px rgba(234, 88, 12, 0.55) !important;
  }
  .st-key-predict_dish_btn button:hover {
      background: linear-gradient(90deg, #9a3412 0%, #c2410c 50%, #ea580c 100%) !important;
      box-shadow: 0 14px 32px -3px rgba(251, 146, 60, 0.7) !important;
  }
  .st-key-wrong_food_btn button:hover {
      background: linear-gradient(135deg, #dc2626, #991b1b) !important;
      border-color: #f87171 !important;
      box-shadow: 0 10px 20px -5px rgba(220, 38, 38, 0.55) !important;
  }
  .glass-footer {
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      padding: 14px 22px;
      margin-top: 38px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 14px;
      font-size: 0.85rem;
      color: #94a3b8;
      text-align: center;
  }
  .footer-sep {
      color: rgba(255, 255, 255, 0.15);
  }
  .footer-link {
      color: #fb923c !important;
      text-decoration: none !important;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: color 0.2s ease;
  }
  .footer-link:hover {
      color: #fdba74 !important;
      text-decoration: underline !important;
  }
</style>
""", unsafe_allow_html=True)

def upload_image_to_github(img_bytes: bytes, file_path: str, commit_message: str):
    token = st.secrets["GITHUB_TOKEN"]
    repo  = st.secrets["GITHUB_REPO"]
    url   = f"https://api.github.com/repos/{repo}/contents/{file_path}"

    img = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64_content = base64.b64encode(buf.getvalue()).decode("utf-8")

    data = json.dumps({
        "message": commit_message,
        "content": b64_content
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=data, method="PUT",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        })
    try:
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read())
            return res_data["content"]["html_url"]
    except Exception:
        return None

def create_github_issue(wrong: str, correct: str, img_bytes: bytes):
    token = st.secrets["GITHUB_TOKEN"]
    repo  = st.secrets["GITHUB_REPO"]
    issues_url = f"https://api.github.com/repos/{repo}/issues"

    file_name = f"correction_{int(time.time())}.jpg"
    file_path = f"training_corrections/{file_name}"

    img_url = upload_image_to_github(
        img_bytes,
        file_path,
        commit_message=f"Add training image for correct dish: {correct} (Predicted: {wrong})"
    )

    image_md = f"[🔗 View & Download Training Image]({img_url})" if img_url else "*Failed to upload image to repository*"

    data = json.dumps({
        "title": f"Correction: {wrong} ← {correct}",
        "body": (
            f"**Model Prediction:** {wrong}\n\n"
            f"**Correct Dish:** {correct}\n\n"
            f"**Training Image:** {image_md}\n\n"
            f"*Automatic correction from user*"
        ),
        "labels": ["correction", "training-data"],
    }).encode("utf-8")

    req = urllib.request.Request(
        issues_url, data=data, method="POST",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        })
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())["html_url"]

@st.cache_resource
def load_assets():
    model = keras.models.load_model("food_model_v1.keras")
    with open("food_classes.json", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

AR = {
    "traditional macaroon": "ماكرون تقليدي", "tomatoes and onion salad": "سلطة الطماطم والبصل",
    "TAKTOUKA": "تكتوكة", "zitoun": "زيتون", "sweet bread": "خبز حلو", "tkalya": "تقلية",
    "sellou": "سلّو", "amlou": "أملو", "sfenje": "سفنج", "snowballs": "كرات الثلج",
    "traditional bread": "خبز تقليدي", "white beans with tomatoes": "لوبيا بيضاء بالطماطم",
    "tagine with quince": "طاجين بالسفرجل", "tagine with vegetables": "طاجين بالخضار",
    "tagine with artichokes and peas": "طاجين بالخرشوف والبازلاء", "seffa with rice": "سفة بالأرز",
    "crackers with almonds (ghriba)": "غريبة باللوز", "nougat": "نوغا", "fekkas": "فقاص",
    "better beldi": "زبدة بلدية", "chebakia": "شباكية", "gazelle horn": "كعب الغزال",
    "BISSARA(feves puree)": "بيصارة", "maakouda": "معقودة", "FEVES with sauce": "فول بالصلصة",
    "chicken with potatoes and olives": "دجاج بالبطاطس والزيتون", "bahla": "بهلة",
    "Feet of beef": "كرعين البقر", "lentils": "عدس", "briouate  with almonds": "بريوات باللوز",
    "briouate with almonds": "بريوات باللوز", "meatball with tomato sauce": "كفتة بصلصة الطماطم",
    "salmon": "سلمون", "jam": "مربى", "basbousa": "بسبوسة", "apple": "تفاح", "banana": "موز",
    "batbout": "بطبوط", "beghrir": "بغرير", "caesar salad": "سلطة سيزر",
    "chicken basstila": "بسطيلة بالدجاج", "chicken nuggets": "ناجتس الدجاج",
    "chocolate cake": "كعكة الشوكولاتة", "couscous": "كسكس", "croissant": "كرواسون",
    "dates": "تمر", "fish and vegetables": "سمك بالخضار", "fish basstila": "بسطيلة بالسمك",
    "french fries": "بطاطس مقلية", "fried calamari": "حبّار مقلي", "hamburger": "همبرغر",
    "harcha": "حرشة", "harira": "حريرة", "lasagne": "لازانيا", "meat brochettes": "أسياخ لحم مشوي",
    "msamen": "مسمن", "omelette": "عجّة (أومليت)", "orange": "برتقال", "paella": "باييا",
    "pear": "إجاص", "pizza": "بيتزا", "rfissa": "رفيسة", "roasted chicken": "دجاج مشوي",
    "seffa": "سفة", "spagetti": "سباغيتي", "tagine with beef": "طاجين باللحم", "zaalouk": "زعلوك",
    "mhancha": "محنشة", "rghayf": "رغايف", "kaak": "كعك", "liver with sauce": "كبدة بالصلصة",
}
names_ar = [AR.get(c, c) for c in class_names]

for key in ("top_indices", "probs", "feedback_sent", "img_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("""
<div class="hero-banner">
  <div class="hero-title-row">
    <div class="hero-icon-box">🍲</div>
    <h1 class="hero-title">Moroccan Food Classifier</h1>
  </div>
  <p class="hero-sub">📸 Upload or take a photo of a Moroccan dish</p>
  <span class="hero-creator-pill">
    ✨ Created by: <b>Hamza Oulbayyad</b>
  </span>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Live Camera"])
with tab_upload:
    uploaded_file = st.file_uploader(
        "Drag & Drop your image here", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed")
with tab_camera:
    camera_photo = st.camera_input("Take a photo with the camera", label_visibility="collapsed")

img_source = uploaded_file or camera_photo

if img_source is not None:
    image = Image.open(img_source)
    image = ImageOps.exif_transpose(image)
    prev_buf = _io.BytesIO()
    image.convert("RGB").save(prev_buf, format="JPEG", quality=90)
    prev_b64 = base64.b64encode(prev_buf.getvalue()).decode("utf-8")

    st.markdown(f"""
    <div class="preview-card">
      <img src="data:image/jpeg;base64,{prev_b64}" alt="Dish Preview" />
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍  What is this dish?", key="predict_dish_btn", width="stretch"):
        with st.spinner("Analyzing the dish..."):
            img_r = image.convert("RGB").resize((224, 224))
            x = np.expand_dims(np.array(img_r, dtype="float32"), axis=0)
            probs = model.predict(x, verbose=0)[0]
        st.session_state.probs = probs
        st.session_state.top_indices = np.argsort(probs)[::-1][:3].tolist()
        st.session_state.feedback_sent = False
        buf = _io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        st.session_state.img_bytes = buf.getvalue()

if st.session_state.top_indices is not None:
    medals = ["🥇", "🥈", "🥉"]
    ranks  = ["rank-1", "rank-2", "rank-3"]
    st.markdown("### Top 3 Predictions")

    for i, idx in enumerate(st.session_state.top_indices):
        pct = st.session_state.probs[idx] * 100
        st.markdown(f"""
        <div class="bar-wrap {ranks[i]}">
          <div class="bar-label">{medals[i]} {names_ar[idx]}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%">{pct:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

    if not st.session_state.feedback_sent:
        st.divider()
        st.markdown("**Is the top prediction correct? / هل التوقع الأول صحيح؟**")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("✅ Yes, Correct / نعم، صحيح", width="stretch"):
                st.session_state.feedback_sent = "correct"
                st.rerun()
        with col2:
            if st.button("❌ No, Wrong / لا، خطأ", key="wrong_food_btn", width="stretch"):
                st.session_state.feedback_sent = "wrong"
                st.rerun()

    elif st.session_state.feedback_sent == "correct":
        st.success("Great! Thanks for confirming 🎉 / ممتاز! شكراً على تأكيدك")

    elif st.session_state.feedback_sent == "wrong":
        st.divider()
        st.markdown("**What is the correct dish? / ما اسم الطبق الصحيح؟**")
        fix = st.selectbox(
            "Select from the list",
            options=[""] + names_ar,
            index=0,
            label_visibility="collapsed")

        if fix and st.button("📨 Submit Correction / إرسال التصحيح", width="stretch"):
            wrong_name = names_ar[st.session_state.top_indices[0]]
            with st.spinner("Saving image and creating GitHub issue..."):
                try:
                    create_github_issue(
                        wrong_name, fix,
                        st.session_state.img_bytes)
                    st.session_state.feedback_sent = "done"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    elif st.session_state.feedback_sent == "done":
        st.success("Thank you! Image saved and correction documented on GitHub successfully 🎉")
        st.info("You can use these images later to retrain your model.")

st.markdown("""
<div class="glass-footer">
  <span>ℹ️ This model recognizes <b>70 Moroccan dishes</b> only.</span>
  <span class="footer-sep">|</span>
  <span><b>Model Version:</b> v1.3</span>
  <span class="footer-sep">|</span>
  <span><b>Hamza Oulbayyad</b> — MFOOD_70</span>
  <span class="footer-sep">|</span>
  <a href="https://github.com/exora999/moroccan-food-classifier" target="_blank" class="footer-link">
    GitHub Repository ↗
  </a>
</div>
""", unsafe_allow_html=True)
