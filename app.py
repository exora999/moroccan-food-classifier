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
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Cairo:wght@600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
  }
  .stApp {
      background: radial-gradient(circle at 50% 0%, #111c44 0%, #070b19 60%, #03050c 100%);
      color: #f8fafc;
  }
  .block-container {
      max-width: 860px !important;
      padding-top: 2rem !important;
  }
  .hero-banner {
      position: relative;
      background: linear-gradient(95deg, #3b1219 0%, #8c2d19 32%, #ea580c 72%, #f59e0b 100%);
      border: 1px solid rgba(251, 146, 60, 0.45);
      border-radius: 24px;
      padding: 32px 24px;
      text-align: center;
      color: white;
      margin-bottom: 26px;
      box-shadow: 0 20px 45px -12px rgba(234, 88, 12, 0.45);
      overflow: hidden;
  }
  .hero-pattern {
      position: absolute;
      inset: 0;
      background-image: radial-gradient(rgba(255, 255, 255, 0.14) 1.2px, transparent 1.2px);
      background-size: 22px 22px;
      opacity: 0.35;
      pointer-events: none;
  }
  .hero-svg-left {
      position: absolute;
      left: 14px;
      bottom: -6px;
      width: 145px;
      height: 145px;
      pointer-events: none;
      opacity: 0.95;
  }
  .hero-svg-right {
      position: absolute;
      right: 12px;
      bottom: 0;
      width: 155px;
      height: 145px;
      pointer-events: none;
      opacity: 0.35;
  }
  .hero-content {
      position: relative;
      z-index: 2;
  }
  .hero-title-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      margin-bottom: 8px;
  }
  .hero-utensils {
      background: linear-gradient(135deg, #ff7e29, #ea580c);
      border: 1.5px solid rgba(255, 255, 255, 0.45);
      border-radius: 14px;
      width: 48px;
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
      flex-shrink: 0;
  }
  .hero-title {
      margin: 0;
      font-size: 2.1rem;
      font-weight: 800;
      color: #ffffff;
      text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
  }
  .hero-sub {
      margin: 0 0 18px 0;
      font-size: 1.02rem;
      color: #ffedd5;
      font-weight: 500;
  }
  .hero-pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      background: linear-gradient(90deg, rgba(251, 146, 60, 0.45), rgba(245, 158, 11, 0.55));
      backdrop-filter: blur(8px);
      border: 1.5px solid rgba(255, 255, 255, 0.65);
      border-radius: 999px;
      padding: 9px 26px;
      font-size: 0.96rem;
      color: #ffffff;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 6px;
      margin-bottom: 12px;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px 14px 4px 4px;
      padding: 10px 22px;
      color: #94a3b8;
      font-weight: 700;
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)) !important;
      border-color: rgba(251, 146, 60, 0.5) !important;
      border-bottom: 3px solid #f97316 !important;
      color: #ffffff !important;
  }
  [data-testid="stFileUploader"] section {
      background: linear-gradient(180deg, rgba(13, 21, 45, 0.85) 0%, rgba(8, 13, 30, 0.92) 100%) !important;
      border: 1.5px dashed rgba(99, 102, 241, 0.35) !important;
      border-radius: 20px !important;
      padding: 26px 20px !important;
      transition: all 0.25s ease !important;
  }
  [data-testid="stFileUploader"] section:hover {
      border-color: #fb923c !important;
      background: rgba(23, 32, 66, 0.9) !important;
  }
  [data-testid="stFileUploader"] section button {
      background: rgba(30, 41, 59, 0.9) !important;
      border: 1px solid rgba(251, 146, 60, 0.5) !important;
      border-radius: 12px !important;
      color: #fdba74 !important;
      font-weight: 700 !important;
  }
  .preview-card {
      background: rgba(13, 21, 45, 0.85);
      border: 1.5px solid rgba(251, 146, 60, 0.4);
      border-radius: 20px;
      padding: 10px;
      margin: 14px 0 18px 0;
      box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.65);
  }
  .preview-card img {
      width: 100%;
      border-radius: 14px;
      display: block;
      max-height: 340px;
      object-fit: contain;
  }
  .pred-panel {
      background: linear-gradient(180deg, rgba(13, 22, 52, 0.85) 0%, rgba(8, 14, 34, 0.92) 100%);
      border: 1px solid rgba(59, 130, 246, 0.22);
      border-radius: 24px;
      padding: 24px;
      margin-top: 22px;
      box-shadow: 0 20px 45px -12px rgba(0, 0, 0, 0.7);
  }
  .pred-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
  }
  .pred-header-left {
      display: flex;
      align-items: center;
      gap: 14px;
  }
  .pred-icon-box {
      width: 48px;
      height: 48px;
      border-radius: 14px;
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
      flex-shrink: 0;
  }
  .pred-title {
      font-size: 1.5rem;
      font-weight: 800;
      color: #ffffff;
      margin: 0;
      line-height: 1.2;
  }
  .pred-title span {
      color: #fb7185;
      background: linear-gradient(90deg, #fb923c, #f43f5e);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }
  .pred-subtitle {
      font-size: 0.88rem;
      color: #94a3b8;
      margin: 2px 0 0 0;
  }
  .pred-row {
      display: flex;
      align-items: center;
      gap: 16px;
      border-radius: 18px;
      padding: 14px 20px;
      margin-bottom: 14px;
      background: rgba(11, 19, 43, 0.85);
      transition: transform 0.2s ease;
  }
  .pred-row:hover {
      transform: translateX(3px);
  }
  .pred-row-head {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
  }
  .row-rank-1 {
      border: 1.5px solid rgba(249, 115, 22, 0.65);
      box-shadow: inset 0 0 24px rgba(249, 115, 22, 0.12), 0 6px 18px rgba(0, 0, 0, 0.35);
  }
  .row-rank-2 {
      border: 1.5px solid rgba(56, 189, 248, 0.55);
      box-shadow: inset 0 0 24px rgba(56, 189, 248, 0.1), 0 6px 18px rgba(0, 0, 0, 0.35);
  }
  .row-rank-3 {
      border: 1.5px solid rgba(168, 85, 247, 0.55);
      box-shadow: inset 0 0 24px rgba(168, 85, 247, 0.1), 0 6px 18px rgba(0, 0, 0, 0.35);
  }
  .medal-badge {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.1rem;
      color: #ffffff;
      flex-shrink: 0;
  }
  .medal-1 {
      background: radial-gradient(circle at 30% 30%, #fde047, #f59e0b 65%, #b45309);
      box-shadow: 0 0 16px rgba(245, 158, 11, 0.55);
      color: #451a03;
  }
  .medal-2 {
      background: radial-gradient(circle at 30% 30%, #f1f5f9, #94a3b8 65%, #475569);
      box-shadow: 0 0 16px rgba(148, 163, 184, 0.45);
      color: #0f172a;
  }
  .medal-3 {
      background: radial-gradient(circle at 30% 30%, #fdba74, #d97706 65%, #78350f);
      box-shadow: 0 0 16px rgba(217, 119, 6, 0.45);
      color: #451a03;
  }
  .v-divider {
      width: 3px;
      height: 34px;
      border-radius: 99px;
      flex-shrink: 0;
  }
  .div-1 { background: #f97316; }
  .div-2 { background: #38bdf8; }
  .div-3 { background: #a855f7; }
  .dish-info {
      min-width: 155px;
      max-width: 185px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      font-family: 'Cairo', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      color: #ffffff;
      text-align: right;
  }
  .p-track {
      flex-grow: 1;
      width: 100%;
      height: 24px;
      background: rgba(23, 37, 74, 0.9);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .p-fill {
      height: 100%;
      border-radius: 999px;
      display: flex;
      align-items: center;
      padding-left: 12px;
      font-size: 0.86rem;
      font-weight: 800;
      color: #ffffff;
      min-width: 62px;
      transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .fill-1 { background: linear-gradient(90deg, #ea580c, #f97316, #fbbf24); }
  .fill-2 { background: linear-gradient(90deg, #0284c7, #38bdf8); }
  .fill-3 { background: linear-gradient(90deg, #7e22ce, #a855f7); }
  .pred-footer-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      background: rgba(15, 26, 58, 0.75);
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 14px;
      padding: 10px 16px;
      margin-top: 16px;
      font-size: 0.85rem;
      color: #94a3b8;
  }
  .conf-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(88, 28, 135, 0.45);
      border: 1px solid rgba(192, 132, 252, 0.4);
      color: #f3e8ff;
      padding: 4px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.8rem;
      flex-shrink: 0;
  }
  @media (max-width: 640px) {
      .hero-banner {
          padding: 24px 16px;
      }
      .hero-title {
          font-size: 1.55rem;
      }
      .hero-utensils {
          width: 40px;
          height: 40px;
      }
      .hero-svg-left {
          width: 85px;
          height: 85px;
          left: 6px;
          bottom: -4px;
          opacity: 0.3;
      }
      .hero-svg-right {
          width: 90px;
          height: 85px;
          right: 6px;
          opacity: 0.25;
      }
      .hero-pill {
          padding: 7px 16px;
          font-size: 0.85rem;
      }
      .pred-panel {
          padding: 16px;
      }
      .pred-title {
          font-size: 1.25rem;
      }
      .pred-row {
          flex-direction: column;
          align-items: stretch;
          gap: 12px;
          padding: 14px 16px;
      }
      .pred-row-head {
          width: 100%;
          justify-content: space-between;
      }
      .dish-info {
          min-width: unset;
          max-width: unset;
          flex-grow: 1;
          font-size: 1.08rem;
      }
      .pred-footer-bar {
          flex-direction: column;
          text-align: center;
      }
  }
  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(20, 30, 60, 0.85) !important;
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
      background: rgba(13, 21, 45, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      padding: 14px 22px;
      margin-top: 34px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 14px;
      font-size: 0.84rem;
      color: #94a3b8;
      text-align: center;
  }
  .footer-link {
      color: #fb923c !important;
      text-decoration: none !important;
      font-weight: 700;
  }
  .footer-link:hover {
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
  <div class="hero-pattern"></div>
  <svg class="hero-svg-left" viewBox="0 0 160 160" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M58 45 C54 32, 66 24, 62 12" stroke="rgba(255,255,255,0.45)" stroke-width="3" stroke-linecap="round"/>
    <path d="M74 42 C70 28, 82 20, 78 8" stroke="rgba(255,255,255,0.35)" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M22 108 C22 132, 138 132, 138 108 L130 100 L30 100 Z" fill="#9a3412" stroke="#fdba74" stroke-width="2"/>
    <ellipse cx="80" cy="100" rx="52" ry="12" fill="#f59e0b"/>
    <circle cx="65" cy="99" r="6" fill="#dc2626"/>
    <circle cx="82" cy="101" r="7" fill="#16a34a"/>
    <circle cx="98" cy="98" r="5" fill="#fef08a"/>
    <path d="M32 95 L72 34 C76 28, 84 28, 88 34 L128 95 Z" fill="url(#tajineGrad)" stroke="#fed7aa" stroke-width="2"/>
    <circle cx="80" cy="26" r="8" fill="#c2410c" stroke="#fed7aa" stroke-width="2"/>
    <defs>
      <linearGradient id="tajineGrad" x1="32" y1="30" x2="128" y2="95" gradientUnits="userSpaceOnUse">
        <stop stop-color="#ea580c"/>
        <stop offset="1" stop-color="#7c2d12"/>
      </linearGradient>
    </defs>
  </svg>
  <svg class="hero-svg-right" viewBox="0 0 180 160" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 155 Q90 125 175 145 L175 160 L10 160 Z" fill="#ffffff"/>
    <rect x="95" y="45" width="28" height="100" rx="2" fill="#ffffff"/>
    <rect x="102" y="28" width="14" height="17" fill="#ffffff"/>
    <polygon points="109,16 101,28 117,28" fill="#ffffff"/>
    <rect x="45" y="95" width="50" height="50" fill="#ffffff"/>
    <path d="M145 145 L145 95 M145 95 C132 88, 125 94, 122 102 M145 95 C135 82, 124 82, 118 88 M145 95 C158 88, 165 94, 168 102 M145 95 C155 82, 166 82, 172 88" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round"/>
  </svg>
  <div class="hero-content">
    <div class="hero-title-row">
      <div class="hero-utensils">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="white">
          <path d="M11 9H9V2H7v7H5V2H3v7c0 2.12 1.66 3.84 3.75 3.97V22h2.5v-9.03C11.34 12.84 13 11.12 13 9V2h-2v7zm5-3v8h2.5v8H21V2c-2.76 0-5 2.24-5 4z"/>
        </svg>
      </div>
      <h1 class="hero-title">Moroccan Food Classifier</h1>
    </div>
    <p class="hero-sub">📸 Upload or take a photo of a Moroccan dish</p>
    <div class="hero-pill">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
        <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
      </svg>
      <span>Created by: <b>Hamza Oulbayyad</b></span>
    </div>
  </div>
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
    mini_medals = ["🥇", "🥈", "🥉"]
    rows_list = []
    for i, idx in enumerate(st.session_state.top_indices):
        rank_num = i + 1
        pct = st.session_state.probs[idx] * 100
        rows_list.append(
            f'<div class="pred-row row-rank-{rank_num}">'
            f'<div class="pred-row-head">'
            f'<div class="medal-badge medal-{rank_num}">{rank_num}</div>'
            f'<div class="v-divider div-{rank_num}"></div>'
            f'<div class="dish-info"><span>{names_ar[idx]}</span><span>{mini_medals[i]}</span></div>'
            f'</div>'
            f'<div class="p-track"><div class="p-fill fill-{rank_num}" style="width:{pct:.1f}%">{pct:.1f}%</div></div>'
            f'</div>'
        )
    rows_html = "".join(rows_list)

    st.markdown(f"""
<div class="pred-panel">
  <div class="pred-header">
    <div class="pred-header-left">
      <div class="pred-icon-box">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
          <path d="M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z"/>
        </svg>
      </div>
      <div>
        <h3 class="pred-title">Top 3 <span>Predictions</span></h3>
        <p class="pred-subtitle">Model Confidence for Each Class</p>
      </div>
    </div>
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" opacity="0.45">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
      <polyline points="17 6 23 6 23 12"></polyline>
    </svg>
  </div>
  {rows_html}
  <div class="pred-footer-bar">
    <span>ℹ️ Predictions are based on our trained model</span>
    <span class="conf-pill">🧠 Model Confidence</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    if not st.session_state.feedback_sent:
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
  <span>|</span>
  <span><b>Model Version:</b> v1.3</span>
  <span>|</span>
  <span><b>Hamza Oulbayyad</b> — Dataset: MFOOD_70 (CC BY-SA 4.0)</span>
  <span>|</span>
  <a href="https://github.com/exora999/moroccan-food-classifier" target="_blank" class="footer-link">
    GitHub Repository ↗
  </a>
</div>
""", unsafe_allow_html=True)
