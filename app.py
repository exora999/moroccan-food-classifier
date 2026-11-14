"""
Moroccan Food Classifier — Streamlit version
Saves corrections and failed images directly to GitHub Repository
Author: Hamza Oulbayyad
"""
import json, urllib.request, urllib.parse, base64, io as _io
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Moroccan Food Classifier — Hamza Oulbayyad",
    page_icon="🍽️", layout="centered")

# ── Stunning Dark Glassmorphism CSS ──────────────────────────────────────────
st.markdown("""
<style>
  /* خلفية مظلمة فخمة وناعمة تريح العين */
  .stApp {
      background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
      color: #f3f4f6;
  }
  
  /* تحسين مظهر الأشرطة النسبية بتصميم مظلم متناسق */
  .bar-wrap{margin:8px 0 16px 0;direction:rtl}
  .bar-label{font-size:1.1rem;font-weight:700;margin-bottom:6px;
             color:#fb923c;text-shadow:0 1px 2px rgba(0,0,0,.5)}
  .bar-track{background:rgba(255, 255, 255, 0.08);border-radius:999px;height:24px;
             width:100%;overflow:hidden;box-shadow: inset 0 2px 4px rgba(0,0,0,.4);
             border: 1px solid rgba(255, 255, 255, 0.05);}
  .bar-fill{height:24px;border-radius:999px;display:flex;align-items:center;
            padding-right:12px;justify-content:flex-end;color:white;
            font-size:.9rem;font-weight:700;min-width:48px;
            transition: width 1s ease-in-out;}
  .rank-1 .bar-fill{background:linear-gradient(90deg,#c2410c,#ea580c,#fb923c)}
  .rank-2 .bar-fill{background:linear-gradient(90deg,#9a3412,#d97706,#fcd34d);opacity:.95}
  .rank-3 .bar-fill{background:linear-gradient(90deg,#374151,#4b5563,#9ca3af);opacity:.9}

  /* أزرار تفاعلية عصرية ذات مظهر زجاجي متحرك */
  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(255, 255, 255, 0.05) !important;
      border: 1px solid rgba(255, 255, 255, 0.15) !important;
      color: #ffffff !important;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #ea580c, #c2410c) !important;
      border-color: #fb923c !important;
      transform: translateY(-2px) scale(1.02) !important;
      box-shadow: 0 10px 20px -3px rgba(234, 88, 12, 0.4) !important;
  }
  
  /* تنسيق الصناديق والتبويبات بلمسة زجاجية */
  .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 10px 22px;
      color: #e5e7eb;
      font-weight: 600;
  }
  .stTabs [data-baseweb="tab"]:hover {
      background-color: rgba(234, 88, 12, 0.2);
  }
</style>
""", unsafe_allow_html=True)

# ── GitHub Integration (Issue + Image Upload) ─────────────────────────────────
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
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None

def create_github_issue(wrong: str, correct: str, img_bytes: bytes):
    token = st.secrets["GITHUB_TOKEN"]
    repo  = st.secrets["GITHUB_REPO"]
    issues_url = f"https://api.github.com/repos/{repo}/issues"

    import time
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

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = keras.models.load_model("food_model_v1.keras")
    with open("food_classes.json", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

# ── Arabic names ──────────────────────────────────────────────────────────────
AR = {
    "traditional macaroon":"ماكرون تقليدي","tomatoes and onion salad":"سلطة الطماطم والبصل",
    "TAKTOUKA":"تكتوكة","zitoun":"زيتون","sweet bread":"خبز حلو","tkalya":"تقلية",
    "sellou":"سلّو","amlou":"أملو","sfenje":"سفنج","snowballs":"كرات الثلج",
    "traditional bread":"خبز تقليدي","white beans with tomatoes":"لوبيا بيضاء بالطماطم",
    "tagine with quince":"طاجين بالسفرجل","tagine with vegetables":"طاجين بالخضار",
    "tagine with artichokes and peas":"طاجين بالخرشوف والبازلاء","seffa with rice":"سفة بالأرز",
    "crackers with almonds (ghriba)":"غريبة باللوز","nougat":"نوغا","fekkas":"فقاص",
    "better beldi":"زبدة بلدية","chebakia":"شباكية","gazelle horn":"كعب الغزال",
    "BISSARA(feves puree)":"بيصارة","maakouda":"معقودة","FEVES with sauce":"فول بالصلصة",
    "chicken with potatoes and olives":"دجاج بالبطاطس والزيتون","bahla":"بهلة",
    "Feet of beef":"كرعين البقر","lentils":"عدس","briouate  with almonds":"بريوات باللوز",
    "briouate with almonds":"بريوات باللوز","meatball with tomato sauce":"كفتة بصلصة الطماطم",
    "salmon":"سلمون","jam":"مربى","basbousa":"بسبوسة","apple":"تفاح","banana":"موز",
    "batbout":"بطبوط","beghrir":"بغرير","caesar salad":"سلطة سيزر",
    "chicken basstila":"بسطيلة بالدجاج","chicken nuggets":"ناجتس الدجاج",
    "chocolate cake":"كعكة الشوكولاتة","couscous":"كسكس","croissant":"كرواسون",
    "dates":"تمر","fish and vegetables":"سمك بالخضار","fish basstila":"بسطيلة بالسمك",
    "french fries":"بطاطس مقلية","fried calamari":"حبّار مقلي","hamburger":"همبرغر",
    "harcha":"حرشة","harira":"حريرة","lasagne":"لازانيا","meat brochettes":"أسياخ لحم مشوي",
    "msamen":"مسمن","omelette":"عجّة (أومليت)","orange":"برتقال","paella":"باييا",
    "pear":"إجاص","pizza":"بيتزا","rfissa":"رفيسة","roasted chicken":"دجاج مشوي",
    "seffa":"سفة","spagetti":"سباغيتي","tagine with beef":"طاجين باللحم","zaalouk":"زعلوك",
    "mhancha":"محنشة","rghayf":"رغايف","kaak":"كعك","liver with sauce":"كبدة بالصلصة",
}
names_ar = [AR.get(c, c) for c in class_names]

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("top_indices","probs","feedback_sent","img_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#9a3412,#ea580c 55%,#fb923c);
            border-radius:24px;padding:35px 20px;text-align:center;color:white;
            margin-bottom:25px; box-shadow: 0 15px 30px -5px rgba(234, 88, 12, 0.3);">
  <h1 style="margin:0 0 8px 0;font-size:2.1rem;color:white;text-shadow:0 2px 4px rgba(0,0,0,0.3);">🍽️ Moroccan Food Classifier</h1>
  <p style="margin:0 0 16px 0;font-size:1.1rem;color:white;opacity:.95;">
    📸 Upload or take a photo of a Moroccan dish</p>
  <span style="display:inline-block;background:rgba(255,255,255,0.2);
                border:1px solid rgba(255,255,255,0.5);border-radius:999px;
                padding:8px 22px;font-size:1.05rem;color:white;backdrop-filter: blur(5px);">
      ✨ Created by: <b>Hamza Oulbayyad</b></span>
</div>
""", unsafe_allow_html=True)

# ── Image input ───────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Live Camera"])
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed")
with tab_camera:
    camera_photo = st.camera_input("Take a photo with the camera")

img_source = uploaded_file or camera_photo

# ── Predict ───────────────────────────────────────────────────────────────────
if img_source is not None:
    image = Image.open(img_source)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 What is this dish?", type="primary", use_container_width=True):
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

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.top_indices is not None:
    medals = ["🥇","🥈","🥉"]
    ranks  = ["rank-1","rank-2","rank-3"]
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

    # ── Feedback ──────────────────────────────────────────────────────────────
    if not st.session_state.feedback_sent:
        st.divider()
        st.markdown("**Is the top prediction correct? / هل التوقع الأول صحيح؟**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Correct / نعم، صحيح", use_container_width=True):
                st.session_state.feedback_sent = "correct"
                st.rerun()
        with col2:
            if st.button("❌ No, Wrong / لا، خطأ", use_container_width=True):
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

        if fix and st.button("📨 Submit Correction / إرسال التصحيح", use_container_width=True):
            wrong_name = names_ar[st.session_state.top_indices[0]]
            with st.spinner("Saving image and creating GitHub issue..."):
                try:
                    issue_url = create_github_issue(
                        wrong_name, fix,
                        st.session_state.img_bytes)
                    st.session_state.feedback_sent = "done"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    elif st.session_state.feedback_sent == "done":
        st.success("Thank you! Image saved and correction documented on GitHub successfully 🎉")
        st.info("You can use these images later to retrain your model.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:.85rem;color:#9ca3af;margin-top:35px;direction:ltr;border-top:1px solid rgba(255,255,255,0.1);padding-top:15px;">
  ⚠️ This model recognizes 70 Moroccan dishes only. | <b>Model Version:</b> v1.3<br>
  <b>Hamza Oulbayyad</b> — Dataset: MFOOD_70 (CC BY-SA 4.0) | 
  <a href="https://github.com/exora999/moroccan-food-classifier" target="_blank" style="color: #fb923c; text-decoration: none;">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
