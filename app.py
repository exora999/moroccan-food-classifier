"""
Moroccan Food Classifier — Streamlit version
Deploy on Streamlit Community Cloud (share.streamlit.io)
Needs: food_model_v1.keras  food_classes.json  in the same repo
Author: Hamza Oulbayyad
"""
import csv
import datetime
import json
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ما هذا الطبق؟ — Hamza Oulbayyad",
    page_icon="🍽️",
    layout="centered",
)

st.markdown("""
<style>
  .bar-wrap { margin: 6px 0 14px 0; direction: rtl; }
  .bar-label { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; color: #ea580c; text-shadow: 0 1px 2px rgba(0,0,0,.4); }
  .bar-track { background: #f0f0f0; border-radius: 999px; height: 22px; width: 100%;
               overflow: hidden; }
  .bar-fill  { height: 22px; border-radius: 999px;
               background: linear-gradient(90deg, #9a3412, #ea580c, #fdba74);
               display: flex; align-items: center; padding-right: 10px;
               justify-content: flex-end; color: white; font-size: .85rem;
               font-weight: 700; min-width: 44px; transition: width .4s ease; }
  .rank-1 .bar-fill { background: linear-gradient(90deg,#9a3412,#ea580c,#fdba74); }
  .rank-2 .bar-fill { background: linear-gradient(90deg,#78350f,#b45309,#fcd34d); opacity:.85; }
  .rank-3 .bar-fill { background: linear-gradient(90deg,#57534e,#78716c,#d6d3d1); opacity:.75; }
  .stButton > button { border-radius: 14px !important; }
  .feedback-box { background:#fafafa; border:1px solid #e5e7eb;
                  border-radius:14px; padding:18px 16px; margin-top:14px; direction:rtl; }
</style>
""", unsafe_allow_html=True)

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
    "traditional macaroon": "ماكرون تقليدي",
    "tomatoes and onion salad": "سلطة الطماطم والبصل",
    "TAKTOUKA": "تكتوكة",
    "zitoun": "زيتون",
    "sweet bread": "خبز حلو",
    "tkalya": "تقلية",
    "sellou": "سلّو",
    "amlou": "أملو",
    "sfenje": "سفنج",
    "snowballs": "كرات الثلج",
    "traditional bread": "خبز تقليدي",
    "white beans with tomatoes": "لوبيا بيضاء بالطماطم",
    "tagine with quince": "طاجين بالسفرجل",
    "tagine with vegetables": "طاجين بالخضار",
    "tagine with artichokes and peas": "طاجين بالخرشوف والبازلاء",
    "seffa with rice": "سفة بالأرز",
    "crackers with almonds (ghriba)": "غريبة باللوز",
    "nougat": "نوغا",
    "fekkas": "فقاص",
    "better beldi": "زبدة بلدية",
    "chebakia": "شباكية",
    "gazelle horn": "كعب الغزال",
    "BISSARA(feves puree)": "بيصارة",
    "maakouda": "معقودة",
    "FEVES with sauce": "فول بالصلصة",
    "chicken with potatoes and olives": "دجاج بالبطاطس والزيتون",
    "bahla": "بهلة",
    "Feet of beef": "كرعين البقر",
    "lentils": "عدس",
    "briouate  with almonds": "بريوات باللوز",
    "briouate with almonds": "بريوات باللوز",
    "meatball with tomato sauce": "كفتة بصلصة الطماطم",
    "salmon": "سلمون",
    "jam": "مربى",
    "basbousa": "بسبوسة",
    "apple": "تفاح",
    "banana": "موز",
    "batbout": "بطبوط",
    "beghrir": "بغرير",
    "caesar salad": "سلطة سيزر",
    "chicken basstila": "بسطيلة بالدجاج",
    "chicken nuggets": "ناجتس الدجاج",
    "chocolate cake": "كعكة الشوكولاتة",
    "couscous": "كسكس",
    "croissant": "كرواسون",
    "dates": "تمر",
    "fish and vegetables": "سمك بالخضار",
    "fish basstila": "بسطيلة بالسمك",
    "french fries": "بطاطس مقلية",
    "fried calamari": "حبّار مقلي",
    "hamburger": "همبرغر",
    "harcha": "حرشة",
    "harira": "حريرة",
    "lasagne": "لازانيا",
    "meat brochettes": "أسياخ لحم مشوي",
    "msamen": "مسمن",
    "omelette": "عجّة (أومليت)",
    "orange": "برتقال",
    "paella": "باييا",
    "pear": "إجاص",
    "pizza": "بيتزا",
    "rfissa": "رفيسة",
    "roasted chicken": "دجاج مشوي",
    "seffa": "سفة",
    "spagetti": "سباغيتي",
    "tagine with beef": "طاجين باللحم",
    "zaalouk": "زعلوك",
    "mhancha": "محنشة",
    "rghayf": "رغايف",
    "kaak": "كعك",
    "liver with sauce": "كبدة بالصلصة",
}
names_ar = [AR.get(c, c) for c in class_names]

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("top_indices", "probs", "feedback_sent"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#9a3412,#ea580c 55%,#fdba74);
            border-radius:20px;padding:30px 20px;text-align:center;color:white;
            margin-bottom:20px;direction:rtl;">
  <h1 style="margin:0 0 6px 0;font-size:1.9rem;color:white;">🍽️ ما هذا الطبق؟</h1>
  <p style="margin:0 0 14px 0;font-size:1.05rem;color:white;opacity:.95;">
    📸 Upload or take a photo of a Moroccan dish — ارفع صورة أو التقطها</p>
  <span style="display:inline-block;background:rgba(255,255,255,.22);
               border:1px solid rgba(255,255,255,.55);border-radius:999px;
               padding:6px 18px;font-size:1rem;color:white;">
     ✨ Created by: <b>Hamza Oulbayyad</b></span>
</div>
""", unsafe_allow_html=True)

# ── Image input ───────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 رفع صورة", "📷 تصوير مباشر"])
with tab_upload:
    uploaded_file = st.file_uploader(
        "اختر صورة", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed")
with tab_camera:
    camera_photo = st.camera_input("التقط صورة بالكاميرا")

img_source = uploaded_file or camera_photo

# ── Predict button ────────────────────────────────────────────────────────────
if img_source is not None:
    image = Image.open(img_source)
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    if st.button("🔍 ما اسم هذا الطبق؟", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل الطبق..."):
            img_r = image.convert("RGB").resize((224, 224))
            x = np.expand_dims(np.array(img_r, dtype="float32"), axis=0)
            probs = model.predict(x, verbose=0)[0]
        st.session_state.probs = probs
        st.session_state.top_indices = np.argsort(probs)[::-1][:3].tolist()
        st.session_state.feedback_sent = False

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.top_indices is not None:
    medals = ["🥇", "🥈", "🥉"]
    ranks  = ["rank-1", "rank-2", "rank-3"]
    st.markdown("### أقرب 3 توقعات (Top 3 Predictions)")

    for i, idx in enumerate(st.session_state.top_indices):
        pct = st.session_state.probs[idx] * 100
        st.markdown(f"""
        <div class="bar-wrap {ranks[i]}">
          <div class="bar-label">{medals[i]} {names_ar[idx]}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%">{pct:.1f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.feedback_sent:
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.markdown("**هل التوقع الأول صحيح؟**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ نعم، صحيح", use_container_width=True):
                st.session_state.feedback_sent = "correct"
                st.rerun()
        with col2:
            if st.button("❌ لا، خطأ", use_container_width=True):
                st.session_state.feedback_sent = "wrong"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.feedback_sent == "correct":
        st.success("ممتاز! شكراً على تأكيدك 🎉")

    elif st.session_state.feedback_sent == "wrong":
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.markdown("**ما اسم الطبق الصحيح؟**")
        fix = st.selectbox(
            "اختر من القائمة",
            options=[""] + names_ar,
            index=0,
            label_visibility="collapsed",
        )
        if fix and st.button("📨 إرسال التصحيح", use_container_width=True):
            top_name = names_ar[st.session_state.top_indices[0]]
            log_file = "corrections.csv"
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["التاريخ", "توقع النموذج", "الصحيح"])
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    top_name, fix,
                ])
            st.session_state.feedback_sent = "done"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.feedback_sent == "done":
        fix_name = ""
        st.success(f"شكراً! تعلّمتُ منك 🎉 — سيتم مراجعة هذا التصحيح لاحقاً.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:.85rem;opacity:.8;margin-top:30px;direction:ltr;">
  ⚠️ Educational project: 70 dishes only — results may be wrong outside this list.<br>
  <b>Hamza Oulbayyad</b> — Dataset: MFOOD_70 (CC BY-SA 4.0)
</div>
""", unsafe_allow_html=True)
