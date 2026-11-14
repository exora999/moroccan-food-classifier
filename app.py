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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ما هذا الطبق؟ — Hamza Oulbayyad",
    page_icon="🍽️",
    layout="centered",
)

# ── Load model (cached so it loads only once) ─────────────────────────────────
@st.cache_resource
def load_assets():
    model = keras.models.load_model("food_model_v1.keras")
    with open("food_classes.json", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

# ── Arabic display names ──────────────────────────────────────────────────────
# Note: some names are best guesses — please review before publishing
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
    "snowballs": "كرات الثلج (حلوى جوز الهند)",
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
    "omelette": "عجّة بالبيض (أومليت)",
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

# ── Hero banner ───────────────────────────────────────────────────────────────
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

# ── Image input: upload OR camera ─────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 رفع صورة", "📷 تصوير مباشر"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "اختر صورة من جهازك",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with tab_camera:
    camera_photo = st.camera_input("التقط صورة بالكاميرا")

img_source = uploaded_file or camera_photo

# ── Prediction ────────────────────────────────────────────────────────────────
if img_source is not None:
    image = Image.open(img_source)
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    if st.button("🔍 ما اسم هذا الطبق؟", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل الطبق..."):
            img_resized = image.convert("RGB").resize((224, 224))
            x = np.expand_dims(np.array(img_resized, dtype="float32"), axis=0)
            probs = model.predict(x, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top_name = names_ar[top_indices[0]]

        st.subheader("أقرب 3 توقعات (Top 3 Predictions):")
        for rank, idx in enumerate(top_indices):
            confidence = float(probs[idx])
            emoji = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
            st.write(f"{emoji} **{names_ar[idx]}** — `{confidence * 100:.1f}%`")
            st.progress(confidence)

        # ── Feedback section ──────────────────────────────────────────────────
        st.divider()
        st.markdown("**هل التوقع صحيح؟**")
        correct = st.radio(
            "feedback",
            ["✅ نعم، صحيح", "❌ لا، خطأ"],
            horizontal=True,
            index=None,
            label_visibility="collapsed",
        )

        if correct == "✅ نعم، صحيح":
            st.success("ممتاز! شكراً على تأكيدك 🎉")

        elif correct == "❌ لا، خطأ":
            fix = st.selectbox(
                "ما اسم الطبق الصحيح؟",
                options=names_ar,
                index=None,
                placeholder="اختر من القائمة...",
            )
            if fix and st.button("📨 إرسال التصحيح", use_container_width=True):
                log_file = "corrections.csv"
                write_header = not os.path.exists(log_file)
                with open(log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["التاريخ", "توقع النموذج", "الصحيح"])
                    writer.writerow([
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        top_name,
                        fix,
                    ])
                st.success(f"شكراً! تعلّمتُ أن الطبق هو **{fix}** 🎉")
                st.info("سيتم مراجعة هذا التصحيح لاحقاً لتحسين النموذج.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-size:.85rem;opacity:.85;margin-top:25px;direction:ltr;">
  <div>⚠️ Educational project: The model only recognizes 70 specific dishes.
  If you show it something else it will pick the closest match, which may be wrong.
  For best results, use a clear top-down photo of a single dish.</div>
  <div style="margin-top:8px;">
    Model design and training: <b>Hamza Oulbayyad</b> —
    Dataset: MFOOD_70 (CC BY-SA 4.0)
  </div>
</div>
""", unsafe_allow_html=True)
