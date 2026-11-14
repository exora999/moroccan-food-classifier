import json
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="ما هذا الطبق؟", page_icon="🍽️", layout="centered")

@st.cache_resource
def load_assets():
    model = keras.models.load_model("food_model_v1.keras")
    with open("food_classes.json", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

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

HERO = """
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
"""

st.markdown(HERO, unsafe_allow_html=True)

uploaded_file = st.file_uploader("صورة الطبق (اختر صورة من جهازك)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("🔍 ما اسم هذا الطبق؟", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل الطبق..."):
            img_resized = image.convert("RGB").resize((224, 224))
            x = np.expand_dims(np.array(img_resized, dtype="float32"), axis=0)
            probs = model.predict(x, verbose=0)[0]
            
            top_indices = np.argsort(probs)[::-1][:3]
            st.subheader("أقرب 3 توقعات (Top 3 Predictions):")
            for idx in top_indices:
                confidence = float(probs[idx])
                st.write(f"**{names_ar[idx]}** — `{confidence * 100:.1f}%`")
                st.progress(confidence)

FOOT = """
<div style="text-align:center;font-size:.85rem;opacity:.85;margin-top:25px;direction:ltr;">
<div>⚠️ Educational project: The model only recognizes 70 specific dishes. If you show it something else
(or something that is not food), it will select the closest match from those 70, which may result in an error. 
For best results, use a clear, top-down photo of a single dish.</div>
<div style="margin-top:8px;">Model design and training: <b>Hamza Oulbayyad</b>
— Dataset: MFOOD_70 (licensed under CC BY-SA 4.0)</div>
</div>
"""

st.markdown(FOOT, unsafe_allow_html=True)
