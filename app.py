"""Moroccan Food Classifier - Gradio demo.

import json
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow import keras
from google.colab import drive

drive.mount('/content/drive')
P = '/content/drive/MyDrive/food_project'

model = keras.models.load_model(f'{P}/food_model_v1.keras')
with open(f'{P}/food_classes.json') as f:
    class_names = json.load(f)
print("تم تحميل النموذج:", len(class_names), "فئة")

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
missing = [c for c in class_names if c not in AR]
print("Categories without translation (will appear under their original name):", missing)

def predict(img):
    if img is None:
        return None
    img = img.convert('RGB').resize((224, 224))
    x = np.expand_dims(np.array(img, dtype='float32'), axis=0)
    probs = model.predict(x, verbose=0)[0]
    return {names_ar[i]: float(probs[i]) for i in range(len(names_ar))}

HERO = """
<div style="background:linear-gradient(135deg,#9a3412,#ea580c 55%,#fdba74);
            border-radius:20px;padding:30px 20px;text-align:center;color:white;
            margin-bottom:14px;direction:rtl;">
  <h1 style="margin:0 0 6px 0;font-size:1.9rem;color:white;">🍽️ ما هذا الطبق؟</h1>
  <p style="margin:0 0 14px 0;font-size:1.05rem;color:white;opacity:.95;">
    📸 Upload or take a photo of a Moroccan dish — أرفع صورة أو التقطها</p>
  <span style="display:inline-block;background:rgba(255,255,255,.22);
               border:1px solid rgba(255,255,255,.55);border-radius:999px;
               padding:6px 18px;font-size:1rem;color:white;">
     ✨ Created by: <b>Hamza Oulbayyad</b></span>
</div>
"""

FOOT = """
<div style="text-align:center;font-size:.9rem;opacity:.9;margin-top:12px;direction:ltr;">
<div>⚠️ Educational project: The model only recognizes 70 specific dishes. If you show it something else
(or something that is not food), it will select the closest match from those 70, which may result in an error. 
For best results, use a clear, top-down photo of a single dish.</div>
<div style="margin-top:8px;">Model design and training: <b>Hamza Oulbayyad</b>
— Dataset: MFOOD_70 (licensed under CC BY-SA 4.0)</div>
</div>
"""

with gr.Blocks(title="ما هذا الطبق؟ — حمزة اولبياض") as demo:
    gr.HTML(HERO)
    inp = gr.Image(type="pil", sources=["upload", "webcam"],
                   label="صورة الطبق (ارفع صورة أو التقطها)", height=360)
    btn = gr.Button("🔍 ما اسم هذا الطبق؟", variant="primary")
    out = gr.Label(num_top_classes=3, label="Top 3 predictions")
    gr.HTML(FOOT)
    btn.click(predict, inputs=inp, outputs=out)

demo.launch(share=True)
