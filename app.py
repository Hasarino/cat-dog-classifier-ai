import gradio as gr
import tensorflow as tf

model = tf.keras.models.load_model('yapay_zeka_beyni_KUSURSUZ.keras')

def yapay_zeka_analiz(yuklenen_resim):
    if yuklenen_resim is None:
        return None
        
    resim = tf.image.resize(yuklenen_resim, [150, 150])
    resim_dizisi = tf.expand_dims(resim, 0)
    
    tahmin = model.predict(resim_dizisi)[0][0]
    
    eminlik_kopek = float(tahmin)
    eminlik_kedi = float(1.0 - tahmin)
    
    return {"Köpek 🐶": eminlik_kopek, "Kedi 🐱": eminlik_kedi}

tasarim = gr.Interface(
    fn=yapay_zeka_analiz,
    inputs=gr.Image(sources=["upload", "webcam"], label="Fotoğraf Yükle veya Kamerayı Aç"),
    outputs=gr.Label(label="Yapay Zekanın Kararı"),
    title="🤖 Kedi & Köpek Yapay Zeka Radarı (Bulut Sürüm)",
    description="Bu yapay zeka 7/24 bulut sunucularında çalışmaktadır. İster dosyadan fotoğraf yükle, ister bilgisayarının kamerasını açıp göster!"
)

tasarim.launch()