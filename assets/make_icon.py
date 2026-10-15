# C:\Users\adity\Downloads\drone-detection-starter\assets\make_icon.py
from PIL import Image
src = "romeo_bg.jpg"
ico = "romeo_eye.ico"
img = Image.open(src).convert("RGBA")
# square crop center and resize to 256x256 (Windows likes 256px in ICO)
s = min(img.size)
x0 = (img.width - s)//2
y0 = (img.height - s)//2
img = img.crop((x0, y0, x0+s, y0+s)).resize((256, 256), Image.LANCZOS)
img.save(ico, format="ICO", sizes=[(256,256)])
print("Wrote", ico)
