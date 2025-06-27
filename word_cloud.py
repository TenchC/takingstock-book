"""
make_wordcloud_pdf.py
Tench Cholnoky · macOS Sequoia 15.5 · Python 3.x
-------------------------------------------------
$ pip install wordcloud pillow reportlab pypdf2
"""

import os, tempfile, math
import pandas as pd
import random
from wordcloud import WordCloud          
from PIL import Image                   
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- CONFIG -----------------------------------------------------------
INPUT_CSV   = "word_cloud_inputs/topic_57_counts.csv"
STOP_LIST_PATH = 'model_files/gender_stopwords_dict.csv'
STOP_LIST = {}
INPUT_PATH  = "word_cloud_inputs/"
OUT_PDF     = "outputs/word_cloud/wordcloud.pdf"  # final file
FONT_FILE   = "fonts/CrimsonText-Regular.ttf"
FONT_NAME   = "CrimsonText"
PAGE_SIZE   = [432, 648]  

#bactch Processing
BATCH_PROCESS = True
PROCESS_SELECT = [57, 58]
CSV_LIST = {}

#cutoff for how many rows of the CSV to add to the textcloud
CUTOFF = False
NUM_ROWS = 3000

# Word-cloud cosmetics
FONT_MIN = 1         # adjust to taste
WC_WIDTH, WC_HEIGHT = 3200, 4500    # px; higher = sharper

BACKGROUND = "white"                       # or "transparent"
# -----------------------------------------------------------------------------
def analyze_csv(input_csv, input_path, num_rows):
    dict_key = [int(s) for s in input_csv.split('_') if s.isdigit()]
    dict_key = str(dict_key)[1:len(dict_key)-2]
    df = pd.read_csv(input_path+input_csv)
    df = df.dropna()
    if CUTOFF:
        df = df.head(num_rows)
    return input_csv, df, 

# ---------- 0) BATCH  -----------------------------------------------------
if BATCH_PROCESS:
    print("Batch processing")
    for file in sorted(os.listdir(INPUT_PATH)):
        if file.endswith(".csv"):
            csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
            CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key

#If not batch, add only the csvs selected from the list
elif len(PROCESS_SELECT) > 0:
    print("processing CSVS " + str(PROCESS_SELECT))
    for file in sorted(os.listdir(INPUT_PATH)):
        for i in PROCESS_SELECT:
            if str(i) in file:
                 if file.endswith(".csv"):
                    csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
                    CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key
else:
    print("Add numbers to PROCESS_SELECT or turn on batch processing")


# ---------- 1) LOAD DATA -----------------------------------------------------
#First need to load the stop list
df = pd.read_csv(STOP_LIST_PATH)

for index, row in df.iterrows():
    key = row[0]
    value = row[1] if not pd.isna(row[1]) else ""
    STOP_LIST[key] = value

print("Stop list processed")


for csv in CSV_LIST:
    print("Processing: " + csv)
    df = pd.read_csv(INPUT_PATH+csv).dropna(subset=["description", "count"])

    #!!!! change to weight when csvs given
    random_vals = []
    for row in df.iterrows():
        random_vals.append(random.random())

    # Frequencies for WordCloud
    freqs  = dict(zip(df["description"], df["count"]))
    relations = dict(zip(df['description'], random_vals))

    print(len(freqs))

    count_min, count_max = df["count"].min(), df["count"].max()

# ---------- 2) CUSTOM GRAYSCALE COLOUR FUNCTION -----------------------------
def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    """Return an rgb() string whose gray level comes from the CSV's 'relation'."""
    rel = relations.get(word, 0)           # 0-1
    g   = int( (1 - rel) * 255 )           # 0: black → 255: white
    return f"rgb({g}, {g}, {g})"

# ---------- 3) BUILD WORD CLOUD ---------------------------------------------
wc = (
    WordCloud(width=WC_WIDTH,
              height=WC_HEIGHT,
              background_color=BACKGROUND,
              prefer_horizontal=1.0,
              min_font_size=FONT_MIN,
              color_func=gray_color,
              font_path=FONT_FILE)
    .generate_from_frequencies(freqs)
)

# Save to a temp PNG
tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
wc.to_file(tmp_png.name)

# ---------- 4) RENDER WORD CLOUD ON A PDF PAGE ------------------------------
tmp_wc_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))

c = canvas.Canvas(tmp_wc_pdf.name, pagesize=PAGE_SIZE)
img = ImageReader(tmp_png.name)

# Fit image to page (keep aspect ratio, centred)
img_width, img_height = wc.to_image().size
page_w, page_h = PAGE_SIZE
scale = min((page_w / img_width) * 0.9, (page_h / img_height) * 0.9)  # 90 % inset
draw_w, draw_h = img_width * scale, img_height * scale
x = (page_w - draw_w) / 2
y = (page_h - draw_h) / 2
c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
c.showPage()
c.save()

# ---------- 5) CREATE NEW PDF WITH WORD CLOUD PAGE ONLY ---------------------

pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
c = canvas.Canvas(OUT_PDF, pagesize=PAGE_SIZE)
img = ImageReader(tmp_png.name)

# Fit image to page (keep aspect ratio, centred)
img_width, img_height = wc.to_image().size
page_w, page_h = PAGE_SIZE
scale = min((page_w / img_width) * 0.9, (page_h / img_height) * 0.9)  # 90 % inset
draw_w, draw_h = img_width * scale, img_height * scale
x = (page_w - draw_w) / 2
y = (page_h - draw_h) / 2
c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
c.showPage()
c.save()

print(f"✅ Word-cloud PDF created → {OUT_PDF}")
