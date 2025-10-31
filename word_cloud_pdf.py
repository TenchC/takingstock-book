"""
make_wordcloud_pdf.py
Tench Cholnoky · macOS Sequoia 15.5 · Python 3.10
"""

import os, tempfile, math
import pandas as pd
import random
import numpy as np
import csv
from wordcloud import WordCloud                          
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pick import pick


import gensim
from gensim import corpora
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# ---------- CONFIG -----------------------------------------------------------

GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
TAKINGSTOCK_PATH = os.path.join(os.path.dirname(GLOBAL_PATH), 'takingstock/')

INPUT_PATH  = os.path.join(GLOBAL_PATH, "input_csvs/word_cloud/")
MODEL_PATH =  os.path.join(GLOBAL_PATH, "model/")
STOPWORD_PATH = os.path.join(TAKINGSTOCK_PATH, "model_files/")
OUTPUT_PATH = os.path.join(GLOBAL_PATH, 'outputs/word_cloud/')

# print(f"Paths: Input: {INPUT_PATH}, Model: {MODEL_PATH}, Stopwords: {STOPWORD_PATH}, Output: {OUTPUT_PATH}")

PDF_DATA = {}
OUT_PDF     = os.path.join(OUTPUT_PATH, "wordcloud")  # final file
FONT_FILE   = os.path.join(GLOBAL_PATH, "fonts/CrimsonText-Regular.ttf") 
FONT_NAME   = "CrimsonText"
FOOTER_FONT_FILE = os.path.join(GLOBAL_PATH, "fonts/CrimsonText-SemiBold.ttf")
FOOTER_FONT_NAME = "CrimsonText-SemiBold"
PAGE_SIZE   = [432, 648]  

# Margin and gutter settings
OUTER_MARGIN = 36  # 0.5 inches
INNER_MARGIN = 54  # 0.75 inches (larger for binding)
TOP_MARGIN = 36    # 0.5 inches
BOTTOM_MARGIN = 36 # 0.5 inches

# Footer settings
FOOTER_TEXT = "TOPIC: "  # Base text, topic number will be added dynamically
FOOTER_FONT_SIZE = 10

# Cache for word colors
_word_color_cache = {}

# List to track words that pass/clear
passed_words_list = []

SIDE = "left"

#batch Processing
BATCH_PROCESS = True
PROCESS_SELECT = [11,17]
CSV_LIST = {}

#cutoff for how many rows of the CSV to add to the textcloud
CUTOFF = True
NUM_ROWS = 100
print(f"CUTOFF is {CUTOFF}")
print("NUM_ROWS is", NUM_ROWS)
MANUAL_PICK = False
print(f"MANUAL PICK IS {MANUAL_PICK}")

# Word-cloud cosmetics
FONT_MIN = 1         # adjust to taste
WC_WIDTH, WC_HEIGHT = 3200, 4800    # px; higher = sharper
BACKGROUND = "white"      
WEIGHT_MIN, WEIGHT_MAX = 0, .08
# -----------------------------------------------------------------------------
def analyze_csv(input_csv, input_path, num_rows):
    df = pd.read_csv(input_path+input_csv)
    df = df.dropna()
    if CUTOFF:
        df = df.head(num_rows)
    return input_csv, df, 


def get_document_topic_weights_simple(model, bow_vector, topic_id):
    # this seems deprecated
    if not bow_vector:
        return 0
    
    # Check if it's a list containing 'none' or 'blank'
    if isinstance(bow_vector, list) and len(bow_vector) > 0:
        first_item = bow_vector[0]
        if first_item == "none":
            return "none"
        elif first_item == "blank":
            return "blank"
    
    # bow_vector should be a list of (word_id, count) tuples like [(0, 1), (5, 2)]
    doc_topics = model.get_document_topics(bow_vector, minimum_probability=0)
    # for w in doc_topics:
    #     if w[1] > .1:
    #         print("topic match, weight:", w[1], "topic", w[0])
    
    topic_weight = 0
    for topic, weight in doc_topics:
        if topic == topic_id:
            topic_weight = weight
            break
    
    return topic_weight

def map_values_to_range(input_list):
    numerical_values = []
    result = []
    
    for value in input_list:
        if value == "none":
            result.append("outline")
        elif value == "blank":
            result.append("italic")
        else:
            numerical_values.append(float(value))
            result.append(None)
    
    # Apply log transform to spread out values
    log_values = np.log1p(numerical_values)  # log1p handles zero values safely
    
    min_val = np.min(log_values)
    max_val = np.max(log_values)
    
    if min_val == max_val:
        mapped_values = [1.0] * len(log_values)
    else:
        mapped_values = [(max_val - val) / (max_val - min_val) for val in log_values]
    
    mapped_index = 0
    for i, value in enumerate(input_list):
        if value != "none" and value != "blank":
            result[i] = mapped_values[mapped_index]
            mapped_index += 1
    return result
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

#---------- 0.5) Preprocessing and load model  -----------------------------------------------------
def preprocess(text, MY_STOPWORDS):
    result = []
    text = text.lower()
    for token in gensim.utils.simple_preprocess(text):
        if token not in MY_STOPWORDS and len(token) > 3:
            # print("Not stopped:", token)
            result.append(lemmatize_stemming(token))
    return result


stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data_list = file.read().splitlines()
    return data_list


#CUSTOM GRAYSCALE COLOUR FUNCTION 
# def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    # """Return an rgb() string whose gray level comes from the CSV's 'relation'."""
    # rel = relations.get(word, 0)
    # if rel == "outline":
    #     return f"rgb(1, 0, 0)"
    # elif rel == "italic":
    #     return f"rgb(0, 1, 0)"
    # g   = int( (1 - rel) * 255 )           # 0: black → 255: white
    # return f"rgb({g}, {g}, {g})"

#load the model
lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH+'model')
lda_dict = corpora.Dictionary.load(MODEL_PATH+'model.id2word')

# ---------- 1) LOAD DATA -----------------------------------------------------
GENDER_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_age.csv"))                       
SKIP_TOKEN_LIST = read_csv(os.path.join(STOPWORD_PATH, "skip_tokens.csv"))   
# MY_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(set(GENDER_LIST+ETH_LIST+AGE_LIST+SKIP_TOKEN_LIST))
MY_STOPWORDS = (GENDER_LIST+ETH_LIST+AGE_LIST+SKIP_TOKEN_LIST)


# print("SKIP_TOKEN_LIST", SKIP_TOKEN_LIST) 
# print("MY_STOPWORDS", MY_STOPWORDS)

#set up dictionary
DICT_PATH=os.path.join(MODEL_PATH,"dictionary.dict")
dictionary = corpora.Dictionary.load(MODEL_PATH+'model.id2word')

# Calculate min/max once outside the function
# valid_scores = [v for v in key_score_dict.values() if v is not None]
# MIN_SCORE = min(valid_scores)
# MAX_SCORE = max(valid_scores)
MIN_SCORE = 0
MAX_SCORE = .1

stopword_df_path = os.path.join(OUTPUT_PATH, "topic_word_stopword.csv")

if os.path.exists(stopword_df_path):
    topic_word_stopword_df = pd.read_csv(stopword_df_path)
    # Add 'Replace' column if it doesn't exist
    if 'Replace' not in topic_word_stopword_df.columns:
        topic_word_stopword_df['Replace'] = ''
    print(f"Loaded existing stopword data from {stopword_df_path}")
else:
    topic_word_stopword_df = pd.DataFrame(columns=['word', 'stopword', 'stopped', 'Replace'])
    print(f"Created new stopword data file at {stopword_df_path}")

# def gray_color(word, font_size, position, orientation, random_state=None, **kw):
#     """Return an rgb() string whose gray level comes from the key_score_dict."""
#     score = key_score_dict.get(word, None)
#     print("gray_color", word, score)
#     if score is None:
#         return f"rgb(230, 230, 230)"  # Medium gray for None values
    
#     # Normalize score to 0-1 range
#     normalized_score = (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    
#     # Convert to grayscale (0: black → 255: white)
#     g = int(normalized_score * 255)
#     return f"rgb({g}, {g}, {g})"

def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    """Return an rgb() string whose gray level comes from the key_score_dict."""
    global _word_color_cache 
    if MANUAL_PICK:
        # Check cache first to avoid repeated prompts for the same word
        if word in _word_color_cache:
            return _word_color_cache[word]

        chosen_color = None
        global CSV_NUMBER # Access the global CSV_NUMBER for the current topic
        global topic_word_stopword_df # Access the global DataFrame

        # Check if the word for the current topic has already been processed and stored
        # in the topic_word_stopword_df.
        # This prevents re-prompting for words whose color has already been decided
        # and recorded in a previous run or earlier in the current run.
        existing_entry = topic_word_stopword_df[
            (topic_word_stopword_df['word'] == word)
        ]

        if not existing_entry.empty:
            # If an entry exists, determine the color based on the 'stopped' column
            stopped_value = existing_entry.iloc[0]['stopped']
            replace_value = existing_entry.iloc[0].get('Replace', '') if 'Replace' in existing_entry.columns else ''
            
            # Check if stopped is empty (None, NaN, empty string)
            if pd.isna(stopped_value) or stopped_value == '' or stopped_value is None:
                # If stopped is empty, check the Replace column
                if pd.notna(replace_value) and replace_value != '':
                    # If Replace has a value, set color to black
                    chosen_color = "rgb(0,0,0)"
                    _word_color_cache[word] = chosen_color
                    return chosen_color
                # If Replace is also empty, continue to other logic below
            elif stopped_value == False:
                # If 'stopped' is False, it means the user chose Black (not to stop it)
                chosen_color = "rgb(0,0,0)" 
            elif stopped_value == True:
                # If 'stopped' is True, it means the user chose Gray (to stop it)
                chosen_color = "rgb(180,180,180)"
            
            # Only return if we've set a color (i.e., stopped was not empty/None)
            if chosen_color is not None:
                _word_color_cache[word] = chosen_color
                return chosen_color
        # Iterate through stopwords to find the first match based on priority
        # The first match found will trigger the prompt and return, ensuring only one prompt per word.
        for stopword in MY_STOPWORDS:
            
            title = None
            if stopword == word:
                title = f"Word '{word}' is an exact stopword match with '{stopword}'. Choose color:"
            elif stopword in word:
                # This means 'word' contains 'stopword' (e.g., word="people", stopword="peo")
                title = f"Word '{word}' contains stopword '{stopword}'. Choose color:"
            elif word in stopword:
                # This means 'stopword' contains 'word' (e.g., word="people", stopword="massaii people")
                title = f"Word '{word}' contains stopword '{stopword}'. Choose color:"
            
            if title: # A match was found
                options = [
                    "Black (rgb(0,0,0))",
                    "Gray (rgb(180,180,180))" # Using a medium gray for choice
                ]
                selected_option, index = pick(options, title)
                
                if index == 0: # User chose Black
                    print(f"User chose Black for word '{word}'.")
                    chosen_color = f"rgb(0,0,0)"
                    topic_word_stopword_df.loc[len(topic_word_stopword_df)] = {'word': word, 'stopword': stopword, 'stopped': False}
                else: # User chose Gray
                    print(f"User chose Gray for word '{word}'.")
                    chosen_color = f"rgb(180, 180, 180)"
                    topic_word_stopword_df.loc[len(topic_word_stopword_df)] = {'word': word, 'stopword': stopword, 'stopped': True}
                _word_color_cache[word] = chosen_color # Cache the user's choice for this word
                return chosen_color # Return immediately after the first match and user interaction

        # If the loop completes without finding any stopword match
        chosen_color = f"rgb(0, 0, 0)"  # Black for no match (not in stopwords, good!)
        _word_color_cache[word] = chosen_color # Cache this default choice too
        return chosen_color
    else:
        # Non-MANUAL_PICK mode
        existing_entry = topic_word_stopword_df[
            (topic_word_stopword_df['word'] == word)
        ]

        if not existing_entry.empty:
            stopped_value = existing_entry.iloc[0]['stopped']
            if stopped_value is True:
                return "rgb(200,200,200)"
            elif stopped_value is False:
                return "rgb(0,0,0)"

        print(f'Word {word} cleared')
        global passed_words_list
        passed_words_list.append(word)
        return "rgb(0,0,0)"
            



def get_all_topics_words(lda_model):
    # Get all topics with their word probabilities
    topics = lda_model.print_topics(num_topics=-1, num_words=len(lda_model.id2word))

    # Or get the topic-word matrix
    topic_word_matrix = lda_model.get_topics()
    # print(topic_word_matrix.shape)  # (num_topics, vocab_size)

    # Get word probabilities for a specific topic
    # topic_0_words = lda_model.show_topic(0, topn=len(lda_model.id2word))

    # Get all words for all topics
    all_topics_words = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=len(lda_model.id2word))
        # print(topic_id, topic_words)
        all_topics_words.append(topic_words)
    return all_topics_words


all_topics_words = get_all_topics_words(lda_model_tfidf)
#process each csv
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0]
    print("Processing: " + CSV_NUMBER)
    this_topics_words = dict(all_topics_words[int(CSV_NUMBER)])
    df = pd.read_csv(INPUT_PATH+csv).dropna(subset=["description", "count"])

    keyword_list = []
    remove_empty_list = []
    unproccessed_word_dict = {}
    for row in df.iterrows():
        #preprocess the text
        unproccessed_word = row[1].iloc[1]
        tokens = preprocess(unproccessed_word, MY_STOPWORDS)
        # print("token:", tokens)
        keyword_list.append(tokens)
        
        # Check if tokens is empty
        if not tokens:  # This is True when tokens is an empty list
            unproccessed_word_dict["FOOBAR"] = unproccessed_word
        else:
            # print('Unprocessed word dict ', unproccessed_word_dict.get(tokens[0], 'Key not found'))
            unproccessed_word_dict[tokens[0]] = unproccessed_word
    #remove empty lists
    for each in keyword_list:
        if each == []:
            remove_empty_list.append(each)
    for each in remove_empty_list:
        keyword_list.remove(each)

    if CUTOFF and len(keyword_list) > NUM_ROWS:
        keyword_list = keyword_list[:NUM_ROWS]


    key_score_dict = {}
    for key in keyword_list:

        key_score = this_topics_words.get(key[0], None)
        unproccessed_word = unproccessed_word_dict[key[0]]
        key_score_dict[unproccessed_word] = key_score
    #creating bow vector and tokenizing

    # #if the bow vector is empty (only happens if words are in stopword list), replace with "none" for outlining text later
    # bow_vector = []
    # for each in keyword_list:
    #     bow_vector.append(dictionary.doc2bow(each))
    # for i in range(len(bow_vector)):
    #     if len(bow_vector[i]) == 0:
    #         bow_vector[i] = ["none"]
    #     elif len(bow_vector[i]) > 1:
    #         bow_vector[i] = [bow_vector[i][0]]


    sorted_topics = []

    # #get weights in relation to specific topic currently being used
    # for i in range(len(bow_vector)):         
    #     weights = get_document_topic_weights_simple(lda_model_tfidf, bow_vector[i], int(CSV_NUMBER))  
    #     sorted_topics.append(weights)


    for i in range(len(sorted_topics)):
        if sorted_topics[i]== 0.015625:
            sorted_topics[i] = "blank"



    # Frequencies for WordCloud
    #may need to check
    freqs  = dict(zip(df["description"], df["count"]))
    
    # Preprocess: Replace words based on "Replace" column in topic_word_stopword_df
    # If "stopped" is empty and "Replace" has a value, replace the word
    if 'Replace' in topic_word_stopword_df.columns:
        words_to_replace = {}
        for idx, row in topic_word_stopword_df.iterrows():
            word = row['word']
            stopped_value = row.get('stopped', None)
            replace_value = row.get('Replace', '')
            
            # Check if stopped is empty (None, NaN, empty string) and Replace has a value
            if (pd.isna(stopped_value) or stopped_value == '' or stopped_value is None) and \
               pd.notna(replace_value) and replace_value != '':
                # If word exists in freqs, mark it for replacement
                if word in freqs:
                    words_to_replace[word] = replace_value
        
        # Perform replacements and merge counts if replace value already exists
        for old_word, new_word in words_to_replace.items():
            if old_word in freqs:
                count = freqs.pop(old_word)
                # If new_word already exists, add the counts together
                if new_word in freqs:
                    freqs[new_word] += count
                else:
                    freqs[new_word] = count
                print(f"Replaced '{old_word}' with '{new_word}' (count: {count})")
    
    # print("freqs", freqs)
   #relations = dict(zip(df['description'], map_values_to_range(sorted_topics)))
    relations = dict(zip(df['description'], sorted_topics))

    # ---------- 3) BUILD WORD CLOUD ---------------------------------------------
    wc = (
        WordCloud(width=WC_WIDTH,
                height=WC_HEIGHT,
                background_color=BACKGROUND,
                prefer_horizontal=1.0,
                min_font_size=FONT_MIN,
                color_func=gray_color, ##Interpret as colormap rather than single color, use LLM to process in between steps
                font_path=FONT_FILE)
        .generate_from_frequencies(freqs)
    )

    # Save to a temp PNG
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    wc.to_file(tmp_png.name)

    # Store the word cloud data for later PDF creation
    PDF_DATA[CSV_NUMBER] = {
        'freqs': freqs,
        'relations': relations,
        'tmp_png': tmp_png.name,
        'wc': wc,
        'keyword_list': keyword_list,
        'unprocessed_word_dict': unproccessed_word_dict,
        'this_topics_words': this_topics_words,
        'key_score_dict': key_score_dict
    }

    # Toggle side for next iteration
    if SIDE == "left":
        SIDE = "right"
    else:
        SIDE = "left"

# ---------- 4) CREATE FINAL PDF WITH ALTERNATING PAGES -------------------------
print("Creating final PDF with alternating pages...")

# Register fonts
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
pdfmetrics.registerFont(TTFont(FOOTER_FONT_NAME, FOOTER_FONT_FILE))

# Create the final PDF
final_pdf_path = OUT_PDF + '.pdf'
c = canvas.Canvas(final_pdf_path, pagesize=PAGE_SIZE)

#export stopword data to csv
topic_word_stopword_df.to_csv(os.path.join(OUTPUT_PATH, "topic_word_stopword.csv"), index=False)

#export passed words to csv (remove duplicates first)
passed_words_df = pd.DataFrame({'word': passed_words_list})
passed_words_df = passed_words_df.drop_duplicates()
passed_words_df = passed_words_df.sort_values(by='word')
passed_words_df.to_csv(os.path.join(OUTPUT_PATH, "passed words.csv"), index=False)

# Process each CSV in order, alternating left/right
current_side = "left"
page_number = 1
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0]
    
    if CSV_NUMBER not in PDF_DATA:
        continue
        
    data = PDF_DATA[CSV_NUMBER]
    tmp_png = data['tmp_png']
    wc = data['wc']
    
    # Calculate margins based on current side
    if current_side == "left":
        left_margin = OUTER_MARGIN
        right_margin = INNER_MARGIN
    else:
        left_margin = INNER_MARGIN
        right_margin = OUTER_MARGIN
    
    # Calculate available space for word cloud
    available_width = PAGE_SIZE[0] - left_margin - right_margin
    available_height = PAGE_SIZE[1] - TOP_MARGIN - BOTTOM_MARGIN
    
    # Load image
    img = ImageReader(tmp_png)
    img_width, img_height = wc.to_image().size
    
    # Calculate scale to fit within available space
    scale_x = available_width / img_width
    scale_y = available_height / img_height
    scale = min(scale_x, scale_y) * 0.9  # 90% of max size for some padding
    
    # Calculate final dimensions
    draw_w = img_width * scale
    draw_h = img_height * scale
    
    # Calculate position (centered within available space)
    x = left_margin + (available_width - draw_w) / 2
    y = BOTTOM_MARGIN + (available_height - draw_h) / 2
    
    # Draw the word cloud
    c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
    
    # Add footer
    c.saveState()
    c.setFont(FOOTER_FONT_NAME, FOOTER_FONT_SIZE)
    current_footer_text = FOOTER_TEXT + str(CSV_NUMBER)
    
    # Get first three items from key_score_dict for the second line
    key_score_dict = data.get('key_score_dict', {})
    if key_score_dict:
        # Get the first three keywords and extract the unprocessed words
        first_three_keywords = []
        for word in key_score_dict.keys():
            if word not in MY_STOPWORDS and key_score_dict[word] not in [0, None]:
                first_three_keywords.append(word)
            if len(first_three_keywords) == 3:
                break
        keywords_text = ", ".join(first_three_keywords)
    else:
        keywords_text = "No topic words available"
    
    # Choose footer alignment based on even/odd page (left/right side)
    if current_side == "left":
        footer_x = left_margin  # Left-aligned on left pages
    else:
        footer_x = PAGE_SIZE[0] - right_margin - c.stringWidth(current_footer_text, FOOTER_FONT_NAME, FOOTER_FONT_SIZE)  # Right-aligned on right pages
    
    # Draw the topic line
    c.drawString(footer_x, BOTTOM_MARGIN - 10, current_footer_text)
    
    # Draw the keywords line (same font and size)
    if current_side == "left":
        keywords_x = left_margin  # Left-aligned on left pages
    else:
        keywords_x = PAGE_SIZE[0] - right_margin - c.stringWidth(keywords_text, FOOTER_FONT_NAME, FOOTER_FONT_SIZE)  # Right-aligned on right pages
    
    c.drawString(keywords_x, BOTTOM_MARGIN - 25, keywords_text)
    c.restoreState()
    
    c.showPage()
    
    # Toggle side for next page
    current_side = "right" if current_side == "left" else "left"
    page_number += 1

c.save()
print(f"✅ Final word-cloud PDF created → {final_pdf_path}")

# Clean up temporary files
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0]
    if CSV_NUMBER in PDF_DATA:
        try:
            os.unlink(PDF_DATA[CSV_NUMBER]['tmp_png'])
        except:
            pass

