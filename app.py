import sys
import pandas as pd
import os
import glob

#using reportlab to create and style the pdf
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, PageTemplate, Frame, BaseDocTemplate, PageBreak, Flowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

#need pdfmetrics to register the font
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

#global vars
input_path = "input_csvs/"

#CSV Vars
inputCSV = 'input_csvs/topic_13_counts.csv'
number_of_topics = 37*2 #37 lines per page, 2 pages per spread, one spread per csv
csv_list = {}
max_count = 1
min_count = 1

# PDF Vars
outputPDF = 'output.pdf'
font_size = 12
font_file = 'fonts/CrimsonText-Regular.ttf'
footer_font_file = 'fonts/CrimsonText-SemiBold.ttf'
page_size = [432, 648] #in points, 72 points = 1 inch
#margin settings (in points) (72 points = 1 inch)
inner_margin = 90  # 1.25 inch
outter_margin = 54  # 0.75 inch
top_margin = 54
bottom_margin = 54

# for each csv file in the folder 
# take the top number_of_topics and put them in a new dataframe
# then sort them by description alphabetically
# then save that section of data into a dictionary where the csv number is the key and the value is the table
for file in sorted(os.listdir(input_path)):
    if file.endswith(".csv"):
        dict_key = [int(s) for s in file.split('_') if s.isdigit()]
        dict_key = str(dict_key)[1:len(dict_key)-2]
        df = pd.read_csv(input_path+file)
        df = df.head(number_of_topics)
        df = df.sort_values(by="description", ascending = True)
        csv_list.update({str(dict_key): df})

print("csv_list keys")
print(csv_list.keys())
print(' ')

# PDF Work Starts
class BlankPage(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width = 0
        self.height = 528

    def draw(self):
        pass

# Register the custom font
pdfmetrics.registerFont(TTFont('CrimsonText', font_file))
pdfmetrics.registerFont(TTFont('CrimsonText-SemiBold', footer_font_file))
INNER_MARGIN = inner_margin  
OUTER_MARGIN = outter_margin 
TOP_MARGIN = top_margin
BOTTOM_MARGIN = bottom_margin

# Create style for the table
styles = getSampleStyleSheet()
table_style = ParagraphStyle(
    'TableStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText' 
)

# Create style for the footer text
footer_style = ParagraphStyle(
    'FooterStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText-SemiBold',
    alignment=1  # Center alignment
)

# Convert the data to paragraphs for proper wrapping
data_for_pdf = []

# Process all topics in csv_list
for topic_key, topic_data in csv_list.items():
    # Add data rows with only descriptions for this topic
    for _, row in topic_data.iterrows():
        data_row = [Paragraph(str(row[1]), table_style)]  # Only the description column
        data_for_pdf.append(data_row)
    
    # Add a page break after each topic (except the last one)
    if topic_key != list(csv_list.keys())[-1]:
        data_for_pdf.append(PageBreak())

# Create the footer text - will be set dynamically per page
footer_text = "TOPIC: "  # Base text, topic number will be added dynamically

#create a custom doc template
class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, **kw)
        
        # Initialize page counter to 1 (first page)
        self.page = 1
        # Track which topic each page belongs to (2 pages per topic)
        self.page_to_topic = {}
        self._build_page_to_topic_mapping()

        # Create frame for odd pages (outer margin on left)
        odd_frame = Frame(
            OUTER_MARGIN,
            self.bottomMargin +1,
            self.width - OUTER_MARGIN - INNER_MARGIN,  # Adjust width to account for margins
            self.height,
            id='odd'
        )

        # Create frame for even pages (inner margin on left)
        even_frame = Frame(
            INNER_MARGIN,
            self.bottomMargin+1,
            self.width - OUTER_MARGIN - INNER_MARGIN,  # Adjust width to account for margins
            self.height,
            id='even'
        )

        # Page templates
        odd_template = PageTemplate(
            id='odd',
            frames=[odd_frame],
            onPage=self.add_footer
        )

        even_template = PageTemplate(
            id='even',
            frames=[even_frame],
            onPage=self.add_footer
        )

        # Add both templates and set the first page template
        self.addPageTemplates([odd_template, even_template])
        self.pageTemplate = odd_template  # Set initial template

    def _build_page_to_topic_mapping(self):
        """Build a mapping of page numbers to topic keys"""
        topic_keys = list(csv_list.keys())
        for i, topic_key in enumerate(topic_keys):
            # Each topic gets 2 pages (spread)
            start_page = i * 2 + 1  # +1 because first page is blank
            self.page_to_topic[start_page] = topic_key
            self.page_to_topic[start_page + 1] = topic_key

    def handle_pageEnd(self):
        """Switch templates between odd and even pages"""
        if self.page % 2 == 0:  # Even page
            self.pageTemplate = self.pageTemplates[1]  # Use even template
        else:  # Odd page
            self.pageTemplate = self.pageTemplates[0]  # Use odd template
        super().handle_pageEnd()

    def add_footer(self, canvas, doc):
        #skip first page
        if self.page != 1:
            canvas.saveState()
            canvas.setFont('CrimsonText-SemiBold', font_size)

            # Get the topic for this page
            topic_key = self.page_to_topic.get(self.page, "Unknown")
            current_footer_text = footer_text + str(topic_key)

            # Choose footer alignment based on even/odd page
            is_even = doc.page % 2 == 0
            x = OUTER_MARGIN if is_even else (page_size[0] - (OUTER_MARGIN*2))

            canvas.drawString(x, BOTTOM_MARGIN - 20, current_footer_text)
            canvas.restoreState()

#creating pdf
pdf = MyDocTemplate(
    outputPDF,
    pagesize=page_size,
    topMargin=TOP_MARGIN,
    bottomMargin=BOTTOM_MARGIN,
    leftMargin=0,  # Set left margin to 0 since we handle margins in frames
    rightMargin=0  # Set right margin to 0 since we handle margins in frames
)

# Set column width for single column - will be adjusted per page
column_width = [page_size[0] - INNER_MARGIN - OUTER_MARGIN]  # Full width minus margins

table = Table(data_for_pdf, colWidths=column_width)

#simple styling for the table
table.setStyle(TableStyle([
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ('TOPPADDING', (0,0), (-1,-1), 0),
    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ('FONTNAME', (0,0), (-1,-1), 'CrimsonText'), 
]))

# Build the PDF with just the table
pdf.build([BlankPage(), table])

#export the pdf
#pdf.save()
print(f"PDF saved to {outputPDF}")
