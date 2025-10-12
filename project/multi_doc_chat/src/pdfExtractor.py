import fitz               # PyMuPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------- SETTINGS ----------
PDF_PATH     = "NCERT-Class-6-History.pdf"
OUTPUT_PDF   = "chapter1_clean.pdf"

# Coordinates for “main text box” (adjust if your file differs)
TOP_MARGIN   = 70      # drop text higher than this
BOTTOM_MARGIN= 780     # drop text lower than this
RIGHT_LIMIT  = 400     # drop blocks whose left edge is beyond this (side notes)

# # PDF page numbers where Chapter 1 begins/ends (0-based!)
# CH1_START    = 3      # <- find by searching for the Rasheeda line
# CH1_END      = 13      # <- one past the last Chapter 1 page

# --------------------------------
l = [1, 11, 22, 32, 43, 54, 65, 75, 87, 99, 111, 122,132]
doc = fitz.open(PDF_PATH)
all_text = []
for i in range(len(l)-1):
  all_text = []
  CH1_START = l[i]+2
  CH1_END = l[i+1]-1+2-2
  for page_num in range(CH1_START, CH1_END):
      page = doc[page_num]
      blocks = page.get_text("blocks", sort=True)

      # Find the lowest image on this page (if any)
      lowest_img_bottom = None
      for img in page.get_images(full=True):
          rects = page.get_image_rects(img[0])
          if rects:
              y_bottom = max(r.y1 for r in rects)
              lowest_img_bottom = y_bottom if lowest_img_bottom is None else max(lowest_img_bottom, y_bottom)

      page_blocks = []
      for b in blocks:
          x0, y0, x1, y1, text, *_ = b
          text = text.strip()
          if not text:
              continue
          # --- filters ---
          if y0 < TOP_MARGIN or y1 > BOTTOM_MARGIN:
              continue
          if x0 > RIGHT_LIMIT:
              continue
          if lowest_img_bottom and y0 > lowest_img_bottom:
              continue
          page_blocks.append((y0, x0, text))

      # Sort by top-left coordinate for natural reading order
      page_blocks.sort(key=lambda blk: (blk[0], blk[1]))
      all_text.extend(["\n"+blk[2] for blk in page_blocks])

# doc.close()

  clean_text = "\n\n".join(all_text)
  OUTPUT_PDF = f"Chapter {i+1}_clean.pdf"

  # ---------- WRITE NEW PDF ----------
  styles = getSampleStyleSheet()
  story  = [Paragraph(p, styles["Normal"]) for p in clean_text.split("\n\n")]

  SimpleDocTemplate(OUTPUT_PDF).build(story)
  print(f"Created {OUTPUT_PDF}")
doc.close()
