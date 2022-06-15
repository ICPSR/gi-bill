## Tests Tesseract on Great Lakes by performing OCR on a scanned card

import layoutparser as lp
import cv2

image_files = "/nfs/turbo/isr-jtalexnonsen/sample_scans/274-01-0013.jpg"
image = cv2.imread(image_files)

ocr_agent = lp.TesseractAgent()
ocr_text = ocr_agent.detect(image)

print(ocr_text)