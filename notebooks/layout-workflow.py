## Layout Workflow
## 1 - LOAD IMAGES
## 2 - DETECT LAYOUT WITH CUSTOM MODEL
## 3 - EXTRACT TEXT FROM BOUNDING BOXES
## 4 - STRUCTURE AND SAVE TEXT

import layoutparser as lp
import cv2
import pandas as pd

## 1 - LOAD IMAGES
print("Loading image files...")
image_files = "/nfs/turbo/isr-jtalexnonsen/images/783095-02/783095-02-0053.jpg" # random, unseen card
image = cv2.imread(image_files)

## 2 - DETECT LAYOUT WITH CUSTOM MODEL
print("Detecting layout with custom model...")
model = lp.Detectron2LayoutModel(config_path = "../layout-model-training/outputs/cards-v4/fast_rcnn_R_50_FPN_3x/config.yaml", model_path = "../layout-model-training/outputs/cards-v4/fast_rcnn_R_50_FPN_3x/model_final.pth", extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8], label_map={0: "AGENCY", 1: "AMOUNT", 2: "ID", 3:"LOCATION", 4:"NAME", 5:"OTHER", 6:"STATUS"})

# model = lp.Detectron2LayoutModel(config_path = "/nfs/turbo/isr-jtalexnonsen/models/V2_fast_rcnn_R_50_FPN_3x/config.yaml", model_path = "/nfs/turbo/isr-jtalexnonsen/models/V2_fast_rcnn_R_50_FPN_3x/model_0039999.pth", extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])

layout_result = model.detect(image)
# print(layout_result)

text_blocks = lp.Layout([b for b in layout_result])
image_width = len(image[0])
left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(image)
left_blocks = text_blocks.filter_by(left_interval, center=True)._blocks
left_blocks.sort(key = lambda b:b.coordinates[1])
right_blocks = [b for b in text_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])
text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
print(f"{len(text_blocks)} text blocks detected.")

## 3 - EXTRACT TEXT FROM BOUNDING BOXES
ocr_agent = lp.TesseractAgent(languages='eng')
print("Extracting text...")
for block in text_blocks:
    segment_image = (block
                        .pad(left=15, right=15, top=5, bottom=5)
                        .crop_image(image))
    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)

extract_list = []

for txt in text_blocks.get_texts():
    print(txt, end='\n---\n') # txt is layoutparser.elements.layout_elements.TextBlock
    extract_list.append(txt)

# print(extract_list)

## 4 - STRUCTURE AND SAVE TEXT

# df = pd.DataFrame(extract_list, columns=['A', 'B', 'C', 'D', 'E', 'F']) # solution will need to be a list of lists or dictionary
# df.to_csv('/nfs/turbo/isr-jtalexnonsen/extracts/test.csv', index=None)

# save_blocks = json.dumps(extract_list)
# file_path = "/nfs/turbo/isr-jtalexnonsen/extracts"
# blocks_file = open(file_path+"/test.json", "w")
# blocks_file.write(save_blocks)
# blocks_file.close()

# print(f"Extraction completed. File saved to {file_path}.")
