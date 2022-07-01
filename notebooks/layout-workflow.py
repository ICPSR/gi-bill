## Layout Workflow
## 1 - LOAD IMAGES
## 2 - DETECT LAYOUT WITH CUSTOM MODEL
## 3 - EXTRACT TEXT FROM BOUNDING BOXES
## 4 - STRUCTURE AND SAVE TEXT

import layoutparser as lp
import cv2
import pandas as pd
import glob
import os

# image_files = sorted(glob.glob('/nfs/turbo/isr-jtalexnonsen/sample_scans/'+'/*.jpg'))
# image_files = sorted(glob.glob('/nfs/turbo/isr-jtalexnonsen/images/783095-23/'+'/*.jpg'))
image_files = sorted(glob.glob('/nfs/turbo/isr-jtalexnonsen/flat_images/'+'/*.jpg'))
print(f"Loading {len(image_files)} image files...")
ocr_agent = lp.TesseractAgent(languages='eng')
print("Initializing OCR engine...")

df_list = []

for file in image_files:
    file_name = os.path.basename(file)
    image = cv2.imread(file)
    print(f"Detecting layout for {file_name}...")
    model = lp.Detectron2LayoutModel(config_path = "../layout-model-training/outputs/cards-v4/fast_rcnn_R_50_FPN_3x/config.yaml", model_path = "../layout-model-training/outputs/cards-v4/fast_rcnn_R_50_FPN_3x/model_final.pth", extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8], label_map={0: "AGENCY", 1: "AMOUNT", 2: "ID", 3:"LOCATION", 4:"NAME", 5:"OTHER", 6:"STATUS"})
    layout_result = model.detect(image)
    text_blocks = lp.Layout([b for b in layout_result])
    image_width = len(image[0])
    left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(image)
    left_blocks = text_blocks.filter_by(left_interval, center=True)._blocks
    left_blocks.sort(key = lambda b:b.coordinates[1])
    right_blocks = [b for b in text_blocks if b not in left_blocks]
    right_blocks.sort(key = lambda b:b.coordinates[1])
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
    print(f"{len(text_blocks)} text blocks detected.")
    for block in text_blocks:
        segment_image = (block
                        .pad(left=15, right=15, top=5, bottom=5)
                        .crop_image(image))
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    df = text_blocks.to_dataframe()
    df['file_name'] = file_name
    df_list.append(df)
    print(f"Text extracted for {file_name}...")

df_output = pd.concat(df_list, axis=0, ignore_index=True)
print(f"{len(df_list)} dataframes concatenated...")

df_pivot = df_output.pivot_table(index=['file_name'],
                                     columns='type', 
                                     values='text',
                                     aggfunc=lambda x: ' '.join(x))

out_path = '/nfs/turbo/isr-jtalexnonsen/extracts/'
df_pivot.to_csv(out_path+'flat_images.csv')
print(f"Extraction completed. File saved to {out_path}.")

## 1 - LOAD IMAGES
# file_name = "783095-02-0053.jpg"
# image_files = image_dir+file_name # random, unseen card
# image = cv2.imread(image_files)

## 2 - DETECT LAYOUT WITH CUSTOM MODEL
# print("Detecting layout with custom model...")
# model = lp.Detectron2LayoutModel(config_path = "../layout-model-training/outputs/cards-v5/fast_rcnn_R_50_FPN_3x/config.yaml", model_path = "../layout-model-training/outputs/cards-v5/fast_rcnn_R_50_FPN_3x/model_final.pth", extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8], label_map={0: "AGENCY", 1: "AMOUNT", 2: "ID", 3:"LOCATION", 4:"NAME", 5:"OTHER", 6:"STATUS"})
# layout_result = model.detect(image)
# text_blocks = lp.Layout([b for b in layout_result])
# image_width = len(image[0])
# left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(image)
# left_blocks = text_blocks.filter_by(left_interval, center=True)._blocks
# left_blocks.sort(key = lambda b:b.coordinates[1])
# right_blocks = [b for b in text_blocks if b not in left_blocks]
# right_blocks.sort(key = lambda b:b.coordinates[1])
# text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
# print(f"{len(text_blocks)} text blocks detected.")

## 3 - EXTRACT TEXT FROM BOUNDING BOXES
# ocr_agent = lp.TesseractAgent(languages='eng')
# print("Extracting text...")
# for block in text_blocks:
#     segment_image = (block
#                         .pad(left=15, right=15, top=5, bottom=5)
#                         .crop_image(image))
#     text = ocr_agent.detect(segment_image)
#     block.set(text=text, inplace=True)

## 4 - STRUCTURE AND SAVE TEXT
# print("Saving text...")
# file_path = "/nfs/turbo/isr-jtalexnonsen/extracts"
# df = text_blocks.to_dataframe()
# df['file_name'] = file_name
# df_pivot = df.set_index(['file_name'], drop=True).unstack("type")
# df_pivot.to_csv('/nfs/turbo/isr-jtalexnonsen/extracts/test.csv', index=None)
# print(f"Extraction completed. File saved to {file_path}.")
