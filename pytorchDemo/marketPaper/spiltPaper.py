##############################安装kraken##############################
# pip install kraken
# pip install click==7.1.2
#
#####################################################################
import layoutparser as lp
from PIL import Image

# 读取图像
image = Image.open('testPaper.jpg')

# 加载预训练模型
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# 进行版面分析
layout = model.detect(image)

# 可视化结果
lp.draw_box(image, layout, box_width=3).show()

# 提取并保存每个文本区域
for idx, block in enumerate(layout):
    if block.type == "Text":
        segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
        segment_image.save(f'segment_{idx}.jpg')