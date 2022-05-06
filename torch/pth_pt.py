import torch
from nets.deeplabv3_plus import DeepLab
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 导入已经训练好的模型
num_classes = 21
backbone = "mobilenet"
pretrained = True
downsample_factor   = 16
input_shape = [512, 512]
model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)

weight = torch.load('logs/ep001-loss0.709-val_loss0.889.pth')
model.load_state_dict(weight, strict=True)
model = model.eval()
# 注意模型输入的尺寸
example = torch.rand(1, 3, 512, 512)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("logs/resout.pt")

img_path = r"1.jpg"
save_path = r"logs\res8.jpg"
module = torch.jit.load("logs/resout.pt")
image = cv2.imread(img_path)
orininal_h,orininal_w,_ = image.shape
image=cv2.resize(image, (512, 512))
#image.convertTo(image,cv2.CV_32F, 1.0 / 255, 0)
image = image/255.0
image_data  = np.expand_dims(np.transpose((np.array(image, np.float32)), (2, 0, 1)), 0)
img_tensor = torch.from_numpy(image_data)

pr = module.forward(img_tensor)[0]
pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().detach().numpy()

pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
pr = pr.argmax(axis=-1)
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
               (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
               (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               (128, 64, 12)]
seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
image = Image.fromarray(np.uint8(seg_img))
image.save(save_path)

