import cv2
import torch
import numpy as np
import bisenetv2_260MB as bv2
from torchvision import transforms

INPUt_SIZE = (160, 96)

def preprocess(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUt_SIZE[1], INPUt_SIZE[0]), interpolation=cv2.INTER_AREA) 
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0).to(device)
    return image

# 推理函数
def infer(model, image, device):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess(image, device)
        output = model(input_tensor)[0]
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
    return mask

# 叠加半透明图层
def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]  # 红色标注
    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

# 处理单张图片
def process_image(image_path, model, device):
    image = cv2.imread(image_path)
    mask = infer(model, image, device)
    result = overlay_mask(image, mask)
    cv2.imwrite("output.png", result)
    print("Image processing complete. Saved as output.png")

# 处理视频
def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mask = infer(model, frame, device)
        result = overlay_mask(frame, mask)
        out.write(result)
    
    cap.release()
    out.release()
    print("Video processing complete. Saved as output.avi")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_model_path = "res/model_260MB_epoch_100.pth"
    model = bv2.BiSeNetV2(2).to(device)
    model.load_state_dict(torch.load(pth_model_path, map_location=device, weights_only=True))
    
    input_type = input("Enter 'image' for image processing or 'video' for video processing: ")
    input_path = input("Enter the file path: ")
    
    if input_type == 'image':
        process_image(input_path, model, device)
    elif input_type == 'video':
        process_video(input_path, model, device)
    else:
        print("Invalid input type.")