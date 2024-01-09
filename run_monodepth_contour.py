"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse
import math
import util.io
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util.misc import visualize_attention

width = 856
height = 480
# def extract_frames(video_path, output_path, num_frames):
#     vidcap = cv2.VideoCapture(video_path)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames_to_skip = total_frames // num_frames

#     success,image = vidcap.read()
#     count = 0
#     frame_number = 0

#     while success:
#         if frame_number % frames_to_skip == 0:
#             rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#             cv2.imwrite(f"{output_path}/frame{count:04d}.jpg", rotated_image) 
#             count += 1
#         success,image = vidcap.read()
#         frame_number += 1

#         if count == num_frames:
#             break

#     vidcap.release()
    

#計算圖像中心點至邊的最短距離(直接帶公式)
def point_distance_line(point, line_point1, line_point2):
    #檢查直線兩點是否重疊，如果重疊則回傳到其中一點的距離
    if np.array_equal(line_point1, line_point2):
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)
    
    #Ax + By + C = 0
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]

    #點到直線垂直距離公式
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    
    # 計算座標點H
    H = ( B**2 * point[0] - A*B*point[1] - A*C ) / (A**2 + B**2)
    K = ( -A*B*point[0] + A**2 * point[1] - B*C ) / (A**2 + B**2)
    h_point = np.array([H, K])
    return distance, h_point


#找DPT後的最大輪廓&矩形
def find_contours(image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(image_path)
    resize_image = cv2.resize(image, (width, height))
    img_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    
    
        
    # enhanced_image = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(11, 11))
    clahe_image = clahe.apply(img_gray)
    equalized_image = cv2.equalizeHist(clahe_image)
    _, thresh = cv2.threshold(equalized_image, 170, 255, cv2.THRESH_BINARY)
    
    #儲存gray_image
    gray_output_path = os.path.join(output_folder_path, f"gray_{os.path.basename(image_path)}")
    cv2.imwrite(gray_output_path, thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    if largest_contour is not None:
        total_area = resize_image.shape[0] * resize_image.shape[1]  #圖像總面積
        
        #最大輪廓面積
        area = cv2.contourArea(largest_contour)
        print(' Largest Contour Area :', area)
        print('---------------------------------------------')
        #-----------------------------------------------------------------------
        
        # 計算整張影像的中心點
        image_center_point = (int(resize_image.shape[1] / 2), int(resize_image.shape[0] / 2))
        print(' Image Center Point:', image_center_point)
        print('---------------------------------------------')
        #-----------------------------------------------------------------------
        
        # 最大外接矩形(可轉向)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        # print("Approximated Polygon:","\n" ,approx_polygon)
        rect = cv2.minAreaRect(approx_polygon)
        
        #box用途:方便繪圖
        box = cv2.boxPoints(rect)
        box = np.intp(box)
            
        #(x,y):外接矩形中心點 ; (w,h):寬、高 ; angle:旋轉角度
        (x, y), (w, h), angle = rect
        
        #計算外接矩形的中心座標
        center_x = int(x)
        center_y = int(y)
        print(' 外接矩形的中心座標點:', '(', center_x, ',', center_y, ')')
        
        #用途:方便計算座標點轉向位置
        half_width = int(w / 2)
        half_height = int(h / 2)
        
        #計算矩形的四個角的相對坐標
        corner1 = (-half_width, -half_height)  # 左上角
        corner2 = (half_width, -half_height)   # 右上角
        corner3 = (half_width, half_height)    # 右下角
        corner4 = (-half_width, half_height)   # 左下角
        
        #角度>0 : 向左
        #角度<0 : 向右
        print(' 旋轉角度 : ',round(float(angle), 3),'度')
        
        #將相對坐標旋轉回原始坐標
        cos_theta = np.cos(np.radians(angle))
        sin_theta = np.sin(np.radians(angle))
        
        #由於angle預設為正(往左旋轉直到回正)，
        #若angle > 45度下去做cos、sin的相對座標旋轉回原始座標
        #會造成原始座標判斷錯誤，故以此判斷式校正錯誤的座標
        if abs(angle) > 45:
            rotated_corner2 = (
                int(center_x + corner1[0] * cos_theta - corner1[1] * sin_theta),
                int(center_y + corner1[0] * sin_theta + corner1[1] * cos_theta)
            )
            rotated_corner3 = (
                int(center_x + corner2[0] * cos_theta - corner2[1] * sin_theta),
                int(center_y + corner2[0] * sin_theta + corner2[1] * cos_theta)
            )
            rotated_corner4 = (
                int(center_x + corner3[0] * cos_theta - corner3[1] * sin_theta),
                int(center_y + corner3[0] * sin_theta + corner3[1] * cos_theta)
            )
            rotated_corner1 = (
                int(center_x + corner4[0] * cos_theta - corner4[1] * sin_theta),
                int(center_y + corner4[0] * sin_theta + corner4[1] * cos_theta)
            )
            print('∵angle > 45 , ∴進行校正')
        else:
            # center_x、center_y : 讓旋轉前後的中心點一致
            # [new_x] = R * [x] = [cos(Θ) -sin(Θ)] * [x]
            # [new_y]       [y]   [sin(Θ)  cos(Θ)]   
            rotated_corner1 = (
                int(center_x + corner1[0] * cos_theta - corner1[1] * sin_theta),
                int(center_y + corner1[0] * sin_theta + corner1[1] * cos_theta)
            )
            rotated_corner2 = (
                int(center_x + corner2[0] * cos_theta - corner2[1] * sin_theta),
                int(center_y + corner2[0] * sin_theta + corner2[1] * cos_theta)
            )
            rotated_corner3 = (
                int(center_x + corner3[0] * cos_theta - corner3[1] * sin_theta),
                int(center_y + corner3[0] * sin_theta + corner3[1] * cos_theta)
            )
            rotated_corner4 = (
                int(center_x + corner4[0] * cos_theta - corner4[1] * sin_theta),
                int(center_y + corner4[0] * sin_theta + corner4[1] * cos_theta)
            )
        print('---------------------------------------------')
        
        # 轉換為整數座標
        rotated_corner1 = tuple(map(int, rotated_corner1))
        rotated_corner2 = tuple(map(int, rotated_corner2))
        rotated_corner3 = tuple(map(int, rotated_corner3))
        rotated_corner4 = tuple(map(int, rotated_corner4))

        #矩形四個角落的座標點
        print(' 矩形四個角落的座標點')
        print(' Rotated Corner 1:', rotated_corner1)
        print(' Rotated Corner 2:', rotated_corner2)
        print(' Rotated Corner 3:', rotated_corner3)
        print(' Rotated Corner 4:', rotated_corner4)
        print('---------------------------------------------')
        #-----------------------------------------------------------------------
        
        #計算中心點到四邊的最短距離、座標點
        top_distance, top_h_point = point_distance_line(image_center_point, rotated_corner1, rotated_corner2)
        bottom_distance, bottom_h_point = point_distance_line(image_center_point, rotated_corner3, rotated_corner4)
        left_distance, left_h_point = point_distance_line(image_center_point, rotated_corner1, rotated_corner4)
        right_distance, right_h_point = point_distance_line(image_center_point, rotated_corner2, rotated_corner3)

        print(' Distance to Top:', int(top_distance), '\n','Top垂點座標:', '(', int(top_h_point[0]),',', int(top_h_point[1]), ')', '\n')
        print(' Distance to Bottom:', int(bottom_distance), '\n','Bottom垂點座標:', '(', int(bottom_h_point[0]),',', int(bottom_h_point[1]), ')', '\n')
        print(' Distance to Left:', int(left_distance), '\n','Left垂點座標:', '(', int(left_h_point[0]),',', int(left_h_point[1]), ')', '\n')
        print(' Distance to Right:', int(right_distance), '\n','Right垂點座標:', '(', int(right_h_point[0]), ',', int(right_h_point[1]), ')', '\n')
        print('---------------------------------------------')
        min_distance, min_distance_direction = min(
            (top_distance, 'Top'),
            (bottom_distance, 'Bottom'),
            (left_distance, 'Left'),
            (right_distance, 'Right'),
            key=lambda x: x[0]
        )
        
        if min_distance_direction == 'Top':
            min_h_point = top_h_point
        elif min_distance_direction == 'Bottom':
            min_h_point = bottom_h_point
        elif min_distance_direction == 'Left':
            min_h_point = left_h_point
        else:
            min_h_point = right_h_point
        
        print(' Minimum Distance:', min_distance)
        print(' Move to:', '『', min_distance_direction , '』', 'side')
        print(' Move to:', min_h_point)
        print('---------------------------------------------')
        
        #以下繪圖-------------------------------------------------------------------------------------------------------
        
        #影像中心點與各邊的垂線
        cv2.line(resize_image, image_center_point, tuple(top_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(resize_image, image_center_point, tuple(bottom_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(resize_image, image_center_point, tuple(left_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(resize_image, image_center_point, tuple(right_h_point.astype(int)), (0, 0, 0), 4, cv2.LINE_AA)        
        #影像中心點與最短邊的垂線
        cv2.line(resize_image, image_center_point, tuple(min_h_point.astype(int)), (0, 150, 255), 12, cv2.LINE_AA)
        
        #最大輪廓
        cv2.drawContours(resize_image, [largest_contour], -1, (0, 255, 0), 3)   
        #最大多邊形
        cv2.drawContours(resize_image, [approx_polygon], -1, (0, 255, 255), 3)  
        #最大外接矩形
        cv2.drawContours(resize_image, [box], 0, (255, 0, 0), 4)                
        
        #外接矩形corner座標點
        # cv2.circle(resize_image, rotated_corner1, 3, (0, 0, 255), cv2.FILLED)  # Red
        # cv2.circle(resize_image, rotated_corner2, 3, (0, 255, 0), cv2.FILLED)  # Green
        # cv2.circle(resize_image, rotated_corner3, 3, (255, 0, 0), cv2.FILLED)  # Blue
        # cv2.circle(resize_image, rotated_corner4, 3, (0, 255, 255), cv2.FILLED)  # Yellow
        
        #影像中心點與各邊的垂足點
        cv2.circle(resize_image, tuple(top_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
        cv2.circle(resize_image, tuple(bottom_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
        cv2.circle(resize_image, tuple(left_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
        cv2.circle(resize_image, tuple(right_h_point.astype(int)), 5, (0, 0, 0), cv2.FILLED)
        #最短邊的垂足點
        cv2.circle(resize_image, tuple(min_h_point.astype(int)), 8, (0, 0, 255), cv2.FILLED)
        
        #影像中心點
        cv2.circle(resize_image, image_center_point, 10, (0, 0, 255), cv2.FILLED)
        
        #文字說明
        
        # cv2.putText(resize_image, 'Red:Center', (10, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(resize_image, 'Green:Max Contour', (10, 50), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(resize_image, 'Blue:Max Rectangle', (10, 75), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(resize_image, 'Black:Foot Drop', (10, 100), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(resize_image, 'Orange:Perpendicular line', (10, 125), font, 0.5, (0, 110, 255), 1, cv2.LINE_AA)
        #於圖像顯示判斷結果
        # solution = 'Move to:'+ min_distance_direction + ' side'
        # cv2.putText(resize_image, solution, (150, 390), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        return resize_image


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#儲存找完DPT後的最大輪廓&矩形的圖片
def process_and_save_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print('#################### Start Contour Detection ####################')
    print('')
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        processed_image = find_contours(input_path)
        output_path = os.path.join(output_folder, f"processed_{image_file}")
        cv2.imwrite(output_path, processed_image)
        print(f"【{image_file}】 has already processed")
        print('')
        print('#################################################################')
        print('')

#DPT程式
def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.io.read_image(img_name)

        if args.kitti_crop is True:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if model_type == "dpt_hybrid_kitti":
                prediction *= 256

            if model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)
    
    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # video_file = 'D:/image_experience/DPT-main/2023_12_09_15_33_31.mp4'
    # output_folder = 'video_seg_output'
    # extract_frames(video_file, output_folder, num_frames=10)
    
    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
    print('')
    input_folder_path = "D:/image_experience/DPT-main/output_monodepth"
    output_folder_path = "D:/image_experience/DPT-main/contour_detection_output"

    process_and_save_images(input_folder_path, output_folder_path)
    
    cv2.destroyAllWindows()
