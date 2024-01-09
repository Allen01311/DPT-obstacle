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


#找DPT後的最大輪廓&矩形
def find_contours(image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(image_path)
    resize_image = cv2.resize(image, (width, height))
    img_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img_gray)
    equalized_image = cv2.equalizeHist(clahe_image)
    _, thresh = cv2.threshold(equalized_image, 170, 255, cv2.THRESH_BINARY)
    
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(denoised_image, kernel, iterations = 1)
    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    
    # denoised_image = cv2.GaussianBlur(thresh, (5, 5), 0)
    
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
        
        # 最大外接矩形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        # print("Approximated Polygon:","\n" ,approx_polygon)
        x, y, w, h = cv2.boundingRect(approx_polygon)
        #最大外接矩形的4個角座標
        print("Top-left corner: ({}, {})".format(x, y))
        print("Top-right corner: ({}, {})".format(x + w, y))
        print("Bottom-left corner: ({}, {})".format(x, y + h))
        print("Bottom-right corner: ({}, {})".format(x + w, y + h))
        print('---------------------------------------------')
        
        regions = [
            ((0, 0), (x, y)),                               # 左上
            ((x, 0), (x + w, y)),                           # 上
            ((x + w, 0), (resize_image.shape[1], y)),       # 右上
            ((0, y + h), (x, resize_image.shape[0])),       # 左下
            ((x, y + h), (x + w, resize_image.shape[0])),   # 下
            ((x + w, y + h), (resize_image.shape[1], resize_image.shape[0])),  # 右下
            ((0, y), (x, y + h)),                           # 左
            ((x + w, y), (resize_image.shape[1], y + h))    # 右
        ]
        
        region_x = 300
        region_y = 150
        region_w = 256
        region_h = 180
        max_black_pixels = 0
        max_black_region = None
        
        if (x < region_x + region_w and x + w > region_x and
            y < region_y + region_h and y + h > region_y):
            print("Bounding rectangle intersects with the region")
            # 計算每個區域的黑色和白色像素
            for idx, (start_point, end_point) in enumerate(regions):
                x1, y1 = start_point
                x2, y2 = end_point
                region = thresh[y1:y2, x1:x2]

                total_pixels = region.size
                if total_pixels > 0:
                    black_pixels = np.count_nonzero(region == 0)  # 像素值為0表示黑色
                    white_pixels = total_pixels - black_pixels

                    print(f"Region {idx + 1}: Black Pixels: {black_pixels}, White Pixels: {white_pixels}")

                    if black_pixels > max_black_pixels:
                        max_black_pixels = black_pixels
                        max_black_region = ((x1, y1), (x2, y2))
                else:
                    print(f"Region {idx + 1} is empty.")
            print('---------------------------------------------')
            
            (max_top_left, max_bottom_right) = max_black_region
            print(f"Region with the most black pixels: Top Left: {max_top_left}, Bottom Right: {max_bottom_right}, Black Pixels: {max_black_pixels}")
            
            
            if max_top_left[0] == 0 and max_top_left[1] == 0:
                print("避障方向：左上")
            elif max_top_left[1] == 0:
                print("避障方向：上")
            elif max_bottom_right[0] == resize_image.shape[1] and max_top_left[1] == 0:
                print("避障方向：右上")
            elif max_top_left[0] == 0 and max_bottom_right[1] == resize_image.shape[0]:
                print("避障方向：左下")
            elif max_bottom_right[1] == resize_image.shape[0]:
                print("避障方向：下")
            elif max_bottom_right[0] == resize_image.shape[1] and max_bottom_right[1] == resize_image.shape[0]:
                print("避障方向：右下")
            elif max_top_left[0] == 0:
                print("避障方向：左")
            elif max_bottom_right[0] == resize_image.shape[1]:
                print("避障方向：右")
            else:
                print("未知方向")
            print('---------------------------------------------')
            # top_area = w * y
            # bottom_area = w * (resize_image.shape[0] - (y + h))
            # left_area = h * x
            # right_area = h * (resize_image.shape[1] - (x + w))
            # print("Area above rectangle: {}".format(top_area))
            # print("Area below rectangle: {}".format(bottom_area))
            # print("Area to the left of rectangle: {}".format(left_area))
            # print("Area to the right of rectangle: {}".format(right_area))
        
            # max_area = max(top_area, bottom_area, left_area, right_area)
            # print("Largest area: {}".format(max_area))
        else:
            print("No obstacle detected in the region")
            
        #以下繪圖-------------------------------------------------------------------------------------------------------
        
        #針對resize_image繪圖
        
        #最大輪廓
        # cv2.drawContours(resize_image, [largest_contour], -1, (0, 255, 0), 4)   
        #最大多邊形
        # cv2.drawContours(resize_image, [approx_polygon], -1, (0, 255, 255), 4)  
                      
        #外接矩形corner座標點
        cv2.circle(resize_image, (x, y), 6, (0, 0, 255), cv2.FILLED)  # Red
        cv2.circle(resize_image, (x + w, y), 6, (0, 255, 0), cv2.FILLED)  # Green
        cv2.circle(resize_image, (x, y + h), 6, (255, 0, 0), cv2.FILLED)  # Blue
        cv2.circle(resize_image, (x + w, y + h), 6, (0, 255, 255), cv2.FILLED)  # Yellow
        
        
        #四周面積繪製
        cv2.rectangle(resize_image, (0, 0), (x, y), (255, 225, 0), 2)      #左上
        cv2.rectangle(resize_image, (x, 0), (x + w, y), (0, 140, 255), 2)   #上
        cv2.rectangle(resize_image, (x + w, 0), (resize_image.shape[1], y), (255, 225, 0), 2)   #右上
        cv2.rectangle(resize_image, (0, y), (x, y + h), (0, 255, 150), 2)   #左
        cv2.rectangle(resize_image, (x + w, y), (resize_image.shape[1], y + h), (135, 0, 245), 2)   #右
        cv2.rectangle(resize_image, (0, y + h), (x, resize_image.shape[0]), (0, 140, 255), 2)   #左下
        cv2.rectangle(resize_image, (x, y + h), (x + w, resize_image.shape[0]), (255, 225, 0), 2)   #下
        cv2.rectangle(resize_image, (x + w, y + h), (resize_image.shape[1], resize_image.shape[0]), (0, 140, 255), 2)   #右下
        

        #最大外接矩形
        cv2.rectangle(resize_image, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(regions):
            if ((x1, y1), (x2, y2)) == max_black_region:
                cv2.rectangle(resize_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 最多黑色像素的區域用綠色標示
            else:
                cv2.rectangle(resize_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 其他區域用紅色標示

        #自訂區域
        cv2.rectangle(resize_image, (region_x, region_y), (region_x + region_w, region_y + region_h), (0, 255, 255), 2)
        
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
        
        #對二值化的圖繪圖
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.circle(thresh_colored , image_center_point, 10, (255, 0, 0), cv2.FILLED)
        cv2.rectangle(thresh_colored, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        cv2.circle(thresh_colored, (x, y), 6, (0, 0, 255), cv2.FILLED)
        cv2.circle(thresh_colored, (x + w, y), 6, (0, 0, 255), cv2.FILLED) 
        cv2.circle(thresh_colored, (x, y + h), 6, (0, 0, 255), cv2.FILLED) 
        cv2.circle(thresh_colored, (x + w, y + h), 6, (0, 0, 255), cv2.FILLED)
        #自訂區域
        cv2.rectangle(thresh_colored, (region_x, region_y), (region_x + region_w, region_y + region_h), (255, 255, 0), 2)
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(regions):
            if ((x1, y1), (x2, y2)) == max_black_region:
                cv2.rectangle(thresh_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 最多黑色像素的區域用綠色標示
            else:
                cv2.rectangle(thresh_colored, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 其他區域用紅色標示
        plt.subplot(122), plt.imshow(cv2.cvtColor(thresh_colored, cv2.COLOR_BGR2RGB)), plt.title('Colored Drawing on Binary Image')
        colored_drawing_path = os.path.join(output_folder_path, f"colored_drawing_{os.path.basename(image_path)}")
        cv2.imwrite(colored_drawing_path, cv2.cvtColor(thresh_colored, cv2.COLOR_BGR2RGB))
        
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
    
    # compute depth maps
    # run(
    #     args.input_path,
    #     args.output_path,
    #     args.model_weights,
    #     args.model_type,
    #     args.optimize,
    # )
    print('')
    input_folder_path = "D:/image_experience/DPT-main/output_monodepth"
    output_folder_path = "D:/image_experience/DPT-main/contour_obstacle_output"

    process_and_save_images(input_folder_path, output_folder_path)
    
    cv2.destroyAllWindows()
