import os
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import ImageFile
from PIL import ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True

mean = [0.46058371663093567, 0.4468299448490143, 0.3933452367782593]
std = [0.2081516534090042, 0.20522701740264893, 0.20532798767089844]
classes = ["cat", "dog", "other"]

dict_to_russian ={'cat': 'кот/кошка','dog': 'собака', 'other': 'другое'}
classifier_path = os.path.join(os.getcwd(), "models", "cat_dog_other.pth")
model_for_inference = torch.load(classifier_path)

mean_s = [0.48001906275749207, 0.4483553469181061, 0.39609295129776]
std_s = [0.22524648904800415, 0.22261489927768707, 0.22534485161304474]
classes_s = ["other", "samoyed"]
model_for_inference_samoyed = torch.load(os.path.join(os.getcwd(), "models", "samoyed.pth"))

detector_model_path = os.path.join(os.getcwd(), "models", "best.pt")
model_for_detection = YOLO(detector_model_path)

samoyed_detector_model_path = os.path.join(os.getcwd(),
                                   "models", "samoyed.pt")
samoyed_model_for_detection = YOLO(samoyed_detector_model_path)

def image_transforms(mean,std):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

def classify_bytes(model, image_trf, image_bytes, cls):
    """This function classifies an image given as bytes using a model, image transforms, classes
    and returns the predicted class and its percentage.
    Args:
    model (torch.nn.Module): The model used for classification.
    image_trf (torchvision.transforms): The transforms applied to the image.
    image_bytes (bytes): The image as bytes.
    classes (list): List of classes.
    Returns:
    cls (str): Predicted class.
    percentage (float): Percentage of the predicted class. """
    with torch.no_grad():
        model = model.eval()
        image = image_trf(image_bytes).float()
        image = torch.unsqueeze(image, dim=0)

        output = model(image)
        _, predicted = torch.max(output.data, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    return cls[predicted.item()], percentage[predicted.item()].item()

def yolo_to_xml_bbox(bbox, width, height):
    """Convert YOLO bbox to XML bbox format.
    Args:
        bbox (list): YOLO bbox in format [x, y, w, h].
        width (int): Image width.
        height (int): Image height.
    Returns:
        list: XML bbox in format [xmin, ymin, xmax, ymax].
    """
    w_half_len = (bbox[2] * width) / 2
    h_half_len = (bbox[3] * height) / 2
    xmin = int((bbox[0] * width) - w_half_len)
    ymin = int((bbox[1] * height) - h_half_len)
    xmax = int((bbox[0] * width) + w_half_len)
    ymax = int((bbox[1] * height) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def draw_image(img, bbox, i):
    """This function draws an image with a given bounding box (bbox) and index (i).
    It uses the ImageDraw module to draw the rectangle and
    then saves the image to the specified save_path created at specified index."""
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline="red", width=2)
    save_path = os.path.join(os.getcwd(), "image_folders", f"example{i}.jpg")
    img.save(save_path)

def make_classification_img(img, model=model_for_inference, image_transforms=image_transforms,
                            mean=mean, std=std, classes=classes):
    """This function takes an image as an argument and returns a string with the
    classification of the image according to the model_for_inference,
    image_transforms and classes.
    The result is translated to Russian using the dict_to_russian dictionary."""
    res = classify_bytes(model, image_transforms(mean, std), img, classes)
    if res[0] in dict_to_russian:
        return "Класс: " + dict_to_russian[res[0]] + ". Вероятность: " + str(round(res[1], 2)) + "%."
    else:
        return "samoyed"

def make_detection_img(model, img):
    """This function takes a model and an image as arguments and returns the boxes,
    classes and probabilities of the detected objects in the image.
    It uses the model to detect objects in the image with a confidence of 0.5."""
    result = model(img, conf=0.5)[0]
    cls = result.boxes.cls.cpu().numpy()
    probs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    return boxes, cls, probs
