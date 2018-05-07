import numpy as np
from PIL import Image
import keras
import config

#size for rescaling images
size = (256, 256)
channels = 3
model = keras.models.load_model(config.model_file)


def get_image_processed(img, method=Image.ANTIALIAS):
    """takes PIL image file and returns processed NumPy image
    1. image resized
    2. image normalised to [-0.5;0.5]"""
    img = img.resize(size)
    img = np.array(img)
    img = img.astype(np.float32)

    return img / 255.0


def denorm(img):
    """denormalises image from [-0.5;0.5] to [0;255]"""
    img = img*255.0
    img = np.clip(img.astype(np.uint8), 0, 255)
    return img


def deprocess(img, size):
    """takes NumPy image returns PIL Image
    1. denormalises image from [-0.5;0.5] to [0;255]
    2. deresizes image to size"""
    img = Image.fromarray(denorm(img))
    return img.resize(size)


def predict_image(img, model=model):
    """get predicted probabilities for img"""
    # check that image is resized already
    if img.shape[:2] != size:
        raise Exception("Image should be resized")
    p = model.predict(np.array([img]))[0][0]
    return np.array([p, 1-p])


def get_heatmap_and_pred(img, occlusion_size=(64, 64), stride=(16, 16), occlusion_color=0):
    """create heatmap of feature importance for class predicted as most probable"""
    # check that image is resized already
    if img.shape[:2] != size:
        raise Exception("Image should be resized")

    cumulative = np.zeros(shape=size)
    used = np.zeros(shape=size)
    pred = predict_image(img)
    label = np.argmax(pred)

    # we iterate over image occluding small regions
    # and add predicted probability of true class to every occluded pixel on heatmap
    # so that most important pixels have small average prob among all occlusions
    for offset_x in range(-occlusion_size[1], size[1], stride[1]):
        if offset_x + occlusion_size[1] > size[1]:
            break
        for offset_y in range(-occlusion_size[0], size[0], stride[0]):
            if offset_y + occlusion_size[0] > size[0]:
                break
            # create copy of image array so we don't occlude it too
            img_occluded = img.copy()
            img_occluded[offset_y:offset_y + occlusion_size[0], offset_x:offset_x + occlusion_size[1]] = occlusion_color

            # get probability of class when part of image is occluded
            prob = predict_image(img_occluded)[label]
            cumulative[offset_y:offset_y + occlusion_size[0], offset_x:offset_x + occlusion_size[1]] += prob
            # increase by 1 counter of occlusion times
            used[offset_y:offset_y + occlusion_size[0], offset_x:offset_x + occlusion_size[1]] += 1

    # get average prediction when the pixel is occluded
    heatmap = cumulative / used
    # normalize heatmap to [-0.5;0.5]
    heatmap = (heatmap - heatmap.min())
    heatmap = heatmap / heatmap.max() - 0.5
    # reverse so that most important features are more visible
    heatmap *= -1
    return heatmap, pred


def get_result(img):
    original_size = img.size

    processed = get_image_processed(img)
    heatmap, pred = get_heatmap_and_pred(processed, config.occlusion_size, config.stride, config.occlusion_color)

    print("Got heatmap and prediction " + str(pred))
    # join original image and deprocessed heatmap
    img.putalpha(deprocess(heatmap, original_size))

    return img, deprocess(heatmap, original_size), pred
