from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import numpy as np


class FeatureExtractor:

    def __init__(self):

        base_model = VGG16(weights='imagenet')

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc1').output
        )

    def extract_inputs(self, img):

        image = img.resize((224, 224))

        image = image.convert('RGB')

        image_array = np.array(image)

        x = np.expand_dims(image_array, axis=0)

        x = preprocess_input(x)

        features = self.model.predict(x)[0]

        return features


# img = Image.open("Database/C01-00001.png")
# FeatureExtractor = FeatureExtractor()
# feeature = FeatureExtractor.extract_inputs(img)
# print(feeature)
