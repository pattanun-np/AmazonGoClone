import json
from PIL import Image
import annoy
from FeatureExtractor import FeatureExtractor
from tqdm.auto import tqdm
import cv2


class ProductorSearch:

    def __init__(self):
        self.tree = annoy.AnnoyIndex(4096, 'angular')
        self.products = {}
        self.FeatureExtractor = FeatureExtractor()
        file = open("Maping.json", "r")
        self.data = json.load(file)

        # self.tree.load("./Database/database.ann")
        try:
            self.tree.load("./Database/database.ann")
        except:
            print("Could not load database index")
            print("Try to run training")
            self.train()

        finally:
            print("Database index loaded")
        # print(self.data)

    def train(self):
        for idx, product in tqdm(enumerate(self.data), total=len(self.data)):
            img_path = f"./Database/{product['Image']}"
            img = Image.open(img_path)
            embreding = self.FeatureExtractor.extract_inputs(img)
            self.tree.add_item(idx, embreding)
        self.tree.build(10)
        self.tree.save('./Database/database.ann')
        print("Training Done")

    def __getproduct(self, macthed, theshold=0.5):
        results = []

        for idx, dist in zip(macthed[0], macthed[1]):
            if dist <= theshold:
                result = {
                    "product": self.data[idx]["name"],
                    "price": self.data[idx]["price"],
                    "image": f"./Database/{self.data[idx]['Image']}",
                    "distance": dist
                }
            results.append(result)
        return results

    def search(self, imagepath, showimage=False):
        img = Image.open(imagepath)
        # print(img)
        feature = self.FeatureExtractor.extract_inputs(img)
        match = self.tree.get_nns_by_vector(
            feature, n=2, search_k=5, include_distances=True)
        # print(match)
        results = self.__getproduct(match, theshold=0.9)
        # print(results)

        if showimage:
            for i in range(len(results)):
                print(results[i])
                image_show = cv2.imread(results[i]['image'])
                image_show = cv2.resize(image_show, (120, 160))

                cv2.imshow(f"ProductorSearch {i+1}", image_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    ProductorSearch = ProductorSearch()
    ProductorSearch.search(imagepath="test1.jpg", showimage=True)
# ProductorSearch.train()
