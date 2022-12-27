# this app extracts the mnist data sets for use with from http://yann.lecun.com/exdb/mnist/
from pathlib import Path
from typing import Any, Optional

from mnist import MNIST
from numpy import ndarray
from PIL import Image


class MnistDecoder(BaseModel):
    input_path: Path
    output_path: Path
    image_width: int = 28
    image_height: int = 28
    mndata: Optional[MNIST] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.mndata = self._load_mndata()

    def _load_mndata(self) -> MNIST:
        print(f"etl input path: {self.input_path.resolve().as_posix()}")
        mndata: MNIST = MNIST(self.input_path.resolve().as_posix())
        mndata.gz = True
        return mndata

    def export_data(self):
        # training data
        images, labels = self.mndata.load_training()
        dataset_name = "train"
        self.export_images(dataset_name=dataset_name, images=images, labels=labels)

        # testing data
        images, labels = self.mndata.load_testing()
        dataset_name = "test"
        self.export_images(dataset_name=dataset_name, images=images, labels=labels)

    def export_images(self, dataset_name: str, images: ndarray, labels: ndarray) -> None:
        for i in range(len(images)):
            category: str = str(labels[i])
            image_data = images[i]

            img: Image = Image.new("RGB", (self.image_height, self.image_width))
            pixels = img.load()
            index: int = 0
            for y in range(self.image_height):
                for x in range(self.image_width):
                    pixels[x, y] = (image_data[index], image_data[index], image_data[index])
                    index += 1

            full_output_path: Path = self.output_path / str(dataset_name) / str(category)
            full_output_path.mkdir(parents=True, exist_ok=True)
            artifact_name: str = f"mnist_{dataset_name}_{category}_{self.image_width}x{self.image_height}_{i}.png"
            img.save((full_output_path / artifact_name).resolve().as_posix())


if __name__ == "__main__":
    decoder: MnistDecoder = MnistDecoder(input_path=Path("data"), output_path=Path("data"))
    decoder.export_data()
