import collections
import os
import re, six

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.image.caltech_birds import CaltechBirds2011, CaltechBirdsInfo

_DESCRIPTION = """\
Caltech-UCSD Birds 200 (CUB-200) is an image dataset with photos 
of 200 bird species (mostly North American). The total number of 
categories of birds is 200 and there are 6033 images in the 2010 
dataset and 11,788 images in the 2011 dataset.
Annotations include bounding boxes, segmentation labels.
"""

_URL = ("http://www.vision.caltech.edu/visipedia/CUB-200.html")
_CITATION = """\
@techreport{WelinderEtal2010,
Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
Institution = {California Institute of Technology},
Number = {CNS-TR-2010-001},
Title = {{Caltech-UCSD Birds 200}},
Year = {2010}
}
"""
_NAME_RE = re.compile(r"((\w*)/)*(\d*).(\w*)/(\w*.jpg)$")


class CaltechBirds(CaltechBirds2011):
    """Caltech Birds 2011 dataset. Adapted from tensorflow_datasets to include binary attributes."""

    VERSION = tfds.core.Version("0.1.0")

    @property
    def _caltech_birds_info(self):
        return CaltechBirdsInfo(
            name=self.name,
            images_url="http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
            split_url=None,
            annotations_url="http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz"
        )

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # Images are of varying size
                "image": tfds.features.Image(),
                "image/filename": tfds.features.Text(),
                "label": tfds.features.ClassLabel(num_classes=200),
                "label_name": tfds.features.Text(),
                "bbox": tfds.features.BBoxFeature(),
                "attributes": tfds.features.Tensor(shape=(312,), dtype=tf.float32),
                "segmentation_mask": tfds.features.Image(shape=(None, None, 1)),
            }),
            supervised_keys=("image", "label"))

    def _generate_examples(self, archive, file_names, annotations):
        """Generate birds images, labels and bounding box given the directory path.
        Args:
            archive: object that iterates over the zip
            file_names : list of train/test image file names obtained from mat file
            annotations : dict of image file names and bbox attributes, segmentation
              labels
        Yields:
            Image path, Image file name, its corresponding label and
            bounding box values
        """

        for fname, fobj in archive:
            res = _NAME_RE.match(fname)

            # Checking if filename is present in respective train/test list

            if not res or "/".join(fname.split("/")[-2:]) not in file_names:
                continue
            matches = res.groups()
            label_name = matches[-2].lower()
            label_key = int(matches[-3]) - 1
            file_name = matches[-1].split(".")[0]
            segmentation_mask = annotations[file_name][1]
            attributes = annotations[file_name][2]

            height, width = segmentation_mask.shape

            bbox = self._get_bounding_box_values(annotations[file_name][0], width,
                                                 height)

            if tf.is_tensor(fobj):
                img = fobj
            elif isinstance(fobj, six.string_types):
                with tf.io.gfile.GFile(image_or_path_or_fobj, 'rb') as image_f:
                    encoded_image = image_f.read()
                    img = tf.image.decode_image(
                        encoded_image, channels=3)
            else:
                encoded_image = fobj.read()
                img = tf.image.decode_image(
                    encoded_image, channels=3)

            img = tf.image.crop_and_resize(
                tf.expand_dims(img, 0),
                np.array([[min(bbox[0], 1.0), min(bbox[1], 1.0), min(bbox[2], 1.0), min(bbox[3], 1.0)]]),
                [0],
                (128, 128),
                method='bilinear',
                extrapolation_value=0,
                name=None
            )[0]

            yield fname, {
                "image":
                    img.numpy().astype(np.uint8),
                "image/filename":
                    fname,
                "label":
                    label_key,
                "label_name":
                    label_name,
                "bbox":
                    tfds.features.BBox(
                        ymin=min(bbox[0], 1.0),
                        xmin=min(bbox[1], 1.0),
                        ymax=min(bbox[2], 1.0),
                        xmax=min(bbox[3], 1.0)),
                "attributes": attributes.astype(np.float32),
                "segmentation_mask":
                    segmentation_mask[:, :, np.newaxis],
            }

    def _split_generators(self, dl_manager):

        download_path = dl_manager.download([
            self._caltech_birds_info.images_url,
        ])

        extracted_path = dl_manager.download_and_extract([
            self._caltech_birds_info.images_url,
            self._caltech_birds_info.annotations_url
        ])

        image_names_path = os.path.join(extracted_path[0],
                                        "CUB_200_2011/images.txt")
        split_path = os.path.join(extracted_path[0],
                                  "CUB_200_2011/train_test_split.txt")
        bbox_path = os.path.join(extracted_path[0],
                                 "CUB_200_2011/bounding_boxes.txt")
        attribute_path = os.path.join(extracted_path[0],
                                      "CUB_200_2011/attributes/image_attribute_labels.txt")

        train_list, test_list = [], []
        img_dict = {}
        attributes = collections.defaultdict(list)

        with tf.io.gfile.GFile(split_path) as f, tf.io.gfile.GFile(
                image_names_path) as f1, tf.io.gfile.GFile(bbox_path) as f2:
            for line, line1, line2 in zip(f, f1, f2):
                img_idx, val = line.split()
                idx, img_name = line1.split()
                res = _NAME_RE.match(img_name)
                matches = res.groups()
                attributes[matches[-1].split(".")[0]].append(line2.split()[1:])
                img_dict[img_idx] = (matches[-1].split(".")[0], -np.ones(312))
                if img_idx == idx:
                    if int(val) == 1:
                        train_list.append(img_name)
                    else:
                        test_list.append(img_name)

        for root, _, files in tf.io.gfile.walk(extracted_path[1]):
            for fname in files:
                if fname.endswith(".png"):
                    with tf.io.gfile.GFile(os.path.join(root, fname), "rb") as png_f:
                        mask = tfds.core.lazy_imports.cv2.imdecode(
                            np.fromstring(png_f.read(), dtype=np.uint8), flags=0)
                    attributes[fname.split(".")[0]].append(mask)

        with tf.io.gfile.GFile(attribute_path) as f:
            for line in f:
                img_idx, attr_idx, val = line.split()[:3]
                img_dict[img_idx][1][int(attr_idx) - 1] = float(val)

        for k, v in img_dict.items():
            attributes[v[0]].append(v[1])

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(download_path[0]),
                    "file_names": train_list,
                    "annotations": attributes,
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(download_path[0]),
                    "file_names": test_list,
                    "annotations": attributes,
                }),
        ]
