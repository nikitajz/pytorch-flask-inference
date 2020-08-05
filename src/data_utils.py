from urllib.parse import urlparse

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from torch.utils.data import Dataset

s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path.

    Args:
        url (str): S3 url.

    Returns:
        tuple: (bucket_name, s3_path)
    """
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    return bucket_name, s3_path.lstrip('/')


def s3_get_file(url):
    """Download a file from S3.

    Args:
        url (str): http(s)- or S3-compliant url

    Returns:
        bytes: Image
    """
    bucket_name, s3_path = split_s3_path(url)
    try:
        s3_response_object = s3_resource.Object(bucket_name, s3_path).get()
        file_content = s3_response_object['Body'].read()
        return file_content
    except ClientError as exc:
        if int(exc.response["Error"]["Code"]) == 404:
            raise FileNotFoundError("file {} not found".format(url))
        else:
            raise


class ImageDataset(Dataset):

    def __init__(self, urls, transform=None):
        """Remote dataset (http(s) or S3 urls)

        Args:
            urls (list[str]): http(s) or S3 urls
            transform (class, optional): A class instance with __call__ method implemented. Default: None
        """
        self.urls = urls
        self.transform = transform

    @staticmethod
    def _download_image(url):
        """Download image file from http(s) or S3 url.

        Args:
            url (str): http(s)- or S3-compliant url

        Returns:
            bytes: Image
        """
        if url.startswith("s3://"):
            img_bytes = s3_get_file(url)
        elif url.startswith("http"):
            response = requests.get(url)
            if response.status_code == 200:
                img_bytes = response.content
            else:
                raise FileNotFoundError
        else:
            raise ValueError(f"Incorrect url scheme: {url}")
        return img_bytes

    def __len__(self):
        """Length of urls list.

        Returns:
            int: Length of urls list
        """
        return len(self.urls)

    def __getitem__(self, idx):
        """Standard dataset method to fetch an item by index. Downloads an image from url on the fly.

        Args:
            idx (int): Index of dataset item.

        Returns:
            bytes: Image
        """
        img = self._download_image(self.urls[idx])
        if self.transform:
            img = self.transform(img)
        return img
