import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_transform = A.Compose([
    # 非破坏性转换
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5), 
    ], p=0.7),
    # 非刚体转换
    A.OneOf([
        A.ElasticTransform(p=0.5, border_mode=cv2.BORDER_REFLECT101, alpha_affine=5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        A.ShiftScaleRotate(p=0.5, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_REFLECT101),
        ], p=0.7),
    # Dropout & Shuffle
    # A.OneOf(
    #         [
    #             A.RandomGridShuffle(p=0.5),
    #             A.CoarseDropout(p=0.5),
    #         ], p=0.7),
        
    # Add occasion blur
    A.OneOf([A.GaussianBlur(p=0.5), A.GaussNoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.5)], p=0.7),

    A.Normalize(mean=(0.485, 0.456, 0.406, 0.45), std=(0.229, 0.224, 0.225, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    # A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406, 0.45), std=(0.229, 0.224, 0.225, 0.225)),
    ToTensorV2(),
])