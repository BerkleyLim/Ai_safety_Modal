"""
데이터 증강 모듈
이미지와 bbox를 함께 증강합니다.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import List, Tuple, Dict
import random


class ImageAugmenter:
    """이미지 및 bbox 증강기"""

    def __init__(
        self,
        rotation_range: int = 15,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        blur_probability: float = 0.1,
    ):
        """
        Args:
            rotation_range: 회전 각도 범위 (±도)
            brightness_range: 밝기 조절 범위 (배율)
            contrast_range: 대비 조절 범위 (배율)
            flip_horizontal: 좌우 반전 여부
            flip_vertical: 상하 반전 여부
            blur_probability: 블러 적용 확률
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.blur_probability = blur_probability

    def rotate_image_and_bbox(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        angle: float
    ) -> Tuple[Image.Image, List[List[float]]]:
        """
        이미지와 bbox를 함께 회전시킵니다.

        Args:
            image: PIL Image
            bboxes: bbox 리스트 [[x, y, w, h], ...]
            angle: 회전 각도 (도)

        Returns:
            (회전된 이미지, 회전된 bbox 리스트)
        """
        # 이미지 회전
        rotated_img = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))

        # bbox 회전 (중심점 기준)
        width, height = image.size
        center_x, center_y = width / 2, height / 2
        angle_rad = np.radians(angle)

        rotated_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox

            # bbox의 4개 코너 계산
            corners = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ]

            # 각 코너를 회전
            rotated_corners = []
            for px, py in corners:
                # 중심점으로 이동
                px -= center_x
                py -= center_y

                # 회전
                new_x = px * np.cos(angle_rad) - py * np.sin(angle_rad)
                new_y = px * np.sin(angle_rad) + py * np.cos(angle_rad)

                # 다시 원래 위치로
                new_x += center_x
                new_y += center_y

                rotated_corners.append((new_x, new_y))

            # 회전된 코너에서 새 bbox 계산
            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]

            new_x = max(0, min(xs))
            new_y = max(0, min(ys))
            new_w = min(width, max(xs)) - new_x
            new_h = min(height, max(ys)) - new_y

            # 유효한 bbox만 추가
            if new_w > 0 and new_h > 0:
                rotated_bboxes.append([new_x, new_y, new_w, new_h])

        return rotated_img, rotated_bboxes

    def flip_image_and_bbox(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        horizontal: bool = True
    ) -> Tuple[Image.Image, List[List[float]]]:
        """
        이미지와 bbox를 반전시킵니다.

        Args:
            image: PIL Image
            bboxes: bbox 리스트 [[x, y, w, h], ...]
            horizontal: True면 좌우 반전, False면 상하 반전

        Returns:
            (반전된 이미지, 반전된 bbox 리스트)
        """
        width, height = image.size

        if horizontal:
            flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                new_x = width - (x + w)
                flipped_bboxes.append([new_x, y, w, h])
        else:
            flipped_img = image.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                new_y = height - (y + h)
                flipped_bboxes.append([x, new_y, w, h])

        return flipped_img, flipped_bboxes

    def adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """밝기 조절"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """대비 조절"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def add_blur(self, image: Image.Image) -> Image.Image:
        """블러 추가"""
        return image.filter(ImageFilter.GaussianBlur(radius=1))

    def augment(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        augment_all: bool = False
    ) -> List[Tuple[Image.Image, List[List[float]], str]]:
        """
        이미지와 bbox를 증강합니다.

        Args:
            image: 원본 이미지
            bboxes: 원본 bbox 리스트
            augment_all: True면 모든 증강 적용, False면 랜덤 선택

        Returns:
            [(증강된 이미지, 증강된 bbox, 증강 방법), ...] 리스트
        """
        augmented_data = []

        # 원본 추가
        augmented_data.append((image.copy(), bboxes.copy(), "original"))

        if augment_all or random.random() < 0.5:
            # 회전
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            rotated_img, rotated_bboxes = self.rotate_image_and_bbox(
                image, bboxes, angle
            )
            augmented_data.append((
                rotated_img,
                rotated_bboxes,
                f"rotate_{angle:.1f}"
            ))

        if augment_all or (self.flip_horizontal and random.random() < 0.5):
            # 좌우 반전
            flipped_img, flipped_bboxes = self.flip_image_and_bbox(
                image, bboxes, horizontal=True
            )
            augmented_data.append((
                flipped_img,
                flipped_bboxes,
                "flip_horizontal"
            ))

        if augment_all or (self.flip_vertical and random.random() < 0.5):
            # 상하 반전
            flipped_img, flipped_bboxes = self.flip_image_and_bbox(
                image, bboxes, horizontal=False
            )
            augmented_data.append((
                flipped_img,
                flipped_bboxes,
                "flip_vertical"
            ))

        # 밝기 조절 (bbox는 변경 없음)
        if augment_all or random.random() < 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            bright_img = self.adjust_brightness(image, brightness_factor)
            augmented_data.append((
                bright_img,
                bboxes.copy(),
                f"brightness_{brightness_factor:.2f}"
            ))

        # 대비 조절 (bbox는 변경 없음)
        if augment_all or random.random() < 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            contrast_img = self.adjust_contrast(image, contrast_factor)
            augmented_data.append((
                contrast_img,
                bboxes.copy(),
                f"contrast_{contrast_factor:.2f}"
            ))

        # 블러 (bbox는 변경 없음)
        if random.random() < self.blur_probability:
            blurred_img = self.add_blur(image)
            augmented_data.append((
                blurred_img,
                bboxes.copy(),
                "blur"
            ))

        return augmented_data


def augment_image(
    image: Image.Image,
    bboxes: List[List[float]],
    **kwargs
) -> List[Tuple[Image.Image, List[List[float]], str]]:
    """
    간편 증강 함수 (함수형 인터페이스)

    Args:
        image: 원본 이미지
        bboxes: 원본 bbox 리스트
        **kwargs: ImageAugmenter 초기화 파라미터

    Returns:
        [(증강된 이미지, 증강된 bbox, 증강 방법), ...] 리스트
    """
    augmenter = ImageAugmenter(**kwargs)
    return augmenter.augment(image, bboxes)


if __name__ == "__main__":
    # 테스트 코드
    print("데이터 증강기 테스트")
    print("실제 데이터 다운로드 후 테스트하세요.")

    # 간단한 테스트
    test_img = Image.new('RGB', (640, 480), color='red')
    test_bboxes = [[100, 100, 200, 150]]

    augmenter = ImageAugmenter()
    results = augmenter.augment(test_img, test_bboxes, augment_all=False)

    print(f"생성된 증강 이미지 수: {len(results)}")
    for img, bboxes, method in results:
        print(f"  - {method}: {len(bboxes)} objects")