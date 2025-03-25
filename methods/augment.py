# augment.py


class DummyAugment:
    def augment(self, video_tensor, bboxes):
        return video_tensor, bboxes


# TODO: ADD AUGMENTATION CLASSES HERE