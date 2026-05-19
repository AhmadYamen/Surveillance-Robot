import numpy as np
import dataclasses
import time

from typing import List, Tuple


@dataclasses.dataclass
class FaceProfile:
    """
        DataClass to represent each face as a structural object
    """
    face_crop: np.ndarray
    face_dimensions: Tuple[int, int, int, int]
    face_embeddings: np.ndarray
    face_id: int = None
    face_name: str = None
    last_frame: np.ndarray = None
    last_faces: list = None
    first_seen: float = None
    last_seen: float = None
    times_seen: int = 1
    is_target: bool = False
    track_history: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = time.time()
        if self.last_seen is None:
            self.last_seen = time.time()
        if self.track_history is None:
            self.track_history = []
        if len(self.track_history) > 30:
            self.track_history = self.track_history[-30:]
    
    def update_seen(self):
        self.last_seen = time.time()
        self.times_seen += 1
    
    def update_position(self, center_x: int, center_y: int):
        self.track_history.append((center_x, center_y))
        if len(self.track_history) > 30:
            self.track_history = self.track_history[-30:]
    
    @DeprecationWarning
    def get_smoothed_position(self) -> Tuple[int, int]:
        if not self.track_history:
            return self.center_x, self.center_y
        
        weights = np.exp(np.linspace(-1, 0, len(self.track_history)))
        weights = weights / weights.sum()
        
        smooth_x = int(sum(pos[0] * w for pos, w in zip(self.track_history, weights)))
        smooth_y = int(sum(pos[1] * w for pos, w in zip(self.track_history, weights)))
        
        return smooth_x, smooth_y

    @property
    def center_x(self) -> int:
        return (self.face_dimensions[0] + self.face_dimensions[2]) // 2

    @property
    def center_y(self) -> int:
        return (self.face_dimensions[1] + self.face_dimensions[3]) // 2
    
    @property
    def size(self) -> int:
        return self.face_dimensions[2] * self.face_dimensions[3]

    @property
    def age_existed(self) -> float:
        return time.time() - self.last_seen