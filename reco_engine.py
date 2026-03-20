import time
import numpy as np
import cv2 as cv
import dataclasses
import os
import json
import threading
import embedding_engine as eng
import yaml

from typing import List, Optional, Tuple
from scipy.spatial.distance import cosine

class EngineConfiguration:
    """
        Engine Configuration is a class used to load and store the important configuration for the Recognition Engine
    """
    def __init__(self):
        self.embeddings: List[np.ndarray] = [] ## Embeddings load from the dataset
        self.labels: List[str] = [] ## Labels load from the dataset
        self.DEFAULT_CAMERA_INDEX = 1 ## Default index for the attached Camera
        self._default_settings = False ## Flag if there is no Remote Camera
        self.unknown_dir = 'Unknown' ## Directory of the Unknown faces
        
        self._parse_config() ## Load stored data, and handle the absense of data and files

    def _parse_config(self):
        """
            Load data from dataset, ensure consistency
        """
        data = None
        try:
            if os.path.exists('config.yaml'):
                with open('config.yaml', 'r') as file:
                    data = yaml.safe_load(file)
            else:
                print(f'File config.yaml not found')
        except yaml.YAMLError as e:
            print(f'Error parsing YAML: {e}')
        
        if data:
            if 'remote_camera' in data:
                remote_camera_attrs = data['remote_camera']
                remote_camera_domain = remote_camera_attrs['domain']

                PROTOCOL = remote_camera_domain['PROTOCOL']
                IP = f"{remote_camera_domain['IP']['SEG_A']}.{remote_camera_domain['IP']['SEG_B']}.{remote_camera_domain['IP']['SEG_C']}"
                PORT = remote_camera_domain['PORT']
                stream_path = remote_camera_attrs['stream']['PATH']
                
                self.streaming_url = f'{PROTOCOL}://{IP}:{PORT}/{stream_path}'
            else:
                self._default_settings = True
                self.streaming_url = self.DEFAULT_CAMERA_INDEX

            if 'camera' in data:
                camera_attrs = data['camera']
                self.CAM_WIDTH = camera_attrs.get('CAM_WIDTH', 720)
                self.CAM_HEIGHT = camera_attrs.get('CAM_HEIGHT', 480)
                self.CAM_ORIENTATION = camera_attrs.get('CAM_ORIENTATION', -1)
                self.CAM_INDEX = camera_attrs.get('CAM_INDEX', self.DEFAULT_CAMERA_INDEX)
                self._default_settings = False
            else:
                self._default_settings = True
        else:
            print('No Remote Camera Was Found, using local camera')
            self._default_settings = True

        if self._default_settings:
            self.streaming_url = self.DEFAULT_CAMERA_INDEX
            self.CAM_INDEX = self.DEFAULT_CAMERA_INDEX
            self.CAM_WIDTH = 720
            self.CAM_HEIGHT = 480
            self.CAM_ORIENTATION = -1

@dataclasses.dataclass
class FaceProfile:
    """
        DataClass to represent each face as a structural object
    """
    face_crop: np.ndarray ## Face Cropped from an image
    face_dimensions: Tuple[int, int, int, int] ## Face dimensions as (x, y, w, h), Bounding Box
    face_embeddings: np.ndarray ## Embeddings exctract from a face
    last_frame: np.ndarray = None ## Last captured frame that contains the face
    last_faces: list = None ## Last captured faces that this face was included
    face_name: str = None ## Face label if it is recognized
    first_seen: float = None ## First time the face is seen
    last_seen: float = None ## Last time the face was seen
    times_seen: int = 1 ## Number of times the face was seen
    is_target: bool = False ## Flag indicating if the face is the current target

    def __post_init__(self):
        """
            Initialize first and last seen
        """
        if self.first_seen is None:
            self.first_seen = time.time()
        if self.last_seen is None:
            self.last_seen = time.time()
    
    def update_seen(self):
        """
            Update last seen each time the face was encountered
        """
        self.last_seen = time.time()
        self.times_seen += 1

    @property
    def center_x(self) -> int:
        """
            Return the face's x center as a property
        """
        return (self.face_dimensions[0] + self.face_dimensions[2]) // 2

    @property
    def center_y(self) -> int:
        """
            Return the face's y center as a property
        """
        return (self.face_dimensions[1] + self.face_dimensions[3]) // 2
    
    @property
    def size(self) -> int:
        """
            Return the face's size as a property
        """
        return self.face_dimensions[2] * self.face_dimensions[3]

    @property
    def age_existed(self) -> float:
        """
            Return the face's age since first seen
        """
        return time.time() - self.last_seen
    
class Engine(EngineConfiguration):
    """
        Engine Class inherit from Engine Configuration Class for data managing
    """
    def __init__(self):
        # Thread Safety locks
        self.frame_lock = threading.Lock()
        self.profiles_lock = threading.Lock()
        self.name_cache_lock = threading.Lock()

        # Shared common data
        self.latest_frame = None
        self.latest_faces = []

        # Profiles Data
        self.known_profiles: List[FaceProfile] = []
        self.unknown_profiles: List[FaceProfile] = []
        self.target_profile: Optional[FaceProfile] = None
        
        # Profile Cleanup settings
        self.max_profile_age = 30.0
        self.min_confidence = 0.5
        self.target_lost_threshold = 3.0

        # Control Flags
        self.running = False
        self.face_engine = eng.FaceExtractorEngine()

        self.embeddings = []
        self.labels = []
        
        # Initialize the Engine
        super().__init__()
        self._load_configuration()
        self._ensure_directories()
        self.face_labels = {} # Key: bbox, Value: Name

        print(f"Loaded {len(self.embeddings)} known faces")

    def _ensure_directories(self):
        """
            Ensure directories exist
        """
        if not os.path.exists(self.unknown_dir):
            os.makedirs(self.unknown_dir)

    def _load_configuration(self):
        """
            Load data stored in the dataset
        """
        if os.path.exists('dataset.json'):
            try:
                with open('dataset.json', 'r') as dataset_file:
                    data = json.load(dataset_file)
                    if 'Embedding' in data and 'Label' in data:
                        for embed_list, label in zip(data['Embedding'], data['Label']):
                            try:
                                embed_array = np.array(embed_list, dtype=np.float32)
                                if embed_array.size > 0:
                                    self.embeddings.append(embed_array)
                                    self.labels.append(label)
                            except Exception as e:
                                print(f"Error Parsing embeddings: {e}")
            except Exception as e:
                print(f"Error loading dataset.json: {e}")

    def _save_unknown_face(self, face_profile: FaceProfile):
        """
            Save unknown faces to a directory stamped by the timestamp
        """
        timestamp = int(time.time())
        file_path = os.path.join(self.unknown_dir, f'unknown_{timestamp}_{face_profile.center_x}.jpg')
        cv.imwrite(file_path, face_profile.face_crop)

    def _cleanup_old_profiles(self):
        """
            Periodic Cleaning to prevent data race and duplicates
        """
        current_time = time.time()
        with self.profiles_lock:
            self.unknown_profiles = [p for p in self.unknown_profiles if current_time - p.last_seen < self.max_profile_age]
            self.known_profiles = [p for p in self.known_profiles if current_time - p.last_seen < self.max_profile_age * 2]
            
            if self.target_profile:
                if current_time - self.target_profile.last_seen > self.target_lost_threshold:
                    self.target_profile = None
        
        # Clean up the display cache periodically
        with self.name_cache_lock:
            if len(self.face_labels) > 50:
                self.face_labels.clear()

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
            Find the best match face from the dataset, and return the face's label and the similarity confidence
        """
        if not self.embeddings:
            return "Unknown", 0.0

        best_sim = 0
        best_label = "Unknown"

        for idx, stored_embeddings in enumerate(self.embeddings):
            similarity = 1 - cosine(embedding, stored_embeddings)
            if similarity > best_sim:
                best_sim = similarity
                if similarity > self.min_confidence:
                    best_label = self.labels[idx]
        return best_label, best_sim
    
    def _find_existing_profile(self, embedding: np.ndarray, profiles: List[FaceProfile]) -> Optional[FaceProfile]:
        """
            Find if the current profile already exist in the dataset for the unknown faces, EXTRA CHECK
        """
        for profile in profiles:
            sim = 1 - cosine(embedding, profile.face_embeddings)
            if sim > self.min_confidence:
                return profile
        return None
    
    def _recognize(self):
        """
            Recognition Gate
        """
        last_cleanup = time.time()
        last_label_cleanup = time.time()

        while self.running:
            try:
                current_time = time.time()

                ## Every a period of time do a cleaning for the old profiles
                if current_time - last_cleanup > 1.0:
                    self._cleanup_old_profiles()
                    last_cleanup = current_time
                
                ## Clean up old labels every 0.5 seconds to prevent label persistence
                ## This ensures that when a known face leaves the frame, its label disappears
                if current_time - last_label_cleanup > 0.5:
                    with self.name_cache_lock:
                        ## Remove all labels only use current frame's labels
                        self.face_labels = {}
                    last_label_cleanup = current_time
                
                ## Capture the current frame and faces 
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.05)
                        continue
                    current_frame = self.latest_frame.copy()
                    current_faces = self.latest_faces.copy()

                ## Create temporary storage for new labels from this frame only
                new_labels = {}
                
                ## If faces were detected
                if len(current_faces) > 0:
                    results = self.face_engine.getFaceEmbeddings(current_frame, current_faces) ## Get embeddings for the detected faces (Unknown Yet)

                    ## If embeddings were returned
                    if results:
                        ## For each face detected and embedded
                        for face in results:
                            u_embed = face['embedding'] ## Embeddings
                            bbox = tuple(face['bbox']) ## Bounding Box
                            u_norm = u_embed / (np.linalg.norm(u_embed) + 1e-7) ## Normalize embeddings for better computation
                            
                            name, confidence = self._find_best_match(u_norm) ## Check if the face is identical with any of the stored faces
                            final_name = name 
                            x, y, w, h = bbox

                            ## If a face within these boundaries
                            if w > 10 and h > 10:
                                ## Crop the face
                                face_crop = current_frame[y:y+h, x:x+w]

                                ## If the crop is big enough
                                if face_crop.size > 0:
                                    with self.profiles_lock:
                                        ## If the face was recognized
                                        if name != "Unknown":
                                            found = False
                                            ## If the face's profile already in the list of known faces' profiles, then just update its last seen
                                            for profile in self.known_profiles:
                                                if profile.face_name == name:
                                                    profile.update_seen()
                                                    profile.face_dimensions = bbox  ## Update bounding box for tracking
                                                    final_name = profile.face_name
                                                    found = True
                                                    break
                                            ## If the face's profile was not found, then re-compare it with the existing known profiles
                                            if not found:
                                                existing = self._find_existing_profile(u_norm, self.known_profiles)
                                                ## If it exists, then update its last seen
                                                if existing:
                                                    existing.update_seen()
                                                    existing.face_dimensions = bbox  ## Update bounding box for tracking
                                                    final_name = existing.face_name
                                                ## If it was not found, then just create a new profile and add it to the list
                                                else:
                                                    profile = FaceProfile(face_crop, bbox, u_norm, face_name=name)
                                                    self.known_profiles.append(profile)
                                                    final_name = name

                                        ## If the face was not recognized
                                        else:
                                            found_known = False
                                            ## If this unknown face was recognized to be a known face(Profile Comparison), 
                                            ## then update its last seen, (FOR FOCUS LOST)
                                            for known in self.known_profiles:
                                                sim = 1 - cosine(u_norm, known.face_embeddings)
                                                if sim > self.min_confidence:
                                                    final_name = known.face_name
                                                    known.update_seen()
                                                    known.face_dimensions = bbox  ## Update bounding box for tracking
                                                    found_known = True
                                                    break

                                            ## If the face is still unknown, check if it already in the list of Unknown Profiles
                                            if not found_known:
                                                existing = self._find_existing_profile(u_norm, self.unknown_profiles)
                                                ## If it is already in the list, then just update its last seen
                                                if existing:
                                                    existing.update_seen()
                                                    existing.face_dimensions = bbox  ## Update bounding box for tracking
                                                ## If it has been recognized at all, then create a new profile for it and add it to the list of Unknown Profiles
                                                else:
                                                    profile = FaceProfile(face_crop, bbox, u_norm)
                                                    self.unknown_profiles.append(profile)
                                                    self._save_unknown_face(profile)
                                        
                                        ## Store the label for this face with current timestamp
                                        ## Only store labels for faces that are actually in this frame
                                        new_labels[bbox] = (final_name, current_time)
                
                ## Update the main labels dictionary with ONLY labels from this frame
                ## This completely replaces old labels, preventing persistence issues
                with self.name_cache_lock:
                    self.face_labels = {bbox: name for bbox, (name, _) in new_labels.items()}
                
                time.sleep(0.03) ## Sleep a little bit for CPU Overloading
            except Exception as e:
                print(f"Recognition Error: {e}")

    def _draw_display(self):
        """
            Display the captured frames to the screen
                Draw Rectangles around the Recognized Faces with a Green Color, along with their Labels
                Draw Rectangles around the Un-Recognized Faces with a Red Color, along with 'Unknown' Label
            
        """
        while self.running:
            try:
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.03)
                        continue
                    frame = self.latest_frame.copy()
                    current_faces = self.latest_faces.copy()
                
                ## Get current labels, these are only from the most recent recognition pass
                ## Since it clears labels in _recognize, this only contains labels for faces
                ## that were actually detected in the current frame
                with self.name_cache_lock:
                    current_labels = self.face_labels.copy()
                
                ## Create a mapping of faces to their labels for this frame only
                face_to_label = {}
                
                ## First, match current faces with labels based on position
                ## This ensures each face gets the correct label from the same frame
                for bbox in current_faces:
                    x, y, w, h = bbox
                    cx, cy = x + w//2, y + h//2
                    
                    best_match = "Unknown"
                    min_dist = float('inf')
                    
                    ## Find the closest label from the current frame
                    for label_bbox, label_name in current_labels.items():
                        lx, ly, lw, lh = label_bbox
                        lcx, lcy = lx + lw//2, ly + lh//2
                        
                        ## Calculate distance between face centers
                        dist = np.sqrt((cx - lcx)**2 + (cy - lcy)**2)
                        
                        ## If very close (within 30 pixels), it's the same face
                        if dist < 30 and dist < min_dist:
                            min_dist = dist
                            best_match = label_name
                    
                    ## Assign the best matching label to this face
                    face_to_label[bbox] = best_match
                
                ## Draw each face with its assigned label
                for bbox, name in face_to_label.items():
                    x, y, w, h = bbox
                    
                    ## Green for recognized faces, Red for unknown
                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                    display_name = name
                    
                    ## Check if this face is the current target
                    with self.profiles_lock:
                        if self.target_profile:
                            tx, ty, tw, th = self.target_profile.face_dimensions
                            ## If bounding boxes are close, this is the target face
                            if abs(x - tx) < 20 and abs(y - ty) < 20:
                                color = (255, 0, 255)  ## Magenta for target
                                display_name = f"Target: {name}"
                    
                    ## Draw rectangle around the face
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    ## Draw label with black background for better visibility
                    (text_width, text_height), _ = cv.getTextSize(display_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv.rectangle(frame, 
                               (x, y - text_height - 10), 
                               (x + text_width + 10, y - 5), 
                               (0, 0, 0), -1)
                    cv.putText(frame, display_name, 
                              (x + 5, y - 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                ## Display statistics
                with self.profiles_lock:
                    stats = f"Known: {len(self.known_profiles)} | Unknown: {len(self.unknown_profiles)}"
                cv.putText(frame, stats, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                ## Show the frame
                cv.imshow('Face Recognition', frame)
                
                ## Check for quit command
                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                            
            except Exception as e:
                print(f"Display Error: {e}")
                time.sleep(0.1)

    def _run_camera(self, flag='s'):
        """
            Run the Camera, whether from a Remote Camera using an IP or an attached Camera
                Captured Frames
        """
        self.cap = None 
        try:
            url = self.streaming_url if flag == 's' else self.CAM_INDEX
            self.cap = cv.VideoCapture(url)
            
            if not self.cap or not self.cap.isOpened():
                self.running = False
                return

            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                processed_frame = cv.flip(frame, self.CAM_ORIENTATION)
                faces = self.face_engine.detectFaces(processed_frame)
                
                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_faces = faces
                
                time.sleep(0.01)
        finally:
            if self.cap: self.cap.release()

    def main(self, **kwargs):
        """
            The main thread for the Recognition System
        """
        self.running = True ## Flag for keeping the system running
        c_flag = kwargs.get('c_flag', 's') ## Initializing parameters for the self._run_camera
        
        ## Each of the following Methods has its own thread to work in, to prevent intense computational power on the main thread
        threads = [
            threading.Thread(target=self._run_camera, args=(c_flag,), name="Camera"),
            threading.Thread(target=self._draw_display, name="Display"),
            threading.Thread(target=self._recognize, name="Recognize")
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.running = False
        
        cv.destroyAllWindows()

if __name__ == '__main__':
    Engine().main(c_flag='l')