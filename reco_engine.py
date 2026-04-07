import time
import numpy as np
import cv2 as cv
import dataclasses
import os
import json
import threading
import struct

import yaml
import socket as stc

from collections import deque
from typing import List, Optional, Tuple
from scipy.spatial.distance import cosine
from zeroconf import Zeroconf, ServiceInfo

# Import your face embedding engine
try:
    import embedding_engine as eng
except ImportError:
    print("Warning: embedding_engine not found, face recognition disabled")
    eng = None

class SocketInterface:
    def __init__(self, service_name, service_type, domain,port):
        self.service_name = service_name
        self.service_type = service_type
        self.domain = domain
        self.full_path = self.service_name + '.' + self.service_type + self.domain
        self.port = port
        self.host_ip = None

        self._get_local_ip()
    
    def _get_local_ip(self):
        broad_cast_socket = stc.socket(stc.AF_INET, stc.SOCK_DGRAM)
        ip = '127.0.0.1'
        try:
            broad_cast_socket.connect(('8.8.8.8', 80))
            ip = broad_cast_socket.getsockname()[0]
        except Exception:
            pass
        finally:
            self.host_ip = ip
            broad_cast_socket.close()

    def advertise_mdns(self):
        zeroconf = Zeroconf()

        # Create Service Info, and register it
        service_info = ServiceInfo(self.service_type + self.domain, self.full_path, addresses = [self.host_ip], port = self.port)
        zeroconf.register_service(service_info)
        try:
            # Keep Advertising until Interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            zeroconf.unregister_service(service_info)
            zeroconf.close()

    def run_server(self):
        self.server_socket = stc.socket(stc.AF_INET, stc.SOCK_STREAM)
        self.server_socket.bind((self.host_ip, self.port))
        self.server_socket.listen(5)
        print(F"Server Listening on port: {self.port}")

        while True:
            client_sock, addr = self.server_socket.accept()
            print(F"Connection from {addr}")
            client_sock.send(b"ACK")
       
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
                mDNS = remote_camera_domain["mDNS"]
                IP = f"{remote_camera_domain['IP']['SEG_A']}.{remote_camera_domain['IP']['SEG_B']}.{remote_camera_domain['IP']['SEG_C']}"
                PORT = remote_camera_domain['PORT']
                stream_path = remote_camera_attrs['stream']['PATH']
                
                try:
                    if mDNS:
                        IP = stc.gethostbyname(mDNS)
                except:
                    pass

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
    face_id: int = None ## Unique ID for the face
    face_name: str = None ## Face label if it is recognized
    last_frame: np.ndarray = None ## Last captured frame that contains the face
    last_faces: list = None ## Last captured faces that this face was included
    first_seen: float = None ## First time the face is seen
    last_seen: float = None ## Last time the face was seen
    times_seen: int = 1 ## Number of times the face was seen
    is_target: bool = False ## Flag indicating if the face is the current target
    track_history: List[Tuple[int, int]] = None ## Track history of center positions

    def __post_init__(self):
        """
            Initialize first and last seen
        """
        if self.first_seen is None:
            self.first_seen = time.time()
        if self.last_seen is None:
            self.last_seen = time.time()
        if self.track_history is None:
            self.track_history = []
        # Keep only last 30 positions for smoothing
        if len(self.track_history) > 30:
            self.track_history = self.track_history[-30:]
    
    def update_seen(self):
        """
            Update last seen each time the face was encountered
        """
        self.last_seen = time.time()
        self.times_seen += 1
    
    def update_position(self, center_x: int, center_y: int):
        """
            Update tracking history
        """
        self.track_history.append((center_x, center_y))
        if len(self.track_history) > 30:
            self.track_history = self.track_history[-30:]
    
    def get_smoothed_position(self) -> Tuple[int, int]:
        """
            Get smoothed position from history
        """
        if not self.track_history:
            return self.center_x, self.center_y
        
        # Weighted average, more recent positions have higher weight
        weights = np.exp(np.linspace(-1, 0, len(self.track_history)))
        weights = weights / weights.sum()
        
        smooth_x = int(sum(pos[0] * w for pos, w in zip(self.track_history, weights)))
        smooth_y = int(sum(pos[1] * w for pos, w in zip(self.track_history, weights)))
        
        return smooth_x, smooth_y

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
        self.un_profiles_lock = threading.Lock()

        # Shared common data, use a single tuple to ensure consistency
        self.frame_data = None  # Will store (frame, faces) as a tuple
        
        # Profiles Data
        self.known_profiles: List[FaceProfile] = []
        self.unknown_profiles: List[FaceProfile] = []
        self.target_profile: Optional[FaceProfile] = None
        
        # ID counters
        self.next_known_id = 0
        self.next_unknown_id = 0
        
        # Profile Cleanup settings
        self.max_profile_age = 30.0
        self.min_confidence = 0.5
        self.target_lost_threshold = 3.0

        # Control Flags
        self.running = False
        self.face_engine = eng.FaceExtractorEngine()

        self.embeddings = []
        self.labels = []
        
        # Movement smoothing variables
        self.movement_smoothing = 0.3
        self.last_smoothed_offset = 0
        self.last_movement_time = 0
        
        # PID controller for smoother tracking
        self.pid = {
            'kp': 0.5,
            'ki': 0.1,
            'kd': 0.05,
            'integral': 0,
            'last_error': 0,
            'last_time': time.time()
        }

        # Tracking parameters
        self.centering_dead_zone = 20  # Pixels tolerance for centering
        self.OPTIMAL_FACE_SIZE = 15000  # Optimal face area in pixels (w * h)
        self.MIN_FACE_SIZE = 5000  # Minimum face size to track
        self.MAX_FACE_SIZE = 30000  # Maximum face size
        self.FORWARD_DEAD_ZONE = 2000  # Size tolerance for forward movement (pixels area)
        
        # Initialize the Engine
        super().__init__()
        self._load_configuration()
        self._ensure_directories()
        self.face_labels = {}

        print(f"Loaded {len(self.embeddings)} known faces")

    def _estimate_distance(self, face_profile: FaceProfile) -> float:
        """
            Estimate distance to face based on face size and camera parameters.
            Returns estimated distance in arbitrary units
            The higher the value, the closer the face is
            Uses the formula: 
                distance = (known_face_width * focal_length) / face_width_in_pixels
        """
        # Camera parameters
        KNOWN_FACE_WIDTH_CM = 15.0  # Average face width in cm ADJUST
        FOCAL_LENGTH = 500.0  # Approximate focal length in pixels ADJUST
        
        _, _, w, h = face_profile.face_dimensions
        face_width_px = w
        
        if face_width_px <= 0:
            return 1000.0  # Very far
        
        # Calculate distance using similar triangles
        distance_cm = (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / face_width_px
        
        # Normalize to 0-1 scale 
        OPTIMAL_DISTANCE_CM = 100.0
        normalized_distance = min(1.0, OPTIMAL_DISTANCE_CM / distance_cm)
        
        return normalized_distance
    
    def _ensure_directories(self):
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
        file_path = os.path.join(self.unknown_dir, f'unknown_{timestamp}_{face_profile.face_id}.jpg')
        cv.imwrite(file_path, face_profile.face_crop)

    def _send_to_esp32(self, command: str, intensity: float = 0.0):
        """
            Send command to ESP32-CAM
            Commands:
                'FORWARD': Move forward with intensity
                'BACKWARD': Move backward with intensity
                'LEFT': Turn left with intensity
                'RIGHT': Turn right with intensity
                'STOP': Stop all movement
                'TRACK': Enable/disable tracking mode
        """
        # Convert intensity to 0-100 range
        intensity_val = int(max(0, min(100, intensity * 100)))
        
        # Format command
        cmd_map = {
            'FORWARD': 'F',
            'BACKWARD': 'B',
            'LEFT': 'L',
            'RIGHT': 'R',
            'STOP': 'S',
            'TRACK': 'T'
        }
        
        cmd_char = cmd_map.get(command, 'S')
        cmd_str = f"{cmd_char}{intensity_val:03d}"
        
        returned_state = self.esp32_server_socket.send_command(command)
        if returned_state:
            print(f"[ESP32] Command: {cmd_str} ({command} {intensity:.2f})")
            self.last_command_time = time.time()
        
        return True

    def _cleanup_old_profiles(self):
        """
            Periodic Cleaning to prevent data race and duplicates
        """
        current_time = time.time()
        with self.profiles_lock:
            with self.un_profiles_lock:
                self.unknown_profiles = [p for p in self.unknown_profiles if current_time - p.last_seen < self.max_profile_age]
            self.known_profiles = [p for p in self.known_profiles if current_time - p.last_seen < self.max_profile_age * 2]
            
            self._update_target_profile()
            
            if self.target_profile:
                if current_time - self.target_profile.last_seen > self.target_lost_threshold:
                    print(f"Target {self.target_profile.face_id} lost!")
                    self.target_profile = None
        
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
    
    def _update_target_profile(self):
        """
            Update target profile in case the previous target was lost
        """
        with self.un_profiles_lock:
            if self.target_profile:
                if self.target_profile not in self.unknown_profiles:
                    if self.target_profile.is_target:
                        self.target_profile.is_target = False
                    if self.unknown_profiles:
                        self.target_profile = self._find_largest_face(self.unknown_profiles)
                        print(f"Target switched to face ID: {self.target_profile.face_id}")
                    else:
                        self.target_profile = None
            elif self.unknown_profiles:
                self.target_profile = self._find_largest_face(self.unknown_profiles)
                print(f"New target selected - ID: {self.target_profile.face_id}")
    
    def _calculate_forward_intensity(self, face_area: int, optimal_area: int) -> float:
        """
            Calculate smooth forward/backward movement intensity based on face size.
            Returns intensity from 0.0 to 1.0
        """
        # Calculate error percentage
        if face_area < optimal_area:
            # Too far, have to move forward
            error_ratio = (optimal_area - face_area) / optimal_area
            intensity = min(1.0, error_ratio * 1.5)  # Multiply by 1.5 for more responsive movement
            intensity = max(0.2, intensity)  # Minimum 20% speed to prevent crawling
        else:
            # Too close, have to move backward
            error_ratio = (face_area - optimal_area) / optimal_area
            intensity = min(1.0, error_ratio * 1.5)
            intensity = max(0.2, intensity)
        
        return intensity
    
    def _calculate_pid_movement(self, error):
        """
            Calculate PID Control value for the robot movement
        """
        current_time = time.time()
        dt = current_time - self.pid['last_time']
        
        if dt <= 0:
            dt = 0.01
        
        p_term = self.pid['kp'] * error
        self.pid['integral'] += error * dt
        # Anti-windup clamp
        self.pid['integral'] = max(-100, min(100, self.pid['integral']))
        i_term = self.pid['ki'] * self.pid['integral']
        
        derivative = (error - self.pid['last_error']) / dt
        d_term = self.pid['kd'] * derivative
        
        output = p_term + i_term + d_term
        
        self.pid['last_error'] = error
        self.pid['last_time'] = current_time
        
        return max(-100, min(100, output))
    
    def _find_largest_face(self, un_profiles: List[FaceProfile]):
        """
            Find the largest face, or the closet face to the robot
        """
        if not un_profiles:
            return None
        
        largest = un_profiles[0]
        for un in un_profiles[1:]:
            if un.size > largest.size:
                if largest.is_target:
                    largest.is_target = False
                largest = un
        
        if not largest.is_target:
            largest.is_target = True
        return largest
    
    def _recognize(self):
        """
            Recognition Gate
        """
        last_cleanup = time.time()
        last_label_cleanup = time.time()

        while self.running:
            try:
                current_time = time.time()

                if current_time - last_cleanup > 1.0:
                    self._cleanup_old_profiles()
                    last_cleanup = current_time
                
                if current_time - last_label_cleanup > 0.5:
                    with self.name_cache_lock:
                        self.face_labels = {}
                    last_label_cleanup = current_time
                
                # Get frame and faces
                with self.frame_lock:
                    if self.frame_data is None:
                        time.sleep(0.05)
                        continue
                    current_frame, current_faces = self.frame_data
                    if current_frame is None:
                        time.sleep(0.05)
                        continue
                    # Create copies
                    current_frame = current_frame.copy()
                    current_faces = [tuple(f) for f in current_faces] if current_faces else []

                new_labels = {}
                
                if len(current_faces) > 0 and current_frame is not None:
                    # Verify frame is valid
                    if current_frame.size == 0:
                        time.sleep(0.05)
                        continue
                    
                    h_frame, w_frame = current_frame.shape[:2]
                    
                    try:
                        results = self.face_engine.getFaceEmbeddings(current_frame, current_faces)
                    except Exception as e:
                        print(f"Error in getFaceEmbeddings: {e}")
                        time.sleep(0.05)
                        continue

                    if results:
                        for face in results:
                            try:
                                u_embed = face['embedding']
                                bbox = face['bbox']
                                
                                # Ensure bbox has 4 elements
                                if len(bbox) != 4:
                                    continue
                                    
                                x, y, w, h = bbox
                                
                                # Clamp coordinates to frame boundaries
                                x = max(0, min(int(x), w_frame - 1))
                                y = max(0, min(int(y), h_frame - 1))
                                w = min(int(w), w_frame - x)
                                h = min(int(h), h_frame - y)
                                
                                if w <= 0 or h <= 0:
                                    continue
                                
                                u_norm = u_embed / (np.linalg.norm(u_embed) + 1e-7)
                                name, confidence = self._find_best_match(u_norm)
                                final_name = name

                                if w > 10 and h > 10:
                                    try:
                                        face_crop = current_frame[y:y+h, x:x+w].copy()
                                    except Exception as crop_error:
                                        print(f"Crop error: {crop_error}")
                                        continue

                                    if face_crop.size > 0:
                                        with self.profiles_lock:
                                            if name != "Unknown":
                                                found = False
                                                for profile in self.known_profiles:
                                                    if profile.face_name == name:
                                                        profile.update_seen()
                                                        profile.face_dimensions = (x, y, w, h)
                                                        profile.update_position(x + w//2, y + h//2)
                                                        final_name = profile.face_name
                                                        found = True
                                                        break
                                                if not found:
                                                    existing = self._find_existing_profile(u_norm, self.known_profiles)
                                                    if existing:
                                                        existing.update_seen()
                                                        existing.face_dimensions = (x, y, w, h)
                                                        existing.update_position(x + w//2, y + h//2)
                                                        final_name = existing.face_name
                                                    else:
                                                        profile = FaceProfile(face_crop, (x, y, w, h), u_norm, 
                                                                            face_name=name, face_id=self.next_known_id)
                                                        profile.last_frame = current_frame.copy()
                                                        profile.update_position(x + w//2, y + h//2)
                                                        self.known_profiles.append(profile)
                                                        self.next_known_id += 1
                                                        final_name = name

                                            else:
                                                found_known = False
                                                for known in self.known_profiles:
                                                    sim = 1 - cosine(u_norm, known.face_embeddings)
                                                    if sim > self.min_confidence:
                                                        final_name = known.face_name
                                                        known.update_seen()
                                                        known.face_dimensions = (x, y, w, h)
                                                        known.update_position(x + w//2, y + h//2)
                                                        found_known = True
                                                        break

                                                if not found_known:
                                                    with self.un_profiles_lock:
                                                        existing = self._find_existing_profile(u_norm, self.unknown_profiles)
                                                        if existing:
                                                            existing.update_seen()
                                                            existing.face_dimensions = (x, y, w, h)
                                                            existing.update_position(x + w//2, y + h//2)
                                                            if self.target_profile and self.target_profile.face_id == existing.face_id:
                                                                self.target_profile = existing
                                                        else:
                                                            profile = FaceProfile(face_crop, (x, y, w, h), u_norm,
                                                                                face_id=self.next_unknown_id)
                                                            profile.last_frame = current_frame.copy()
                                                            profile.update_position(x + w//2, y + h//2)
                                                            self.unknown_profiles.append(profile)
                                                            self._save_unknown_face(profile)
                                                            self.next_unknown_id += 1
                                                            
                                                            if self.target_profile is None:
                                                                self.target_profile = profile
                                                                profile.is_target = True
                                                                
                                            new_labels[(x, y, w, h)] = (final_name, current_time, face.get('face_id', -1))
                            except Exception as e:
                                print(f"Error processing face: {e}")
                                continue
                
                with self.name_cache_lock:
                    self.face_labels = {bbox: (name, fid) for bbox, (name, _, fid) in new_labels.items()}
                
                time.sleep(0.03)
            except Exception as e:
                print(f"Recognition Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _process_unknown(self):
        """
            Process unknown faces and generate movement commands for tracking.
            Sends commands to ESP32-CAM for following the target person.
            
            Movement Strategy:
                1. LEFT/RIGHT: Center the target in frame
                2. FORWARD: Approach target when centered
                3. STOP: When target is both centered AND at optimal distance/size
        """
        last_command = None
        command_cooldown = 0.3  # Minimum time between commands (seconds)
        last_command_time = 0
        lost_counter = 0
        max_lost_frames = 10  # Frames to wait before giving up on target
        
        # State tracking for consistent movement
        centering_phase = True  # True = centering, False = approaching
        last_centered_time = 0
        centered_duration_threshold = 1.0  # Seconds before starting to move forward
        
        while self.running:
            try:
                self._update_target_profile()
                
                with self.frame_lock:
                    if self.frame_data is None:
                        time.sleep(0.05)
                        continue
                    current_frame, _ = self.frame_data
                    if current_frame is None:
                        time.sleep(0.05)
                        continue
                    frame_center_x = current_frame.shape[1] // 2
                    frame_width = current_frame.shape[1]
                
                with self.un_profiles_lock:
                    if self.target_profile:
                        if self.target_profile.is_target:
                            # Get smoothed position for smoother tracking
                            smooth_x, smooth_y = self.target_profile.get_smoothed_position()
                            horizontal_offset = smooth_x - frame_center_x
                            
                            # Get face size for distance estimation
                            _, _, w, h = self.target_profile.face_dimensions
                            face_area = w * h
                            
                            current_time = time.time()
                            current_command = None
                            movement_intensity = 0.0
                            
                            # Reset lost counter since we see the target
                            lost_counter = 0
                            
                            # STEP 1: Check if target is centered horizontally
                            is_centered = abs(horizontal_offset) <= self.centering_dead_zone
                            
                            # STEP 2: Check if target is at optimal distance
                            is_at_optimal_distance = abs(face_area - self.OPTIMAL_FACE_SIZE) <= self.FORWARD_DEAD_ZONE
                            is_too_far = face_area < self.MIN_FACE_SIZE
                            is_too_close = face_area > self.MAX_FACE_SIZE
                            
                            # Determine current state
                            if not is_centered:
                                # Priority 1: Center the target
                                centering_phase = True
                                current_state = "CENTERING"
                                
                                if horizontal_offset < 0:  # Face is to the left
                                    current_command = "LEFT"
                                    # Intensity proportional to how far from center
                                    movement_intensity = min(abs(horizontal_offset) / (frame_width / 2), 1.0)
                                else:  # Face is to the right
                                    current_command = "RIGHT"
                                    movement_intensity = min(abs(horizontal_offset) / (frame_width / 2), 1.0)
                                    
                            elif is_centered:
                                # Target is centered, now decide about forward/backward
                                
                                if is_at_optimal_distance:
                                    # Target is both centered and at optimal distance, then stop
                                    current_state = "STOPPED"
                                    current_command = "STOP"
                                    movement_intensity = 0.0
                                    centering_phase = True
                                    
                                    # Reset centered timer since we're stopping
                                    last_centered_time = current_time
                                    
                                elif is_too_far:
                                    # Target is centered but too far, then move FORWARD
                                    current_state = "APPROACHING"
                                    current_command = "FORWARD"
                                    # Intensity based on how far we are from optimal size
                                    size_error = self.OPTIMAL_FACE_SIZE - face_area
                                    movement_intensity = min(size_error / self.OPTIMAL_FACE_SIZE, 1.0)
                                    movement_intensity = max(0.3, movement_intensity)  # Minimum 30% speed
                                    
                                    # Reset centering phase since we're moving forward
                                    centering_phase = False
                                    
                                elif is_too_close:
                                    # Target is centered but too close, then move BACKWARD
                                    current_state = "BACKING_OFF"
                                    current_command = "BACKWARD"
                                    # Intensity based on how close we are
                                    size_error = face_area - self.OPTIMAL_FACE_SIZE
                                    movement_intensity = min(size_error / self.OPTIMAL_FACE_SIZE, 1.0)
                                    movement_intensity = max(0.3, movement_intensity)  # Minimum 30% speed
                                    
                                    # Reset centering phase since we're moving backward
                                    centering_phase = False
                                    
                                else:
                                    # Target is centered but not at optimal distance
                                    # Need to move forward/backward but track centering status
                                    if centering_phase:
                                        # Just finished centering, wait a moment before moving forward
                                        if current_time - last_centered_time > centered_duration_threshold:
                                            centering_phase = False
                                            last_centered_time = current_time
                                        current_command = "STOP"
                                        current_state = "PAUSED"
                                    else:
                                        # Moving toward target
                                        if face_area < self.OPTIMAL_FACE_SIZE:
                                            current_command = "FORWARD"
                                            size_error = self.OPTIMAL_FACE_SIZE - face_area
                                            movement_intensity = min(size_error / self.OPTIMAL_FACE_SIZE, 1.0)
                                            movement_intensity = max(0.2, movement_intensity)
                                            current_state = "ADVANCING"
                                        else:
                                            current_command = "BACKWARD"
                                            size_error = face_area - self.OPTIMAL_FACE_SIZE
                                            movement_intensity = min(size_error / self.OPTIMAL_FACE_SIZE, 1.0)
                                            movement_intensity = max(0.2, movement_intensity)
                                            current_state = "RETREATING"
                            
                            # Apply PID for horizontal movement when centering
                            if current_command in ["LEFT", "RIGHT"]:
                                error = -horizontal_offset
                                pid_output = self._calculate_pid_movement(error)
                                # Blend PID with proportional control
                                movement_intensity = (movement_intensity + abs(pid_output) / 100) / 2
                                movement_intensity = max(0.2, min(1.0, movement_intensity))
                            
                            # Send command with cooldown to prevent spam
                            if (current_command != last_command or 
                                current_time - last_command_time > command_cooldown):
                                
                                # Log the current state for debugging
                                print(f"[TRACKING] State: {current_state} | "
                                      f"ID: {self.target_profile.face_id} | "
                                      f"Offset: {horizontal_offset:+d}px | "
                                      f"Size: {face_area}px² | "
                                      f"Command: {current_command} | "
                                      f"Intensity: {movement_intensity:.2f}")
                                
                                if current_command:
                                    #self._send_to_esp32(current_command, movement_intensity)
                                    pass
                                last_command = current_command
                                last_command_time = current_time
                                
                        else:
                            # Target exists but not marked as target (shouldn't happen)
                            pass
                    else:
                        # No target detected
                        lost_counter += 1
                        if lost_counter >= max_lost_frames:
                            if self.unknown_profiles:
                                self.target_profile = self._find_largest_face(self.unknown_profiles)
                                print(f"Auto-selected new target, ID: {self.target_profile.face_id}")
                                lost_counter = 0
                                centering_phase = True
                            elif last_command != "SEARCHING":
                                print("[TRACKING] No target detected, searching...")
                                # Send search command (slow rotation)
                                #self._send_to_esp32("LEFT", 0.3)
                                last_command = "SEARCHING"
                                last_command_time = time.time()
                        
                time.sleep(0.05)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                print(f"Error in _process_unknown: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _draw_display(self):
        """
            Display the captured frames to the screen
                Draw Rectangles around the Recognized Faces with a Green Color, along with their Labels
                Draw Rectangles around the Un-Recognized Faces with a Red Color, along with 'Unknown' Label
        """
        while self.running:
            try:
                with self.frame_lock:
                    if self.frame_data is None:
                        time.sleep(0.03)
                        continue
                    current_frame, current_faces = self.frame_data
                    if current_frame is None:
                        time.sleep(0.03)
                        continue
                    frame = current_frame.copy()
                    faces = [tuple(f) for f in current_faces] if current_faces else []
                
                with self.name_cache_lock:
                    current_labels = self.face_labels.copy()
                
                face_to_label = {}
                
                for bbox in faces:
                    x, y, w, h = bbox
                    cx, cy = x + w//2, y + h//2
                    
                    best_match = "Unknown"
                    best_id = -1
                    min_dist = float('inf')
                    
                    for label_bbox, (label_name, fid) in current_labels.items():
                        lx, ly, lw, lh = label_bbox
                        lcx, lcy = lx + lw//2, ly + lh//2
                        dist = np.sqrt((cx - lcx)**2 + (cy - lcy)**2)
                        
                        if dist < 30 and dist < min_dist:
                            min_dist = dist
                            best_match = label_name
                            best_id = fid
                    
                    face_to_label[bbox] = (best_match, best_id)
                
                for bbox, (name, fid) in face_to_label.items():
                    x, y, w, h = bbox
                    
                    # Green for recognized faces, Red for unknown
                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                    display_name = f"{name}" if fid == -1 else f"{name} (ID:{fid})"
                    
                    # Check if this face is the current target
                    with self.profiles_lock:
                        if self.target_profile and self.target_profile.face_id == fid:
                            color = (255, 0, 255)  # Magenta for target
                            display_name = f"TARGET: {name} (ID:{fid})"
                    
                    # Draw rectangle around the face
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label with background
                    (text_width, text_height), _ = cv.getTextSize(display_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv.rectangle(frame, 
                               (x, y - text_height - 10), 
                               (x + text_width + 10, y - 5), 
                               (0, 0, 0), -1)
                    cv.putText(frame, display_name, 
                              (x + 5, y - 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display statistics with distance information
                with self.profiles_lock:
                    stats = f"Known: {len(self.known_profiles)} | Unknown: {len(self.unknown_profiles)}"
                    if self.target_profile and frame is not None:
                        smooth_x, _ = self.target_profile.get_smoothed_position()
                        offset = smooth_x - (frame.shape[1] // 2)
                        _, _, w, h = self.target_profile.face_dimensions
                        face_area = w * h
                        
                        # Determine status message
                        if abs(offset) <= self.centering_dead_zone:
                            if face_area < self.OPTIMAL_FACE_SIZE - self.FORWARD_DEAD_ZONE:
                                status = "→ MOVING FORWARD"
                            elif face_area > self.OPTIMAL_FACE_SIZE + self.FORWARD_DEAD_ZONE:
                                status = "← MOVING BACKWARD"
                            else:
                                status = "AT TARGET"
                        else:
                            if offset < 0:
                                status = "CENTERING LEFT"
                            else:
                                status = "CENTERING RIGHT"
                        
                        stats += f" | Target ID: {self.target_profile.face_id} | Offset: {offset:+d}px | Size: {face_area}px² | {status}"
                
                cv.putText(frame, stats, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.imshow('Face Recognition', frame)
                
                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    #self._send_to_esp32("STOP", 0.0)  # Stop robot on quit
                            
            except Exception as e:
                print(f"Display Error: {e}")
                time.sleep(0.1)

    def _run_camera(self, flag='s'):
        """
            Run the Camera
        """
        self.cap = None 
        try:
            url = self.streaming_url if flag == 's' else self.CAM_INDEX
            self.cap = cv.VideoCapture(url)
            if not self.cap or not self.cap.isOpened():
                print(f"Failed to open camera: {url}")
                self.running = False
                return

            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)

            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                processed_frame = cv.flip(frame, self.CAM_ORIENTATION)
                
                if processed_frame is None or processed_frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                # Get faces for this exact frame
                try:
                    faces = self.face_engine.detectFaces(processed_frame)
                except Exception as e:
                    print(f"Face detection error: {e}")
                    faces = []
                
                # Validate faces against frame dimensions
                h, w = processed_frame.shape[:2]
                valid_faces = []
                for face in faces:
                    if len(face) == 4:
                        x, y, fw, fh = face
                        if (0 <= x < w and 0 <= y < h and 
                            x + fw <= w and y + fh <= h and 
                            fw > 10 and fh > 10):
                            valid_faces.append((int(x), int(y), int(fw), int(fh)))
                
                # Store frame and faces together atomically
                with self.frame_lock:
                    self.frame_data = (processed_frame.copy(), valid_faces.copy())
                
                frame_count += 1
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Camera Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()

    def main(self, **kwargs):
        """
            The main thread of the Recognition System
        """
        self.running = True
        c_flag = kwargs.get('c_flag', 's')

        server = SocketInterface('reco_service', '_http._tcp', '.local.', 5000)

        threads = [
            threading.Thread(target=self._run_camera, args=(c_flag,), name="Camera"),
            threading.Thread(target=self._draw_display, name="Display"),
            threading.Thread(target=self._recognize, name="Recognize"),
            threading.Thread(target=self._process_unknown, name="Unknown"),
            threading.Thread(target=server.advertise_mdns)
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        
        try:
            while self.running:
                server.run_server()
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            #self._send_to_esp32("STOP", 0.0)  # Stop robot on shutdown
        
        for t in threads:
            t.join(timeout=1.0)

        cv.destroyAllWindows()
        print("System shutdown complete")

if __name__ == '__main__':
    Engine().main(c_flag = "l")