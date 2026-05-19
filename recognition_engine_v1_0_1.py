import time
import numpy as np
import cv2 as cv
import os
import json
import threading
import socket as stc
from typing import List, Optional, Tuple
from scipy.spatial.distance import cosine
from PIL import Image

from camera_source import CameraSource
from engine_configuration import EngineConfiguration
from face_profile import FaceProfile

try:
    import embedding_engine as eng
except ImportError:
    print("Warning: embedding_engine not found, face recognition disabled")
    eng = None


class Engine(EngineConfiguration):
    """
    Face recognition engine
    """

    def __init__(self):
        # Thread safety locks
        self.frame_lock = threading.Lock()
        self.profiles_lock = threading.Lock()
        self.name_cache_lock = threading.Lock()
        self.un_profiles_lock = threading.Lock()
        self.socket_lock = threading.Lock()

        # Shared data
        self.received_frame = None
        self.frame_overlayed = None          # latest frame with drawn boxes/labels
        self.frame_data = None               # (raw_frame, faces)

        # Profiles
        self.known_profiles: List[FaceProfile] = []
        self.unknown_profiles: List[FaceProfile] = []
        self.target_profile: Optional[FaceProfile] = None

        # ID counters
        self.next_known_id = 0
        self.next_unknown_id = 0

        # Profile cleanup settings
        self.max_profile_age: float = 30.0
        self.min_confidence: float = 0.5
        self.target_lost_threshold: float = 3.0

        # Control flags
        self.running = False
        self.face_engine = None
        self.threads = []
        self.restart_camera_flag = False  # Flag to restart camera thread

        # Frame rate control
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Initialize face engine if available
        if eng:
            try:
                self.face_engine = eng.FaceExtractorEngine()
                print(f"{'=' * 50}\n[INIT] Face recognition engine initialized")
            except Exception as e:
                print(f"[WARNING] Could not initialize face engine: {e}")
                self.face_engine = None

        self.embeddings = []
        self.labels = []

        # Load configuration
        super().__init__()
        self._load_configuration()
        self.face_labels = {}

        # Print cooldown to avoid spam
        self.last_recognition_print = {}
        self.print_cooldown = 2.0

        print(f"[INIT] Loaded {len(self.embeddings)} known faces")

    
    def start(self, source: str = None, url: str = None):
        """
            Start all engine threads.
                source: 'local' or 'remote'
                url: remote camera URL (if source == 'remote')
        """
        if self.running:
            print("[WARNING] Engine already running, stopping first...")
            self.stop()
            time.sleep(1)

        self.running = True
        self.restart_camera_flag = False

        # Configure camera source
        if source == "local":
            self.set_camera_source(CameraSource.LOCAL)
        elif source == "remote":
            if url:
                self.remote_url = url
                self.set_camera_source(CameraSource.REMOTE, self.remote_url)
            else:
                # Try to resolve mDNS name from config.yaml
                try:
                    self.esp32_ip = stc.gethostbyname(self.mdns_name + ".local")
                    self.remote_url = f"{self.protocol}://{self.esp32_ip}/{self.stream_path}"
                    self.set_camera_source(CameraSource.REMOTE, self.remote_url)
                except Exception:
                    print("[ERROR] Could not resolve remote URL")
        else:
            # Default: local camera
            self.set_camera_source(CameraSource.LOCAL)

        # Create and start threads
        self.threads = [
            threading.Thread(target=self._run_camera, name="CameraThread"),
            threading.Thread(target=self._draw_display, name="DisplayThread"),
            threading.Thread(target=self._recognize, name="RecognitionThread"),
            threading.Thread(target=self._process_unknown, name="UnknownThread"),
        ]
        for t in self.threads:
            t.daemon = True
            t.start()

        print("\n" + "=" * 50)
        print("Face Recognition Engine Started")
        print("=" * 50 + "\n")

    def stop(self):
        """
            Stop all engine threads and release resources.
        """

        if not self.running:
            return
        print("[INFO] Shutting down engine...")
        self.running = False
        self.restart_camera_flag = False

        # Wait for threads to finish
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None

        print("[INFO] Engine shutdown complete")

    def is_running(self) -> bool:
        return self.running

    def get_overlayed_frame_as_pil(self) -> Optional[Image.Image]:
        """
            returns the latest overlayed frame as a PIL Image.
            Returns None if no frame is available yet.
        """
        with self.frame_lock:
            if self.frame_overlayed is None:
                return None
            # OpenCV stores BGR, convert to RGB
            rgb = cv.cvtColor(self.frame_overlayed, cv.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    def get_overlayed_frame_as_array(self) -> Optional[np.ndarray]:
        """
            returns the latest overlayed frame as an RGB numpy array.
            Returns None if no frame is available yet.
        """
        with self.frame_lock:
            if self.frame_overlayed is None:
                return None
            return cv.cvtColor(self.frame_overlayed, cv.COLOR_BGR2RGB)

    def switch_to_local_camera(self):
        """
            Switch camera source to local USB/webcam.
        """

        print("[SWITCH] Switching to local camera...")
        self.set_camera_source(CameraSource.LOCAL)
        self._restart_camera()

    def switch_to_remote_camera(self, url: str = None):
        """
            Switch camera source to a remote URL.
        """
        if url:
            self.remote_url = url
        print(f"[SWITCH] Switching to remote camera: {self.remote_url}")
        self.set_camera_source(CameraSource.REMOTE, self.remote_url)
        self._restart_camera()

    def _load_configuration(self):
        """
            Load known faces from dataset.json.
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
                                print(f"[ERROR] Parsing embeddings: {e}")
            except Exception as e:
                print(f"[ERROR] Loading dataset.json: {e}")

    def _save_unknown_face(self, face_profile: FaceProfile):
        """
            Save unknown face crop to disk.
        """
        timestamp = int(time.time())
        file_path = os.path.join(self.unknown_dir, f'unknown_{timestamp}_{face_profile.face_id}.jpg')
        cv.imwrite(file_path, face_profile.face_crop)

    def _cleanup_old_profiles(self):
        """
            Remove profiles that haven't been seen recently.
        """

        current_time = time.time()
        with self.profiles_lock:
            with self.un_profiles_lock:
                self.unknown_profiles = [p for p in self.unknown_profiles
                                         if current_time - p.last_seen < self.max_profile_age]
            self.known_profiles = [p for p in self.known_profiles
                                   if current_time - p.last_seen < self.max_profile_age * 2]
            # self._update_target_profile()
            #if self.target_profile:
                #if current_time - self.target_profile.last_seen > self.target_lost_threshold:
                    #self.target_profile = None

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
            Find closest known face by cosine similarity.
        """

        if not self.embeddings:
            return "Unknown", 0.0
        best_sim = 0
        best_label = "Unknown"
        for idx, stored_emb in enumerate(self.embeddings):
            similarity = 1 - cosine(embedding, stored_emb)
            if similarity > best_sim:
                best_sim = similarity
                if similarity > self.min_confidence:
                    best_label = self.labels[idx]
        return best_label, best_sim

    def _find_existing_profile(self, embedding: np.ndarray, profiles: List[FaceProfile]) -> Optional[FaceProfile]:
        """
            Check if a profile with similar embedding already exists.
        """

        for profile in profiles:
            sim = 1 - cosine(embedding, profile.face_embeddings)
            if sim > self.min_confidence:
                return profile
        return None

    """ def _update_target_profile(self):
            Update the target (largest) unknown face.
    
        
        with self.un_profiles_lock:
            if self.target_profile:
                if self.target_profile not in self.unknown_profiles:
                    if self.target_profile.is_target:
                        self.target_profile.is_target = False
                    if self.unknown_profiles:
                        self.target_profile = self._find_largest_face(self.unknown_profiles)
                    else:
                        self.target_profile = None
            elif self.unknown_profiles:
                self.target_profile = self._find_largest_face(self.unknown_profiles) 
    """

    """  def _find_largest_face(self, profiles: List[FaceProfile]) -> Optional[FaceProfile]:
        if not profiles:
            return None
        largest = profiles[0]
        for profile in profiles[1:]:
            if profile.size > largest.size:
                if largest.is_target:
                    largest.is_target = False
                largest = profile
        if not largest.is_target:
            largest.is_target = True
        return largest 
    """

    def _print_recognition_result(self, face_id: int, name: str):
        """
            Prints the results
        """
        current_time = time.time()
        if face_id in self.last_recognition_print:
            if current_time - self.last_recognition_print[face_id] < self.print_cooldown:
                return
        if name != "Unknown":
            print(f"[FACE RECOGNIZED] {name}")
        else:
            print(f"[UNKNOWN FACE DETECTED]")
        self.last_recognition_print[face_id] = current_time

    def _recognize(self):
        """
            Main recognition thread, matches faces against known embeddings.
        """

        last_cleanup = time.time()
        last_label_cleanup = time.time()
        process_interval = 0.1

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

                with self.frame_lock:
                    if self.frame_data is None:
                        time.sleep(0.01)
                        continue
                    current_frame, current_faces = self.frame_data
                    if current_frame is None or current_frame.size == 0:
                        time.sleep(0.01)
                        continue
                    current_frame = current_frame.copy()
                    current_faces = [tuple(f) for f in current_faces] if current_faces else []

                new_labels = {}
                if len(current_faces) > 0 and self.face_engine:
                    h_frame, w_frame = current_frame.shape[:2]
                    try:
                        results = self.face_engine.getFaceEmbeddings(current_frame, current_faces)
                    except Exception as e:
                        print(f"[ERROR] getFaceEmbeddings: {e}")
                        time.sleep(0.05)
                        continue

                    if results:
                        for face in results:
                            try:
                                u_embed = face['embedding']
                                bbox = face['bbox']
                                if len(bbox) != 4:
                                    continue
                                x, y, w, h = [int(v) for v in bbox]
                                # Clamp coordinates, to not exceed the frame limits
                                x = max(0, min(x, w_frame - 1))
                                y = max(0, min(y, h_frame - 1))
                                w = min(w, w_frame - x)
                                h = min(h, h_frame - y)
                                if w <= 0 or h <= 0:
                                    continue

                                u_norm = u_embed / (np.linalg.norm(u_embed) + 1e-7)
                                name, confidence = self._find_best_match(u_norm)
                                final_name = name

                                if w > 10 and h > 10:
                                    face_crop = current_frame[y:y + h, x:x + w].copy()
                                    if face_crop.size > 0:
                                        with self.profiles_lock:
                                            if name != "Unknown":
                                                # Check known profiles
                                                found = False
                                                for profile in self.known_profiles:
                                                    if profile.face_name == name:
                                                        profile.update_seen()
                                                        profile.face_dimensions = (x, y, w, h)
                                                        profile.update_position(x + w // 2, y + h // 2)
                                                        final_name = profile.face_name
                                                        found = True
                                                        self._print_recognition_result(profile.face_id, name)
                                                        break
                                                if not found:
                                                    existing = self._find_existing_profile(u_norm, self.known_profiles)
                                                    if existing:
                                                        existing.update_seen()
                                                        existing.face_dimensions = (x, y, w, h)
                                                        existing.update_position(x + w // 2, y + h // 2)
                                                        final_name = existing.face_name
                                                        self._print_recognition_result(existing.face_id, name)
                                                    else:
                                                        new_id = self.next_known_id
                                                        self.next_known_id += 1
                                                        profile = FaceProfile(face_crop, (x, y, w, h), u_norm,
                                                                              face_name=name, face_id=new_id)
                                                        profile.update_position(x + w // 2, y + h // 2)
                                                        self.known_profiles.append(profile)
                                                        final_name = name
                                                        self._print_recognition_result(profile.face_id, name)
                                            else:
                                                # Unknown face
                                                with self.un_profiles_lock:
                                                    existing = self._find_existing_profile(u_norm, self.unknown_profiles)
                                                    if existing:
                                                        existing.update_seen()
                                                        existing.face_dimensions = (x, y, w, h)
                                                        existing.update_position(x + w // 2, y + h // 2)
                                                        if self.target_profile and self.target_profile.face_id == existing.face_id:
                                                            self.target_profile = existing
                                                        self._print_recognition_result(existing.face_id, "Unknown")
                                                    else:
                                                        new_id = self.next_unknown_id
                                                        self.next_unknown_id += 1
                                                        profile = FaceProfile(face_crop, (x, y, w, h), u_norm,
                                                                              face_id=new_id)
                                                        profile.update_position(x + w // 2, y + h // 2)
                                                        self.unknown_profiles.append(profile)
                                                        self._save_unknown_face(profile)
                                                        if self.target_profile is None:
                                                            self.target_profile = profile
                                                            profile.is_target = True
                                                        self._print_recognition_result(profile.face_id, "Unknown")
                                            new_labels[(x, y, w, h)] = (final_name, current_time, face.get('face_id', -1))
                            except Exception as e:
                                print(f"[ERROR] Processing face: {e}")
                                continue

                with self.name_cache_lock:
                    self.face_labels = {bbox: (name, fid) for bbox, (name, _, fid) in new_labels.items()}

                time.sleep(process_interval)
            except Exception as e:
                print(f"[ERROR] Recognition: {e}")
                time.sleep(0.1)

    def _process_unknown(self):
        """
            Periodically update the target unknown face.
        """

        while self.running:
            try:
                #self._update_target_profile()
                time.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] _process_unknown: {e}")
                time.sleep(0.1)

    def _draw_display(self):
        """
            Draw bounding boxes, labels, and stats on the current frame.
        """
        last_frame_time = time.time()
        frame_times = []

        while self.running:
            try:
                with self.frame_lock:
                    if self.frame_data is None:
                        time.sleep(0.01)
                        continue
                    current_frame, current_faces = self.frame_data
                    if current_frame is None or current_frame.size == 0:
                        time.sleep(0.01)
                        continue
                    self.frame_overlayed = current_frame.copy()
                    faces = [tuple(f) for f in current_faces] if current_faces else []

                # FPS calculation
                current_time = time.time()
                frame_times.append(current_time)
                frame_times = [t for t in frame_times if current_time - t < 1.0]
                fps = len(frame_times)

                with self.name_cache_lock:
                    current_labels = self.face_labels.copy()

                # Match detected faces to recognition labels
                face_to_label = {}
                for bbox in faces:
                    x, y, w, h = bbox
                    cx, cy = x + w // 2, y + h // 2
                    best_match = "Unknown"
                    best_id = -1
                    min_dist = float('inf')
                    for label_bbox, (label_name, fid) in current_labels.items():
                        lx, ly, lw, lh = label_bbox
                        lcx, lcy = lx + lw // 2, ly + lh // 2
                        dist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
                        if dist < 30 and dist < min_dist:
                            min_dist = dist
                            best_match = label_name
                            best_id = fid
                    face_to_label[bbox] = (best_match, best_id)

                # Draw bounding boxes and text
                for bbox, (name, fid) in face_to_label.items():
                    x, y, w, h = bbox
                    # Color: red for unknown, green for known, magenta for target
                    if name == "Unknown":
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    # Check if this is the target unknown face
                    with self.profiles_lock:
                        if self.target_profile and self.target_profile.face_id == fid:
                            color = (255, 0, 255)
                            name = f"TARGET: {name}"
                    display_name = f"{name}"
                    cv.rectangle(self.frame_overlayed, (x, y), (x + w, y + h), color, 2)
                    (text_width, text_height), _ = cv.getTextSize(display_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv.rectangle(self.frame_overlayed,
                                 (x, y - text_height - 10),
                                 (x + text_width + 10, y - 5),
                                 (0, 0, 0), -1)
                    cv.putText(self.frame_overlayed, display_name,
                               (x + 5, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Statistics overlay
                with self.profiles_lock:
                    stats = (f"Known: {len(self.known_profiles)} | "
                             f"Unknown: {len(self.unknown_profiles)} | "
                             f"FPS: {fps}")
                    if self.target_profile:
                        stats += f" | Target ID: {self.target_profile.face_id}"

                if self.camera_source == CameraSource.REMOTE and self.mdns_name:
                    source_text = f"Camera: {self.camera_source.value} ({self.remote_url})"
                else:
                    source_text = f"Camera: {self.camera_source.value}"

                cv.putText(self.frame_overlayed, source_text, (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.putText(self.frame_overlayed, stats, (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Allow other threads to run
                time.sleep(0.03)

            except Exception as e:
                print(f"[ERROR] Display drawing: {e}")
                time.sleep(0.1)

    def _restart_camera(self):
        """
            Force camera reconnection (called after source change).
        """

        print("[CAMERA] Restarting camera thread...")
        self.restart_camera_flag = True
        # Wait a bit for the camera thread to restart
        time.sleep(0.5)

    def _run_camera(self):
        """
            Capture frames from the selected camera source.
            Stores raw frame + detected face bounding boxes into self.frame_data.
        """
        self.cap = None
        reconnect_delay = 2
        
        # Track if we need to restart due to source change
        while self.running:
            if self.restart_camera_flag:
                print("[CAMERA] Source changed, reconnecting...")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.restart_camera_flag = False
                time.sleep(0.5)
            
            try:
                print(f"[CAMERA] Connecting to source: {self.camera_source.value}")
                print(f"[CAMERA] Remote URL: {self.remote_url}")
                
                if self.camera_source == CameraSource.REMOTE and self.remote_url:
                    print(f"[CAMERA] Connecting to remote: {self.remote_url}")
                    # Use VideoCapture with backend and lower buffer size
                    self.cap = cv.VideoCapture(self.remote_url, cv.CAP_FFMPEG)
                    if self.cap.isOpened():
                        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                else:
                    print(f"[CAMERA] Using local camera (index: {self.CAM_INDEX})")
                    self.cap = cv.VideoCapture(self.CAM_INDEX)

                if not self.cap or not self.cap.isOpened():
                    print(f"[ERROR] Failed to open camera source: {self.camera_source.value}")
                    time.sleep(reconnect_delay)
                    continue

                # Configure camera properties
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)
                self.cap.set(cv.CAP_PROP_FPS, 30)
                if self.camera_source == CameraSource.REMOTE:
                    self.cap.set(cv.CAP_PROP_READ_TIMEOUT_MSEC, 1000)  # 1 second timeout
                    # For MJPEG streams 
                    self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

                print(f"[OK] Camera connected (Source: {self.camera_source.value})")
                frame_count = 0
                frame_skip = 0
                consecutive_failures = 0

                while self.running and not self.restart_camera_flag:
                    # Skip frames for better performance, every second frame
                    frame_skip = (frame_skip + 1) % 2
                    if frame_skip != 0:
                        ret, _ = self.cap.read()
                        if not ret:
                            consecutive_failures += 1
                            if consecutive_failures > 5:
                                print("[WARNING] Multiple frame read failures, reconnecting...")
                                break
                        else:
                            consecutive_failures = 0
                        time.sleep(0.005)
                        continue

                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        print(f"[WARNING] Frame read failed ({consecutive_failures})")
                        if consecutive_failures > 5:
                            print("[WARNING] Too many failures, reconnecting...")
                            break
                        time.sleep(0.1)
                        continue
                    
                    consecutive_failures = 0

                    current_time = time.time()

                    if self.CAM_ORIENTATION != 0:
                        frame = cv.flip(frame, self.CAM_ORIENTATION)

                    if frame is None or frame.size == 0:
                        time.sleep(0.001)
                        continue

                    # Face detection, skip other frames for speed
                    faces = []
                    if frame_count % 2 == 0 and self.face_engine:
                        try:
                            faces = self.face_engine.detectFaces(frame)
                        except Exception as e:
                            faces = []

                    h, w = frame.shape[:2]
                    valid_faces = []
                    for face in faces:
                        if len(face) == 4:
                            x, y, fw, fh = [int(v) for v in face]
                            if (0 <= x < w and 0 <= y < h and
                                    x + fw <= w and y + fh <= h and
                                    fw > 20 and fh > 20):
                                valid_faces.append((x, y, fw, fh))

                    with self.frame_lock:
                        self.frame_data = (frame.copy(), valid_faces.copy())

                    frame_count += 1

                    # Limit capture rate to 50 fps
                    elapsed = time.time() - current_time
                    if elapsed < 0.02:
                        time.sleep(0.02 - elapsed)

                # Clean up before reconnect
                if self.cap:
                    self.cap.release()
                    self.cap = None

                if self.running and not self.restart_camera_flag:
                    print("[INFO] Reconnecting to camera...")
                    time.sleep(reconnect_delay)

            except Exception as e:
                print(f"[ERROR] Camera thread exception: {e}")
                time.sleep(reconnect_delay)
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None


if __name__ == '__main__':
    engine = Engine()
    engine.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        engine.stop()