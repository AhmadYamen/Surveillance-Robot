import time
import numpy as np
import cv2 as cv
import dataclasses
import os
import json
import threading
import embedding_engine as eng
import yaml

from PIL import Image
from scipy.spatial.distance import cosine

@dataclasses.dataclass
class EngineConfiguration:
    embeddings: np.ndarray = None
    labels: list = None
    DEFAULT_CAMERA_INDEX = 1
    _default_settings = False

    try:
        with open('config.yaml', 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f'File config.yaml not found')
    except yaml.YAMLError as e:
        print(f'Error parsing YAML: {e}')
    
    if data:
        ## Remote Camera Configuration
        if 'remote_camera' in data:
            remote_camera_attrs = data['remote_camera']
            remote_camera_credent = remote_camera_attrs['credentials']
            remote_camera_domain = remote_camera_attrs['domain']

            PROTOCOL = remote_camera_domain['PROTOCOL']
            USERNAME = remote_camera_credent['USERNAME']
            PASSWORD = remote_camera_credent['PASSWORD']

            IP = f'{remote_camera_domain['IP']['SEG_A']}.{remote_camera_domain['IP']['SEG_B']}.{remote_camera_domain['IP']['SEG_C']}'
            PORT = remote_camera_domain['PORT']
            stream_path = remote_camera_attrs['stream']['PATH']
            streaming_url = f'{PROTOCOL}://{IP}:{PORT}/{stream_path}'
        else:
            _default_settings = True

        if 'camera' in data:
            ## Camera Settings
            camera_attrs = data['camera']
            CAM_WIDTH = camera_attrs['CAM_WIDTH']
            CAM_HEIGHT = camera_attrs['CAM_HEIGHT']
            CAM_ORIENTATION = camera_attrs['CAM_ORIENTATION']
            CAM_INDEX = camera_attrs['CAM_INDEX']

            if _default_settings:
                _default_settings = False
        else:
            _default_settings = True

    else:
        print('No Remote Camera Was Found')
        streaming_url = DEFAULT_CAMERA_INDEX

    if _default_settings:
        CAM_INDEX = DEFAULT_CAMERA_INDEX
        CAM_WIDTH = 720
        CAM_HEIGHT = 480
        CAM_ORIENTATION = -1

    unknown_dir = 'Unknown'

class Engine(EngineConfiguration):
    def __init__(self):
        self.latest_frame = None
        self.latest_faces = []
        self.face_labels = {} 
        
        self.discovered = []

        self.running = False
        self.face_engine = eng.FaceExtractorEngine()
        self._load_configuration()

    def _load_configuration(self):
        if os.path.exists('dataset.json'):
            with open('dataset.json', 'r') as dataset_file:
                data = json.load(dataset_file)
                self.embeddings = [np.fromstring(embed.strip('[]'), sep = ' ') for embed in data['Embedding']]
                self.labels = data['Label']
        else:
            self.embeddings = []
            self.labels = []

        if not os.path.exists(self.unknown_dir):
            os.mkdir(self.unknown_dir)

    def _save_unknowns(self, face_crop):
        timestamp = int(time.time())
        file_path = os.path.join(self.unknown_dir, f'unknown_{timestamp}.jpg')
        cv.imwrite(file_path, face_crop)
        print(f'Saved unknown face to {file_path}')

    def _recognize(self):
        """
            Recognition Thread
        """
        while self.running:
            if self.latest_frame is None or len(self.latest_faces) == 0:
                time.sleep(0.1)
                continue

            # Sanpshot frames
            snap_frame = self.latest_frame.copy()
            snap_faces = self.latest_faces
            
            # Heavy calculation
            results = self.face_engine.getFaceEmbeddings(snap_frame, snap_faces)
        
            if results:
                new_labels = {}
                for face in results:
                    u_embed = face['embedding']
                    bbox = tuple(face['bbox'])
                    x, y, w, h = bbox

                    # Normalizing Embeddings for unknown face
                    u_norm = u_embed / (np.linalg.norm(u_embed) + 1e-7)
                    
                    best_sim = 0
                    ## need to be fixed
                    if not self.embeddings:
                        name = ''

                    for idx, s_embed in enumerate(self.embeddings):
                        s_norm = s_embed / (np.linalg.norm(s_embed) + 1e-7) # Normalizing Embeddings for known faces
                        sim = 1 - cosine(u_norm, s_norm) # Calculate Similarity using cosine
                        if sim > best_sim:
                            best_sim = sim
                            if sim > 0.7:
                                name = 'Recognized'
                                if self.labels[idx] not in self.discovered:
                                    self.discovered.append(self.labels[idx])

                                #name_labelled = self.labels[idx]
                                #name = f"{name_labelled} ({int(sim*100)}%)"
                                #if name_labelled not in self.discovered:
                                   # self.discovered.append(name_labelled)
                            else:
                                # If face is not known, then save a snapshot of it 
                                name = 'Unknown'
                                face_crop = snap_frame[y:y+h, x:x+w] 
                                if face_crop.size > 0:
                                    self._save_unknowns(face_crop)

                    new_labels[bbox] = name
                self.face_labels = new_labels
            time.sleep(0.5) 

    def _draw_display(self):
        """
            Display Thread
        """

        while self.running:
            # If there is no frame captured, then skip
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            
            for (x, y, w, h) in self.latest_faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                name = self.face_labels.get((x, y, w, h), "...")
                cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv.imshow('Fast Recognition', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    """ def capture_frame_url(self):
        try:    
            session = rs.Session()
            session.auth = ('vanier', 5128)
            response = session.get(snapshot_url, timeout = 10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image

        except rs.exceptions.RequestException as e:
            print(f'Error Fetching Image {e}')
            return None """

    def _run_camera(self, flag = 's'):
        """
            This function will receive a frame from ESP32CAM process it and send commands
            and then display it at the same time
        """
        if flag != 's':
            self.cap = cv.VideoCapture(self.CAM_INDEX)

        else:
            self.cap = cv.VideoCapture(self.streaming_url)

        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        # Note: Set the camera resoulation to lower, if the camera is so slow
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            for _ in range(3):
                self.cap.grab()

            ret, frame = self.cap.retrieve()
            if not ret: break
            

            self.latest_frame = cv.flip(frame, self.CAM_ORIENTATION)
            self.latest_faces = self.face_engine.detectFaces(self.latest_frame)

        self.cap.release()

    def main(self, **kwargs):
        self.running = True

        if kwargs:
            c_flag = kwargs['c_flag']  if 'c_flag' in kwargs else ()
            d_flag = kwargs['d_flag'] if 'd_flag' in kwargs else ()
            r_flag = kwargs['r_flag'] if 'r_flag' in kwargs else ()
        else:
            c_flag, d_flag, r_flag = (), (), ()
            
        camera_thread = threading.Thread(target=self._run_camera, args = c_flag)
        display_thread = threading.Thread(target=self._draw_display, args = d_flag)
        recognize_thread = threading.Thread(target=self._recognize, args = r_flag)
        
        for t in [camera_thread, display_thread, recognize_thread]: t.start()
        for t in [camera_thread, display_thread, recognize_thread]: t.join()
        cv.destroyAllWindows()

if __name__ == '__main__':
    Engine().main()