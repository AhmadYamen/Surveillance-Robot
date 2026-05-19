import numpy as np
import os
import yaml

from typing import List
from camera_source import CameraSource


class EngineConfiguration:
    """
        Engine Configuration class for the Recognition Engine
    """
    
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
        self.DEFAULT_CAMERA_INDEX: int = 0
        self._default_settings: bool = False
        self.unknown_dir: str = 'Unknown'
        self.camera_source: CameraSource = CameraSource.LOCAL
        self.remote_url: str = None
        self.remote_enabled: bool = False
        self.mdns_name: str = None
        self.esp32_ip: str = None  # Store resolved IP
        self._parse_config()
        self._ensure_directories()

    def _ensure_directories(self):
        if not os.path.exists(self.unknown_dir):
            os.makedirs(self.unknown_dir)
        if not os.path.exists('received_images'):
            os.makedirs('received_images')

    def set_camera_source(self, source: CameraSource, url: str = None):
        """
            Switch between local and remote camera sources
        """
        self.camera_source = source
        if source == CameraSource.REMOTE:
            if url is not None:
                self.remote_url = url
        print(f"[CAMERA] Source set to: {source.value}")

    def _parse_config(self):
        """
            Load data from dataset
        """
        data = None
        try:
            if os.path.exists('config.yaml'):
                with open('config.yaml', 'r') as file:
                    data = yaml.safe_load(file)
            else:
                print(f'[WARNING] config.yaml not found, using defaults')
                self._default_settings = True
                
                # Create default config
                default_config = {
                    'camera': {
                        'CAM_WIDTH': 640,
                        'CAM_HEIGHT': 480,
                        'CAM_ORIENTATION': -1,
                        'CAM_INDEX': 0,
                        'remote': {
                            'enabled': False,
                            'mdns_name': 'esp32-cam',
                            'protocol': 'http',
                            'stream_path': 'videostream'
                        }
                    },
                    'socket_server': {
                        'PORT': 5000
                    }
                }
                
                with open('config.yaml', 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                print("[CONFIG] Created default config.yaml")
                
        except yaml.YAMLError as e:
            print(f'[ERROR] Error parsing YAML: {e}')
            self._default_settings = True
        
        if data and not self._default_settings:
            # Socket server configuration
            if 'socket_server' in data:
                socket_config = data['socket_server']
                self.socket_port = socket_config.get('PORT', 5000)
            else:
                self.socket_port = 5000
            
            # Camera settings
            if 'camera' in data:
                camera_attrs = data['camera']
                self.CAM_WIDTH = camera_attrs.get('CAM_WIDTH', 640)
                self.CAM_HEIGHT = camera_attrs.get('CAM_HEIGHT', 480)
                self.CAM_ORIENTATION = camera_attrs.get('CAM_ORIENTATION', -1)
                self.CAM_INDEX = camera_attrs.get('CAM_INDEX', self.DEFAULT_CAMERA_INDEX)
                
                # Remote camera configuration
                if 'remote' in camera_attrs:
                    remote_config = camera_attrs['remote']
                    self.remote_enabled = remote_config.get('enabled', False)
                    self.mdns_name = remote_config.get('mdns_name', 'unknown')
                    self.protocol = remote_config.get('protocol', 'http')
                    self.stream_path = remote_config.get('stream_path', 'videostream')
                    
                    """ if self.remote_enabled:
                        self.camera_source = CameraSource.REMOTE
                        print(f"[CONFIG] Remote camera enabled")
                    else:
                        print(f"[CONFIG] Remote camera configured but disabled")
                        self.remote_url = None """
            else:
                self.CAM_WIDTH = 640
                self.CAM_HEIGHT = 480
                self.CAM_ORIENTATION = -1
                self.CAM_INDEX = self.DEFAULT_CAMERA_INDEX
        
        if self._default_settings:
            self.socket_port = 5000
            self.CAM_WIDTH = 640
            self.CAM_HEIGHT = 480
            self.CAM_ORIENTATION = -1
            self.CAM_INDEX = self.DEFAULT_CAMERA_INDEX
            print("[CONFIG] Using default configuration")
