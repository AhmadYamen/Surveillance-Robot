import json
import os
from typing import Optional, Dict

class CameraConfig:
    """
        Manage camera configurations and preferences.
    """
    
    CONFIG_FILE = "camera_config.json"
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """
            Load saved camera configuration.
        """

        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_config(self):
        """
            Save camera configuration to file.
        """

        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Saving config: {e}")
    
    def get_last_camera(self) -> Optional[str]:
        """
            Get last used camera type/source.
        """

        return self.config.get('last_camera', None)
    
    def set_last_camera(self, camera_type: str, url: Optional[str] = None):
        """
            Save last used camera.
        """

        self.config['last_camera'] = camera_type
        if url:
            self.config['last_url'] = url
        self.save_config()
    
    def get_saved_urls(self) -> list:
        """
            Get list of saved camera URLs.
        """

        return self.config.get('saved_urls', [])
    
    def add_saved_url(self, url: str, name: str = None):
        """
            Add URL to saved list.
        """
        
        if 'saved_urls' not in self.config:
            self.config['saved_urls'] = []
        
        # Check if URL already exists
        for existing in self.config['saved_urls']:
            if existing.get('url') == url:
                return
        
        self.config['saved_urls'].append({
            'url': url,
            'name': name or url,
            'last_used': None
        })
        self.save_config()