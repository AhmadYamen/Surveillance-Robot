import tkinter as tk
import subprocess
import re
import socket
import threading
import time

from PIL import ImageTk
from recognition_engine_v1_0_1 import Engine

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.discovered_cameras = {}  # {display_name: url}
        self.camera_list = ["Local"]
        self.scanning = False
        self.scan_thread = None
        self.current_source = "Local"
        
        self.create_widgets()
        
        # Start the engine (default to local)
        self.engine = Engine()
        
        # Start UDP listener for ESP32-CAM broadcasts
        self.start_udp_listener()
        
        # Start periodic network scan
        self.start_network_scan()
        
        # Polling loop
        self.update_frame()
        self.update_discovery()
        
        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        self.frame1 = tk.Frame(self.root)
        self.frame1.place(x=2, y=40, width=640, height=440)

        self.video_label = tk.Label(self.frame1)
        self.video_label.pack()

        self.frame2 = tk.Frame(self.root)
        self.frame2.place(x=150, y=490, width=350, height=35)

        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_columnconfigure(1, weight=1)

        self.start_btn = tk.Button(self.frame2, text="Start", width=40, relief="flat", bg="#C0C0C0", command=self.start_btn_command)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)

        self.stop_btn = tk.Button(self.frame2, text="Stop", width=40, relief="flat", bg="#C0C0C0", command=self.stop_btn_command)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)

        self.disco_label = tk.Label(self.root, text="Discovered Faces", bg="#C0C0C0")
        self.disco_label.place(x=660, y=20)

        self.frame3 = tk.Frame(self.root)
        self.frame3.place(x=660, y=40, width=200, height=200)

        self.disco_text_box = tk.Text(self.frame3)
        self.disco_text_box.pack()

        self.frame4 = tk.Frame(self.root)
        self.frame4.place(x=2, y=2, width=640, height=35)

        self.source_label = tk.Label(self.frame4, text="Camera Source:")
        self.source_label.grid(row=0, column=0, padx=5, pady=5)

        self.source_string = tk.StringVar(value="Local")
        self.source_menu = tk.OptionMenu(self.frame4, self.source_string, *self.camera_list, command=self.on_source_change)
        self.source_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # Status label
        self.status_label = tk.Label(self.frame4, text="", fg="blue")
        self.status_label.grid(row=0, column=2, padx=10, pady=5)
        
        # Scan button
        self.scan_btn = tk.Button(self.frame4, text="Scan Network", command=self.scan_network_manual)
        self.scan_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Remote URL entry 
        self.url_label = tk.Label(self.frame4, text="Custom URL:")
        self.url_label.grid(row=0, column=4, padx=5, pady=5)
        
        self.url_entry = tk.Entry(self.frame4, width=30)
        self.url_entry.grid(row=0, column=5, padx=5, pady=5)
        
        self.connect_btn = tk.Button(self.frame4, text="Connect", command=self.connect_custom_url)
        self.connect_btn.grid(row=0, column=6, padx=5, pady=5)

    def start_udp_listener(self):
        """
            Start a UDP listener to receive ESP32-CAM broadcasts on port 9999
        """

        def udp_listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(('', 9999))
            sock.settimeout(1.0)
            
            while self.scanning:
                try:
                    data, addr = sock.recvfrom(1024)
                    message = data.decode('utf-8')
                    if '|' in message:
                        hostname, ip = message.split('|')
                        display_name = f"{hostname} ({ip})"
                        url = f"http://{ip}/videostream"
                        if display_name not in self.discovered_cameras:
                            self.discovered_cameras[display_name] = url
                            self.update_camera_menu()
                            print(f"[DISCOVERED] {hostname} at {ip}")
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[UDP ERROR] {e}")
            sock.close()
        
        self.scanning = True
        self.udp_thread = threading.Thread(target=udp_listener, daemon=True)
        self.udp_thread.start()

    def start_network_scan(self):
        """
            Periodically check for ESP32-CAMs via ARP
        """
        def scan_loop():
            while self.scanning:
                self.scan_network()
                time.sleep(30)  # Scan every 30 seconds
        
        self.scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self.scan_thread.start()

    def scan_network(self):
        """
            Scan network for ESP32-CAM devices
        """
        try:
            try:
                mdns_ip = socket.gethostbyname('esp32cam.local')
                display_name = f"esp32cam ({mdns_ip})"
                url = f"http://{mdns_ip}/stream"
                if display_name not in self.discovered_cameras:
                    self.discovered_cameras[display_name] = url
                    self.update_camera_menu()
                    print(f"[mDNS] Found esp32cam at {mdns_ip}")
            except:
                pass
            
            # ARP scan to find devices
            result = subprocess.run(['arp', '-n'], capture_output=True, text=True)
            lines = result.stdout.splitlines()
            
            for line in lines:
                # Look for IP addresses in ARP table
                match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                if match:
                    ip = match.group(1)
                    # Try to resolve hostname
                    try:
                        hostname = socket.gethostbyaddr(ip)[0]
                        if 'esp32' in hostname.lower():
                            display_name = f"{hostname} ({ip})"
                            url = f"http://{ip}/stream"
                            if display_name not in self.discovered_cameras:
                                self.discovered_cameras[display_name] = url
                                self.update_camera_menu()
                                print(f"[ARP] Found {hostname} at {ip}")
                    except:
                        pass
            
        except Exception as e:
            print(f"[SCAN ERROR] {e}")

    def scan_network_manual(self):
        """
            Manual network scan triggered by button
        """

        self.status_label.config(text="Scanning network...", fg="orange")
        self.status_label.update()
        
        def scan():
            # Clear old discoveries
            self.discovered_cameras.clear()
            
            self.scan_network()
            
            # Try ping common ESP32-CAM IPs
            common_ips = ['192.168.1.100', '192.168.1.101', '192.168.0.100', '192.168.0.101']
            for ip in common_ips:
                response = subprocess.run(['ping', '-c', '1', '-W', '1', ip], capture_output=True)
                if response.returncode == 0:
                    display_name = f"esp32cam ({ip})"
                    url = f"http://{ip}/stream"
                    if display_name not in self.discovered_cameras:
                        self.discovered_cameras[display_name] = url
                        self.update_camera_menu()
                        print(f"[MANUAL] Found device at {ip}")
            
            self.root.after(0, lambda: self.status_label.config(text="Scan complete", fg="green"))
            self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))
        
        threading.Thread(target=scan, daemon=True).start()

    def update_camera_menu(self):
        """
            Update the option menu with discovered cameras
        """
        # Rebuild menu with current discovered cameras
        menu = self.source_menu['menu']
        menu.delete(0, 'end')
        
        self.camera_list = ["Local"]
        for display_name in self.discovered_cameras.keys():
            self.camera_list.append(display_name)
        
        for option in self.camera_list:
            menu.add_command(label=option, command=lambda value=option: self.source_string.set(value))
        
        # Keep current selection if still valid
        current = self.source_string.get()
        if current not in self.camera_list:
            self.source_string.set("Local")

    def on_source_change(self, selected):
        """
            Handle camera source selection change,
            connects immediately
        """

        if not selected or selected == self.current_source:
            return
        
        self.current_source = selected
        self.status_label.config(text=f"Connecting to {selected}...", fg="orange")
        self.status_label.update()
        
        if selected == "Local":
            # Stop and restart engine with local source
            def switch():
                if self.engine.is_running():
                    self.engine.stop()
                    time.sleep(0.5)
                self.engine.start(source="local")
                self.root.after(0, lambda: self.status_label.config(text="Using local camera", fg="green"))
                self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))
            threading.Thread(target=switch, daemon=True).start()
        else:
            # Get URL from discovered cameras
            url = self.discovered_cameras.get(selected)
            if url:
                def switch():
                    if self.engine.is_running():
                        self.engine.stop()
                        time.sleep(0.5)
                    self.engine.start(source="remote", url=url)
                    self.root.after(0, lambda: self.status_label.config(text=f"Connected to {selected}", fg="green"))
                    self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))
                threading.Thread(target=switch, daemon=True).start()
            else:
                self.status_label.config(text="Connection failed: No URL found", fg="red")
                self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))

    def connect_custom_url(self):
        """
            Connect to custom streaming URL
        """

        url = self.url_entry.get().strip()
        if url:
            self.status_label.config(text=f"Connecting to custom URL...", fg="orange")
            self.current_source = url
            
            def switch():
                if self.engine.is_running():
                    self.engine.stop()
                    time.sleep(0.5)
                self.engine.start(source="remote", url=url)
                self.root.after(0, lambda: self.status_label.config(text="Connected to custom source", fg="green"))
                self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))
            
            threading.Thread(target=switch, daemon=True).start()
            
            # Add to discovered Text Box
            display_name = f"custom ({url})"
            if display_name not in self.discovered_cameras:
                self.discovered_cameras[display_name] = url
                self.update_camera_menu()
        else:
            self.status_label.config(text="Please enter a URL", fg="red")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="blue"))

    def update_frame(self):
        """
            Get the latest overlayed frame and display it.
        """
        pil_img = self.engine.get_overlayed_frame_as_pil()
        if pil_img is not None:
            # Resize to fit the window
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk # for keeping reference
        self.root.after(30, self.update_frame) # 30 fps

    def update_discovery(self):
        """
            Update discovered faces display
        """

        if hasattr(self.engine, 'known_profiles') and self.engine.known_profiles:
            self.disco_text_box.delete('1.0', tk.END)
            for profile in self.engine.known_profiles:
                self.disco_text_box.insert(tk.END, profile.face_name + '\n')
        self.root.after(1000, self.update_discovery)

    def start_btn_command(self):
        """
            Start the face recognition engine
        """

        if self.engine.is_running():
            self.status_label.config(text="Engine already running", fg="orange")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="blue"))
            return
        
        # Get current source selection
        selected = self.source_string.get()
        
        if selected == "Local":
            self.engine.start(source="local")
            self.status_label.config(text="Started with local camera", fg="green")
        else:
            url = self.discovered_cameras.get(selected)
            if url:
                self.engine.start(source="remote", url=url)
                self.status_label.config(text=f"Started with {selected}", fg="green")
            else:
                self.status_label.config(text="Failed: No URL for selected source", fg="red")
        
        self.root.after(3000, lambda: self.status_label.config(text="", fg="blue"))

    def stop_btn_command(self):
        """
            Stop the face recognition engine
        """

        if self.engine.is_running():
            self.engine.stop()
            self.status_label.config(text="Stopped", fg="orange")
        else:
            self.status_label.config(text="Engine not running", fg="orange")
        self.root.after(2000, lambda: self.status_label.config(text="", fg="blue"))

    def on_close(self):
        """
            Clean up on window close
        """

        self.scanning = False
        if hasattr(self, 'engine'):
            self.engine.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1050x530")
    root.configure(bg="#C0C0C0")
    app = FaceRecognitionGUI(root)
    root.mainloop()