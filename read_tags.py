import serial
import customtkinter as ctk
import json
import time
import threading

from serial.tools import list_ports
from tkinter import messagebox as msg

class TagApp:
    def __init__(self):
        self.labels = []
        self.ports = []
        self.tags = []
        self.loaded_tags = []
        self.loaded_labels = []
        self.already_read_tags = []
        self.port_update_running = True
        self.serial_port = None
        self.tag_variable = None

        self._load_files()

    def _load_files(self):
        try:
            with open('tags.json') as tags_file:
                self.loaded_tags = json.load(tags_file)
            with open('labels.json') as labels_file:
                self.loaded_labels = json.load(labels_file)
        except FileNotFoundError as e:
            print(f'Reading File was not Possible: {e}')

    def available_ports(self):
        try:
            ports = [port.device for port in list_ports.comports()]
            current_ports = self.port_menu.cget('values')
            
            if set(ports) != set(current_ports):
                if ports:
                    self.port_menu.configure(values=ports)
                    if not self.port_menu.get() or self.port_menu.get() not in ports:
                        self.port_menu.set(ports[0])
                else:
                    self.port_menu.configure(values=["No ports available"])
                    self.port_menu.set("No ports available")
                    
            print(f'Available Ports: {ports}')
        except Exception as e:
            print(f'Error Getting Ports: {e}')

        if self.port_update_running:
            self.root_window.after(2000, self.available_ports)

    def listen_port(self):
        while True:
            try:
                selected_port = self.port_menu.get()
                
                if not selected_port or selected_port == "No ports available":
                    time.sleep(1)
                    continue
                    
                if self.serial_port is None or not self.serial_port.is_open:
                    self.serial_port = serial.Serial()
                    self.serial_port.baudrate = 9600
                    self.serial_port.port = selected_port  # Use full port name
                    self.serial_port.timeout = 1
                    self.serial_port.open()
                    print(f"Connected to {selected_port}")

                while self.serial_port.is_open:
                    data = self.serial_port.read_until(b'eee')
                    if data:
                        tag_read = data.decode().strip('\r\neee')
                        if tag_read:  # Only process non-empty tags
                            self.root_window.after(0, self.read_tag, tag_read)

            except serial.SerialException as e:
                print(f'Serial Error: {e}')
                if self.serial_port:
                    self.serial_port.close()
                time.sleep(2)
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(2)

    def read_tag(self, tag: str):
        if tag not in self.already_read_tags:
            self.already_read_tags.append(tag)
            if len(self.tag_entry.get()) == 0:
                self.tag_variable.set(tag.strip())
            if self.tag_entry.get() in self.tags:
                self.tag_variable.set(tag.strip())
            print(f"Tag read: {tag}")

    def export_tags(self):
        try:
            with open('dataset.json', 'r') as dataset_file:
                data = json.load(dataset_file)
        
        except Exception as e:
            print(f'Error reading the file: {e}')

        if data:
            data_labels = data['Label']
            data_tags = data['Tag']

            for index, label in enumerate(self.labels):
                if label in data_labels:
                    label_index = data_labels.index(label)
                    if not data_tags[label_index]:
                        data_tags[label_index] = self.tags[index]
                else:
                    data_labels.append(label)
                    data_tags.append(self.tags[index])

            data['Label'] = data_labels
            data['Tag'] = data_tags
            
        try:
            with open('dataset.json', 'w') as dataset_file:
                json.dump(data, dataset_file)

        except Exception as e:
            print(f'Error Writing to the file')
     
    def save_func(self):
        try: 
            label_entered = self.labeling_entry.get()
            tag_read = self.tag_entry.get()
        except Exception as e:
            print(f'An Error occurred: {e}')
        if len(label_entered) > 0 and len(tag_read) > 0:
            if label_entered not in self.loaded_labels and label_entered not in self.labels:
                self.labels.append(label_entered)
            if tag_read not in self.loaded_tags and tag_read not in self.tags:
                self.tags.append(tag_read)

    def _build_app(self, window):
        self.tag_variable = ctk.StringVar()
        content_frame = ctk.CTkFrame(window, fg_color='transparent')
        content_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        upper_frame = ctk.CTkFrame(content_frame, width=300, height=250, corner_radius=5)
        upper_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        upper_frame.grid_columnconfigure(0, weight=0)
        upper_frame.grid_columnconfigure(1, weight=2)
        upper_frame.grid_columnconfigure(2, weight=1)

        labeling_lb = ctk.CTkLabel(upper_frame, text='Label')
        labeling_lb.grid(row=0, column=0, padx=2, pady=2)

        self.labeling_entry = ctk.CTkEntry(upper_frame)
        self.labeling_entry.grid(row=0, column=1, padx=5, pady=5)

        tag_lb = ctk.CTkLabel(upper_frame, text='Tag')
        tag_lb.grid(row=1, column=0, padx=2, pady=2)

        self.tag_entry = ctk.CTkEntry(upper_frame, textvariable = self.tag_variable)
        self.tag_entry.grid(row=1, column=1, padx=5, pady=5)

        port_lb = ctk.CTkLabel(upper_frame, text='Ports')
        port_lb.grid(row=0, column=2, padx=2, pady=2)

        self.port_menu = ctk.CTkOptionMenu(
            upper_frame, 
            values=["Scanning ports..."], 
            bg_color='#2b2b2b', 
            fg_color='#2b2b2b', 
            button_hover_color='#2b2b2b', 
            dropdown_hover_color='#2b2b2b'
        )
        self.port_menu.grid(row=1, column=2, padx=5, pady=5)

        lower_frame = ctk.CTkFrame(content_frame, width=200, height=100, corner_radius=5)
        lower_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        lower_frame.grid_columnconfigure(0, weight = 1)
        lower_frame.grid_columnconfigure(1, weight = 1)

        save_btn = ctk.CTkButton(lower_frame, text='Save', fg_color='#2b2b2b', corner_radius=10, hover_color='#2b2b2b', command = self.save_func)
        save_btn.grid(row = 0, column = 0, padx = 5, pady = 5)

        export_btn = ctk.CTkButton(lower_frame, text='Export', fg_color='#2b2b2b', corner_radius=10, hover_color='#2b2b2b', command=self.export_tags)
        export_btn.grid(row=0, column=1, padx=5, pady=5)

    def run(self):
        self.root_window = ctk.CTk()
        self.root_window.title('Tag Reader')
        self.root_window.geometry('350x150')
        self.root_window.resizable(False, False)
        self.root_window.protocol('WM_DELETE_WINDOW', self.on_closing)

        self._build_app(self.root_window)
        self.root_window.after(1000, self.available_ports)

        self.serial_thread = threading.Thread(target=self.listen_port, daemon=True)
        self.serial_thread.start()

        self.root_window.mainloop()

    def on_closing(self):
        self.port_update_running = False
        if hasattr(self, 'serial_port') and self.serial_port:
            self.serial_port.close()
        self.root_window.destroy()

if __name__ == '__main__':
    app = TagApp()
    app.run()