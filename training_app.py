import os
import cv2 as cv
import numpy as np
import pandas as pd
import json 

import embedding_engine as emg
import customtkinter as ctk
import tkinter as tk
import tkinter.filedialog as tfl
import tkinter.messagebox as msg
import threading

## Ask for folders and label images

class TrainingApp:
    def __init__(self):
        self.labels = [] # store labels 
        self.images = [] # store images browsed for each label

        self.loaded_embeddings = [] # store loaded embeddings from embeddings.npy
        self.loaded_labels = [] # store loaded labels from labels.json

        self.embeddings = [] # store embeddings for each label
        self.images_folders = [] # store folders containing the images that were chosen
        self.data_gathered = dict() # dictionary to store data gathered from user input

        self.top_pick_window = None # a check flag for the top level window of browsing
        self.window = None # main window placeholder

        self.progress_window = None
        self.images_folder_path = 'images'
        self.images_folder_cropped = 'cropped'

        self.face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # Load Face detector
        self.model = emg.FaceExtractorEngine()

        self._load_data()

    def _show_progress_dialog(self, title: str, message: str):
        if self.progress_window is None:
            self.progress_window = ctk.CTkToplevel(self.window)
            self.progress_window.title(title)
            self.progress_window.geometry("300x100")
            self.progress_window.transient(self.window)
            self.progress_window.grab_set()
            
            label = ctk.CTkLabel(self.progress_window, text=message)
            label.pack(pady=20)
            
            self.progress_bar = ctk.CTkProgressBar(self.progress_window)
            self.progress_bar.pack(pady=10)
            self.progress_bar.set(0)
        
        return self.progress_window

    def _crop_face(self, image):
        """
            Detects and crops the face from a BGR image.
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None  # No face detected
            
        # Get the largest face found
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Optional: Add a 10% margin
        margin = int(w * 0.1)
        y1, y2 = max(0, y - margin), min(image.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(image.shape[1], x + w + margin)
        
        return image[y1:y2, x1:x2]

    def _update_progress(self, value: float):
        if self.progress_window:
            self.progress_bar.set(value)
            self.progress_window.update()

    def _close_progress_dialog(self):
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None

    def _get_embeddings(self):
        """For each label, embed its images using threading"""
        if not self.data_gathered:
            msg.showerror('Failed Training', 'No Data were found', icon=msg.ERROR)
            return
        
        # Show progress
        msg.showinfo("Training", "Processing embeddings... This may take a moment.")
        
        # Use threading to prevent UI freezing
        train_thread = threading.Thread(target=self._train_embeddings_thread)
        train_thread.daemon = True
        train_thread.start()
    
    def _train_embeddings_thread(self):
        for label, images in self.data_gathered.items():
            try:
                best_embed = self._calculate_encodings_best(images)
                self.embeddings.append(best_embed)
                self.labels.append(label)
                print(f"Processed label: {label}")
            except Exception as e:
                print(f"Error processing {label}: {e}")
                continue
        
        print(f"Training complete. Labels: {self.labels}")
        
        # Show completion message (needs to be in main thread)
        self.window.after(0, lambda: msg.showinfo(
            "Training Complete", 
            f"Successfully processed {len(self.labels)} labels"
        ))

    def _export_data_base_embeddings(self):
        try:
            # FIX: Create a dictionary to map labels to embeddings to easily overwrite duplicates
            final_data_map = {}
            
            # Load existing
            for label, embed in zip(self.loaded_labels, self.loaded_embeddings):
                final_data_map[label] = embed
                
            # Overwrite/Add new
            for label, embed in zip(self.labels, self.embeddings):
                final_data_map[label] = embed

            # Extract back to lists
            final_labels = list(final_data_map.keys())
            final_embeddings = list(final_data_map.values())

            data = {
                'Label': final_labels,
                'Embedding': [embed.tolist() for embed in final_embeddings]
            }

            with open('dataset.json', 'w') as dataset_file:
                json.dump(data, dataset_file, indent=2)

            msg.showinfo('Success', 'Data Exported Successfully', icon=msg.INFO)

        except Exception as e:
            print(f'Error Occurred When Exporting: {e}')
            msg.showerror('Export Failed', str(e), icon=msg.ERROR)
        
    def _clear_all(self):
        flag_to_approve = msg.askyesnocancel('Are you Sure?', 'This procedure will delete everything', icon = msg.QUESTION)
        if flag_to_approve:
            try:
                self.data_gathered = dict()
                self.images = []
                self.labels = []
                self.embeddings = []
                self.images_folders = []
                msg.showinfo('Success', 'Data was deleted successfully')
            except:
                msg.showerror('Error', 'An Error Ocurrend', msg.ERROR)
                return
        else:
            return
        
    def _clear_browsed(self):
        flag_to_approeve = msg.askyesnocancel('Are you Sure?', 'This operation will delete all images browsed', icon = msg.QUESTION)

        if flag_to_approeve:
            try:
                self.folders_textbox.delete('1.0', 'end')
                self.images = []

            except:
                msg.showerror('Error', 'An Error Ocurred!', icon = msg.ERROR)
                return
        else:
            return

    def _calculate_encodings_best(self, images = None, embeddings = None):
        def geometric_median(points, eps=1e-5, max_iter=100):
            y = np.mean(points, axis=0)
            for i in range(max_iter):
                distances = np.linalg.norm(points - y, axis=1)
                distances = np.where(distances < eps, eps, distances)
                weights = 1 / distances
                y_new = np.average(points, axis=0, weights=weights)
                
                if np.linalg.norm(y_new - y) < eps:
                    break
                y = y_new
            return y
         
        if embeddings is not None:
            embeddings = np.array(embeddings)
            median = geometric_median(embeddings)
            # Normalize before returning
            return median / (np.linalg.norm(median) + 1e-7)
        
        embeddings_per_label = []
        for img_path in images:
            try:
                img = cv.imread(img_path)
                if img is None: 
                    continue
                face_img = self._crop_face(img)
                target_img = face_img if face_img is not None else img
                
                single_img_embedding = self.model.extractEmbedding(target_img)
                embeddings_per_label.append(single_img_embedding)
            except Exception as e:
                print(f"Error embedding image {img_path}: {e}")
                continue
        
        if not embeddings_per_label:
            raise ValueError(f"No valid embeddings for images")
        
        embeddings_per_label = np.array(embeddings_per_label)
        median = geometric_median(embeddings_per_label)
        
        # Normalize before returning
        return median / (np.linalg.norm(median) + 1e-7)
    
    def _load_data(self):
        if os.path.exists('dataset.json'):
            with open('dataset.json', 'r') as dataset_file:
                data = json.load(dataset_file)
                # Convert lists back to numpy arrays
                if 'Embedding' in data and 'Label' in data:
                    self.loaded_embeddings = [np.array(embed_list, dtype=np.float32) for embed_list in data['Embedding']]
                    self.loaded_labels = data['Label']
                else:
                    # Create empty structure if keys don't exist
                    self.loaded_embeddings = []
                    self.loaded_labels = []
        else:
            # Create new file with empty structure
            with open('dataset.json', 'w') as dataset_file:
                data = {
                    'Label': [],
                    'Embedding': [],
                    'Tag': []
                }
                json.dump(data, dataset_file)
            self.loaded_embeddings = []
            self.loaded_labels = []

    def _re_train(self):
        # Create a copy of loaded embeddings to modify
        updated_embeddings = self.loaded_embeddings.copy()
        updated_labels = self.loaded_labels.copy()
        
        ## read labels from loaded stored data
        for idx, loaded_label in enumerate(self.loaded_labels):
            ## read labels from entered data
            for label in list(self.data_gathered.keys()):
                if label == loaded_label:
                    new_retrained_embeddings = self._calculate_encodings_best(self.data_gathered[label])
                    # Combine old and new embeddings
                    combined = np.array([new_retrained_embeddings, self.loaded_embeddings[idx]])
                    updated_embeddings[idx] = self._calculate_encodings_best(embeddings=combined)
        
        # Update the loaded embeddings with the retrained ones
        self.loaded_embeddings = updated_embeddings
        
        self.window.after(0, lambda: msg.showinfo(
            "Retraining Complete", 
            f"Successfully retrained {len(self.loaded_labels)} labels"
        ))

    def build_app(self, window_root: ctk.CTk):
        # Configure main window
        window_root.grid_rowconfigure(0, weight=1)
        window_root.grid_columnconfigure(0, weight=1)

        ## Main Frame
        main_frame = ctk.CTkFrame(window_root, fg_color='transparent')
        main_frame.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(0, weight=0)  # Header
        main_frame.grid_rowconfigure(1, weight=1)  # Content
        main_frame.grid_columnconfigure(0, weight=1)

        ## Header Frame
        header_frame = ctk.CTkFrame(main_frame, height=70, fg_color='#2b2b2b')
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 20))
        
        # Configure header frame
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ctk.CTkLabel(
            header_frame, 
            text='Recognition System Trainer', 
            font=('Arial', 24, 'bold')
        )
        header_label.grid(row=0, column=0, pady=15)

        ## Content Frame
        content_frame = ctk.CTkFrame(main_frame, fg_color='transparent')
        content_frame.grid(row=1, column=0, sticky='nsew')
        
        # Configure content frame grid
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)  # Left panel
        content_frame.grid_columnconfigure(1, weight=1)  # Right panel

        ## Left Panel - Folder Selection
        left_panel = ctk.CTkFrame(content_frame, fg_color='#2b2b2b', corner_radius=10)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Configure left panel
        left_panel.grid_rowconfigure(0, weight=0)  # Title
        left_panel.grid_rowconfigure(1, weight=1)  # Textbox
        left_panel.grid_rowconfigure(2, weight=0)  # Buttons frame
        left_panel.grid_columnconfigure(0, weight=1)

        # Title
        left_title = ctk.CTkLabel(
            left_panel, 
            text='Image Files', 
            font=('Arial', 16, 'bold')
        )
        left_title.grid(row=0, column=0, pady=(15, 10), padx=15, sticky='w')

        # Textbox for folders
        self.folders_textbox = ctk.CTkTextbox(
            left_panel, 
            font=('Arial', 12),
            wrap='word'
        )
        self.folders_textbox.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 10))
        self.folders_textbox.insert("1.0", "No files selected yet.\n\nClick 'Browse Files' to select image files.")
        self.folders_textbox.configure(state='disabled')  # Make read-only

        # Buttons Frame
        btt_frame = ctk.CTkFrame(left_panel, fg_color='#2b2b2b', corner_radius=10)
        btt_frame.grid(row = 2, column = 0, sticky='nsew', padx=(0, 10))

        btt_frame.grid_columnconfigure(0, weight = 1)
        btt_frame.grid_columnconfigure(1, weight = 1)
        btt_frame.grid_columnconfigure(2, weight = 1)

        # Browse button
        browse_btn = ctk.CTkButton(
            btt_frame,
            text='Browse Files',
            command=self._browse_images,
            height=40,
            font=('Arial', 14, 'bold')
        )
        browse_btn.grid(row=0, column=0, pady=(0, 15), padx=12, sticky='ew')

        capture_images = ctk.CTkButton(
            btt_frame, 
            text = 'Capture',
            height= 40,
            font =  ('Arial', 14, 'bold')
        )
        capture_images.grid(row = 0, column = 1, pady = (0, 15), padx = 12, sticky = 'ew')

        retrain_button = ctk.CTkButton(
            btt_frame,
            text = 'Re-Train',
            height = 40,
            font = ('Arial', 14, 'bold'),
            command = self._re_train
        )
        retrain_button.grid(row = 0, column = 2, pady = (0, 15), padx = 12, stick = 'ew')

        ## Right Panel - Labeling
        right_panel = ctk.CTkFrame(content_frame, fg_color='#2b2b2b', corner_radius=10, width = 200, height = 200)
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        
        # Configure right panel
        right_panel.grid_rowconfigure(0, weight=0)  # Title
        right_panel.grid_rowconfigure(1, weight=0)  # Label entry
        right_panel.grid_rowconfigure(2, weight=0)  # Buttons
        right_panel.grid_rowconfigure(3, weight=1)  # Spacer
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_columnconfigure(1, weight=1)

        # Title
        right_title = ctk.CTkLabel(
            right_panel, 
            text='Label Images', 
            font=('Arial', 16, 'bold')
        )
        right_title.grid(row=0, column=0, columnspan=2, pady=(15, 20), padx=15)

        # Label entry
        label_text = ctk.CTkLabel(
            right_panel, 
            text='Enter Label:', 
            font=('Arial', 14)
        )
        label_text.grid(row=1, column=0, padx=(15, 5), pady=(0, 10), sticky='e')
        
        self.label_entry_variable = ctk.StringVar()
        self.label_entry = ctk.CTkEntry(
            right_panel,
            textvariable=self.label_entry_variable,
            placeholder_text="e.g., Person Name",
            height=40,
            font=('Arial', 14)
        )
        self.label_entry.grid(row=1, column=1, padx=(5, 15), pady=(0, 10), sticky='w')

        # Buttons
        label_btn = ctk.CTkButton(
            right_panel,
            text='Label Images',
            height=40,
            command = self._label_func,
            font=('Arial', 14),
            fg_color='#1f6aa5',
            hover_color='#144870'
        )
        label_btn.grid(row=2, column=0, padx=(15, 5), pady=10, sticky='ew')
        
        delete_btn = ctk.CTkButton(
            right_panel,
            text='Delete Labels',
            height=40,
            font=('Arial', 14),
            fg_color='#d44040',
            hover_color='#9c2e2e'
        )
        delete_btn.grid(row=2, column=1, padx=(5, 15), pady=10, sticky='ew')

        ## Right Lower Frame
        right_lower_frame = ctk.CTkFrame(content_frame, fg_color='#2b2b2a', corner_radius=10, width = 100, height = 100)
        right_lower_frame.grid(row=1, column=1, sticky='nsew', padx=(10, 0), pady = 10)

        right_lower_frame.grid_columnconfigure(0, weight = 1)
        right_lower_frame.grid_columnconfigure(1, weight = 1)
        right_lower_frame.grid_columnconfigure(2, weight = 1)
        right_lower_frame.grid_columnconfigure(3, weight = 1)

        # Buttons, clear images browsed, clear all data, train, export
        export_btn = ctk.CTkButton(
            right_lower_frame,
            text = 'Export',
            fg_color = '#d44040',
            font = ('Arial', 14),
            hover_color = '#9c2e2e',
            command = self._export_data_base_embeddings
        )
        export_btn.grid(row = 0, column = 3, padx = 5, pady = 5)

        train_embeddings = ctk.CTkButton(
            right_lower_frame, 
            text = 'Train', 
            fg_color = 'green', 
            font=('Arial', 14), 
            hover_color='#2ed146',
            command = self._get_embeddings
        )
        train_embeddings.grid(row = 0, column = 0, padx = 5, pady = 5)

        clear_browsed_images = ctk.CTkButton(
            right_lower_frame, 
            text = 'Clear Images', 
            fg_color = '#d44040', 
            font=('Arial', 14), 
            hover_color='#9c2e2e',
            command = self._clear_browsed
            )
        clear_browsed_images.grid(row = 0, column = 1, padx = 5, pady = 5)

        clear_all = ctk.CTkButton(
            right_lower_frame, 
            text = 'Clear All', 
            fg_color = '#d44040', 
            font=('Arial', 14), 
            hover_color='#9c2e2e',
            command = self._clear_all
            )
        clear_all.grid(row = 0, column = 2, padx = 5, pady = 5)

    def _label_func(self):
        self.folders_textbox.configure(state='normal')
        if self.label_entry.get() and self.images:
            try:
                label_entry_flag = self.label_entry.get().strip()
                
                if label_entry_flag in self.data_gathered:
                    msg.showerror('Exist', 'The Label Name is already in the current queue\nChoose another', icon=msg.ERROR)
                    return

                if label_entry_flag in self.loaded_labels:
                    answer_flag = msg.askyesno('Re-Train', f'"{label_entry_flag}" exists in database. Add to retrain queue?', icon=msg.WARNING)
                    if not answer_flag:
                        return
                
                # FIX: Check if string is alphabetical excluding spaces
                if label_entry_flag.replace(' ', '').isalpha():
                    self.folders_textbox.delete('1.0', 'end')
                    self.data_gathered[label_entry_flag] = self.images
                    self.label_entry_variable.set('')
                    self.images = []
                    msg.showinfo('Success', 'Images were labeled successfully', icon=msg.INFO)
                else:
                    raise ValueError
            except ValueError:
                msg.showerror('Error', 'Only include alphabetical labels', icon=msg.ERROR)
        self.folders_textbox.configure(state='disabled')

    def _clear_all(self):
        flag_to_approve = msg.askyesnocancel('Are you Sure?', 'This procedure will delete everything in the current session', icon=msg.QUESTION)
        if flag_to_approve:
            try:
                self.data_gathered.clear()
                self.images.clear()
                self.labels.clear()
                self.embeddings.clear()
                self.images_folders.clear()
                
                self.folders_textbox.configure(state='normal')
                self.folders_textbox.delete('1.0', 'end')
                self.folders_textbox.configure(state='disabled')
                
                msg.showinfo('Success', 'Data was deleted successfully')
            except Exception as e:
                msg.showerror('Error', f'An Error Occurred: {e}', icon=msg.ERROR)
                
    def _browse_images(self):
        images_browsed = tfl.askopenfilenames()
        filtered_images_browsed = []
        for img in images_browsed:
            if img.endswith(('.jpg', '.png')):
                filtered_images_browsed.append(img)
        self.images = filtered_images_browsed

         # Enable textbox for editing
        self.folders_textbox.configure(state='normal')
        self.folders_textbox.delete("1.0", "end")
        
        if filtered_images_browsed:
            for img in filtered_images_browsed:
                self.folders_textbox.insert('1.0', '* ' + os.path.basename(img) + '\n')
        else:
            self.folders_textbox.insert("1.0", "No Files selected.\n\nClick 'Browse Files' to select image files.")
        
        # Make textbox read-only again
        self.folders_textbox.configure(state='disabled')
        
    def run(self):
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.window = ctk.CTk()
        self.window.geometry('900x650')
        self.window.title('Recognition System Trainer')
        self.window.minsize(800, 600)

        self.build_app(self.window)
        self.window.mainloop()

if __name__ == '__main__':
    app = TrainingApp()
    app.run()