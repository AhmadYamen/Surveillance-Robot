import os
import cv2 as cv
import numpy as np
import json 
import threading
import customtkinter as ctk
import tkinter.filedialog as tfl
import tkinter.messagebox as msg

try:
    import embedding_engine as emg
except ImportError:
    print("Warning: embedding_engine not found")
    emg = None


class TrainingApp:
    def __init__(self):
        self.labels = []
        self.images = []
        self.loaded_embeddings = []
        self.loaded_labels = []
        self.embeddings = []
        self.images_folders = []
        self.data_gathered = {}
        self.window = None
        self.progress_window = None
        
        # Create directories if they don't exist
        os.makedirs('images', exist_ok=True)
        os.makedirs('cropped', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Load face cascade
        cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        
        # Initialize embedding engine
        if emg:
            try:
                self.model = emg.FaceExtractorEngine()
            except Exception as e:
                print(f"Warning: Could not initialize face engine: {e}")
                self.model = None
        else:
            self.model = None
        
        self._load_data()

    def _crop_face(self, image):
        """Detects and crops the face from a BGR image."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Add 10% margin
        margin = int(w * 0.1)
        y1, y2 = max(0, y - margin), min(image.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(image.shape[1], x + w + margin)
        
        return image[y1:y2, x1:x2]

    def _geometric_median(self, points, eps=1e-5, max_iter=100):
        """Calculate geometric median of points"""
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

    def _calculate_encodings_best(self, images=None, embeddings=None):
        """Calculate the best embedding for a set of images or embeddings"""
        if embeddings is not None:
            embeddings = np.array(embeddings)
            median = self._geometric_median(embeddings)
            return median / (np.linalg.norm(median) + 1e-7)
        
        embeddings_per_label = []
        for img_path in images:
            try:
                img = cv.imread(img_path)
                if img is None:
                    continue
                face_img = self._crop_face(img)
                target_img = face_img if face_img is not None else img
                
                if self.model:
                    single_img_embedding = self.model.extractEmbedding(target_img)
                    embeddings_per_label.append(single_img_embedding)
            except Exception as e:
                print(f"Error embedding {img_path}: {e}")
                continue
        
        if not embeddings_per_label:
            raise ValueError("No valid embeddings for images")
        
        embeddings_per_label = np.array(embeddings_per_label)
        median = self._geometric_median(embeddings_per_label)
        
        return median / (np.linalg.norm(median) + 1e-7)

    def _get_embeddings(self):
        """Process embeddings for all labeled images"""
        if not self.data_gathered:
            msg.showerror('Training Error', 'No data to train', icon=msg.ERROR)
            return
        
        msg.showinfo("Training", "Processing embeddings... This may take a moment.")
        
        train_thread = threading.Thread(target=self._train_embeddings_thread)
        train_thread.daemon = True
        train_thread.start()

    def _train_embeddings_thread(self):
        """Thread for training embeddings"""
        self.embeddings = []
        self.labels = []
        
        for label, images in self.data_gathered.items():
            try:
                best_embed = self._calculate_encodings_best(images)
                self.embeddings.append(best_embed)
                self.labels.append(label)
                print(f"Processed label: {label} with {len(images)} images")
            except Exception as e:
                print(f"Error processing {label}: {e}")
                continue
        
        # Save to dataset
        self._export_data_base_embeddings()
        
        if self.window:
            self.window.after(0, lambda: msg.showinfo(
                "Training Complete",
                f"Successfully processed {len(self.labels)} labels"
            ))

    def _export_data_base_embeddings(self):
        """Export embeddings to dataset.json"""
        try:
            # Merge with existing data
            final_data_map = {}
            
            # Load existing
            for label, embed in zip(self.loaded_labels, self.loaded_embeddings):
                final_data_map[label] = embed.tolist() if hasattr(embed, 'tolist') else embed
            
            # Add new
            for label, embed in zip(self.labels, self.embeddings):
                final_data_map[label] = embed.tolist() if hasattr(embed, 'tolist') else embed
            
            # Convert to lists
            final_labels = list(final_data_map.keys())
            final_embeddings = list(final_data_map.values())
            
            data = {
                'Label': final_labels,
                'Embedding': final_embeddings,
                'Tag': [''] * len(final_labels)
            }
            
            with open('dataset.json', 'w') as dataset_file:
                json.dump(data, dataset_file, indent=2)
            
            print(f"Exported {len(final_labels)} face embeddings")
            
        except Exception as e:
            print(f"Export error: {e}")

    def _load_data(self):
        """Load existing dataset"""
        if os.path.exists('dataset.json'):
            try:
                with open('dataset.json', 'r') as dataset_file:
                    data = json.load(dataset_file)
                    if 'Embedding' in data and 'Label' in data:
                        self.loaded_embeddings = [np.array(embed, dtype=np.float32) for embed in data['Embedding']]
                        self.loaded_labels = data['Label']
                    else:
                        self.loaded_embeddings = []
                        self.loaded_labels = []
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.loaded_embeddings = []
                self.loaded_labels = []

    def _browse_images(self):
        """Browse and select image files"""
        images_browsed = tfl.askopenfilenames(
            title="Select Face Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        filtered = [img for img in images_browsed if img.endswith(('.jpg', '.png', '.jpeg'))]
        self.images = filtered
        
        # Update text display
        self.folders_textbox.configure(state='normal')
        self.folders_textbox.delete("1.0", "end")
        
        if filtered:
            for img in filtered:
                self.folders_textbox.insert('1.0', '* ' + os.path.basename(img) + '\n')
        else:
            self.folders_textbox.insert("1.0", "No files selected.\n\nClick 'Browse Images' to select.")
        
        self.folders_textbox.configure(state='disabled')

    def _label_func(self):
        """Label the browsed images"""
        self.folders_textbox.configure(state='normal')
        label_text = self.label_entry.get().strip()
        
        if label_text and self.images:
            if label_text in self.data_gathered:
                msg.showerror('Error', f'Label "{label_text}" already exists in current queue')
                return
            
            self.folders_textbox.delete('1.0', 'end')
            self.data_gathered[label_text] = self.images.copy()
            self.label_entry_variable.set('')
            self.images = []
            
            # Show current labels
            label_list = "\n".join([f"• {l} ({len(self.data_gathered[l])} images)" for l in self.data_gathered.keys()])
            self.folders_textbox.insert('1.0', f"Labeled:\n{label_list if label_list else 'None'}")
            msg.showinfo('Success', f'Labeled "{label_text}" with {len(self.data_gathered[label_text])} images')
        
        self.folders_textbox.configure(state='disabled')

    def _clear_all(self):
        """Clear all data in current session"""
        if msg.askyesno('Confirm', 'Delete all unsaved data in current session?'):
            self.data_gathered.clear()
            self.images.clear()
            self.labels.clear()
            self.embeddings.clear()
            
            self.folders_textbox.configure(state='normal')
            self.folders_textbox.delete('1.0', 'end')
            self.folders_textbox.insert("1.0", "Cleared.\n\nSelect new images to label.")
            self.folders_textbox.configure(state='disabled')
            
            msg.showinfo('Success', 'Session data cleared')

    def _clear_browsed(self):
        """Clear currently browsed images"""
        if msg.askyesno('Confirm', 'Clear unsaved images?'):
            self.images.clear()
            self.folders_textbox.configure(state='normal')
            self.folders_textbox.delete('1.0', 'end')
            self.folders_textbox.insert("1.0", "Cleared.\n\nBrowse new images.")
            self.folders_textbox.configure(state='disabled')

    def build_app(self, window_root):
        """Build the training application GUI"""
        window_root.title('Face Recognition Trainer')
        window_root.geometry('800x600')
        window_root.minsize(700, 500)
        
        # Configure grid
        window_root.grid_rowconfigure(0, weight=1)
        window_root.grid_columnconfigure(0, weight=1)
        
        # Main container
        main_frame = ctk.CTkFrame(window_root)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkLabel(main_frame, text="Face Recognition Trainer", font=('Arial', 24, 'bold'))
        header.grid(row=0, column=0, columnspan=2, pady=15)
        
        # Left panel: Image selection
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.grid(row=1, column=0, sticky='nsew', padx=(0, 5))
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(left_panel, text="Selected Images", font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=10)
        
        self.folders_textbox = ctk.CTkTextbox(left_panel, wrap='word')
        self.folders_textbox.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        self.folders_textbox.insert("1.0", "No files selected.\n\nClick 'Browse Images' to select.")
        self.folders_textbox.configure(state='disabled')
        
        browse_btn = ctk.CTkButton(left_panel, text="Browse Images", command=self._browse_images, height=40)
        browse_btn.grid(row=2, column=0, pady=10, padx=20, sticky='ew')
        
        # Right panel: Labeling
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.grid(row=1, column=1, sticky='nsew', padx=(5, 0))
        right_panel.grid_rowconfigure(0, weight=0)
        right_panel.grid_rowconfigure(1, weight=0)
        right_panel.grid_rowconfigure(2, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(right_panel, text="Label Management", font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=10)
        
        # Label input frame
        label_frame = ctk.CTkFrame(right_panel)
        label_frame.grid(row=1, column=0, pady=10, padx=20, sticky='ew')
        label_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(label_frame, text="Name:").grid(row=0, column=0, padx=5)
        self.label_entry_variable = ctk.StringVar()
        self.label_entry = ctk.CTkEntry(label_frame, textvariable=self.label_entry_variable)
        self.label_entry.grid(row=0, column=1, padx=5, sticky='ew')
        
        ctk.CTkButton(label_frame, text="Label Images", command=self._label_func, width=100).grid(row=0, column=2, padx=5, pady = 5)
        
        # Current labels display
        ctk.CTkLabel(right_panel, text="Current Labels Queue", font=('Arial', 14)).grid(row=2, column=0, pady=(10, 0))
        
        self.labels_display = ctk.CTkTextbox(right_panel, height=150, wrap='word')
        self.labels_display.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)
        self.labels_display.configure(state='normal')
        
        # Action buttons
        btn_frame = ctk.CTkFrame(right_panel)
        btn_frame.grid(row=4, column=0, pady=10, padx=20, sticky='ew')
        
        ctk.CTkButton(btn_frame, text="Train", command=self._get_embeddings, fg_color="green").pack(side='left', padx=5, expand=True, fill='x')
        ctk.CTkButton(btn_frame, text="Clear Images", command=self._clear_browsed, fg_color="orange").pack(side='left', padx=5, expand=True, fill='x')
        ctk.CTkButton(btn_frame, text="Clear All", command=self._clear_all, fg_color="red").pack(side='left', padx=5, expand=True, fill='x')
        
        self._update_labels_display()
        
        # Periodically update labels display
        def update_display():
            self._update_labels_display()
            window_root.after(1000, update_display)
        update_display()

    def _update_labels_display(self):
        """Update the labels display textbox"""
        self.labels_display.configure(state='normal')
        self.labels_display.delete('1.0', 'end')
        
        if self.data_gathered:
            for label, images in self.data_gathered.items():
                self.labels_display.insert('end', f"• {label}: {len(images)} images\n")
        else:
            self.labels_display.insert('1.0', "No labels added yet.\n\nLabel images and click 'Train' to process.")
        
        if self.loaded_labels:
            self.labels_display.insert('end', f"\n--- Existing ({len(self.loaded_labels)} labels) ---\n")
            for label in self.loaded_labels[:10]:  # Show first 10
                self.labels_display.insert('end', f"  {label}\n")
            if len(self.loaded_labels) > 10:
                self.labels_display.insert('end', f"  ... and {len(self.loaded_labels)-10} more\n")
        
        self.labels_display.configure(state='disabled')

    def run(self):
        """Run the training application"""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.window = ctk.CTk()
        self.build_app(self.window)
        
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.mainloop()

    def _on_close(self):
        """Handle window close"""
        self.window.destroy()


if __name__ == '__main__':
    app = TrainingApp()
    app.run()