import cv2 as cv
import numpy as np

class FaceExtractorEngine:
	def __init__(self):
		self.face_net = cv.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/detector_model.caffemodel')
		self.embedding_net = cv.dnn.readNetFromTorch('models/openface.t7')

	def detectFaces(self, image, confidence_threshold=0.5):
		"""
			Detect faces in a single image
			Returns: list of face regions (x, y, w, h)
		"""
		(h, w) = image.shape[:2]

		blob = cv.dnn.blobFromImage(
			cv.resize(image, (300, 300)),
			1.0, 
			(300, 300),
			(104.0, 177.0, 123.0),
			swapRB=False,
			crop=False
		)

		self.face_net.setInput(blob)
		detections = self.face_net.forward()

		faces = []
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > confidence_threshold:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(start_x, start_y, end_x, end_y) = box.astype('int')
				start_x, start_y = max(0, start_x), max(0, start_y)
				end_x, end_y = min(w, end_x), min(h, end_y)

				# Only add if dimensions are valid
				if end_x > start_x and end_y > start_y:
					faces.append((start_x, start_y, end_x - start_x, end_y - start_y))
		return faces

	def extractEmbedding(self, face_image):
		"""
			Extract embedding from a single face image
		"""
		# Resize face to 96x96 for OpenFace
		if face_image.shape[0] != 96 or face_image.shape[1] != 96:
			face_image = cv.resize(face_image, (96, 96))
		
		face_blob = cv.dnn.blobFromImage(
			face_image,
			1.0 / 255,
			(96, 96),
			(0, 0, 0),
			swapRB=True,
			crop=False
		) 

		self.embedding_net.setInput(face_blob)
		embedding = self.embedding_net.forward()
		embedding = embedding.flatten()
		embedding = embedding / (np.linalg.norm(embedding) + 1e-7)

		return embedding

	def getFaceEmbeddings(self, image, faces=None):
		"""
			Get Embeddings for all faces in an image.\n
			Return:\n
				bbox : (x, y, w, h)
				embedding : embedding
				face_image : face cropped
		"""
		if not isinstance(image, np.ndarray):
			image = cv.imread(image)
			if image is None:
				return []

		# Detect faces if not passed
		if faces is None:
			faces = self.detectFaces(image)

		embeddings = []
		h_img, w_img = image.shape[:2]
		
		for (x, y, w, h) in faces:
			# Validate coordinates
			x = max(0, min(int(x), w_img - 1))
			y = max(0, min(int(y), h_img - 1))
			w = min(int(w), w_img - x)
			h = min(int(h), h_img - y)
			
			if w <= 0 or h <= 0:
				continue
				
			# Extract face region
			face_region = image[y:y+h, x:x+w].copy()

			if face_region.size > 0:
				try:
					# Get Embedding
					embedding = self.extractEmbedding(face_region)
					embeddings.append({
						'bbox': (x, y, w, h),
						'embedding': embedding,
						'face_image': face_region
					})
				except Exception as e:
					print(f"Error extracting embedding: {e}")
					continue
					
		return embeddings