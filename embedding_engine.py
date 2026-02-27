import cv2 as cv
import numpy as np

#import urllib.request
#import os

## check for the existance of the models

class FaceExtractorEngine:
	def __init__(self):
		self.face_net = cv.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/detector_model.caffemodel')
		self.embedding_net = cv.dnn.readNetFromTorch('models/openface.t7')

	def detectFaces(self, image, confidence_threshold = 0.5):
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
			swapRB = False,
			crop = False
		)

		self.face_net.setInput(blob)
		detections = self.face_net.forward()

		faces = []
		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i,2]

			if confidence > confidence_threshold:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(start_x, start_y, end_x, end_y) = box.astype('int')
				start_x, start_y = max(0, start_x), max(0, start_y)
				end_x, end_y = min(w, end_x), min(h, end_y)

				faces.append((start_x, start_y, end_x - start_x, end_y - start_y))
		return faces

	def extractEmbedding(self, face_image):
		"""
			Extract embedding from a single face image
		"""

		face_blob = cv.dnn.blobFromImage(
			face_image,
			1.0 / 255,
			(96, 96), # OpenFace (96x96)
			(0, 0, 0),
			swapRB = True,
			crop = False
		) 

		self.embedding_net.setInput(face_blob)
		embedding = self.embedding_net.forward()
		embedding = embedding.flatten()
		embedding = embedding / np.linalg.norm(embedding)

		return embedding

	def getFaceEmbeddings(self, image: str | np.ndarray, faces: list = None):
		"""
			Get Embeddings for all faces in an image
		"""
		if not type(image) == np.ndarray:
			image = cv.imread(image)
			if image is None:
				return []

		
		# Detect faces if not passed
		if faces is None:
			faces = self.detectFaces(image)

		embeddings = []
	
		for (x, y, w, h) in faces:
			# Extract face region
			face_region = image[y:y+h, x:x+w]

			if face_region.size > 0:
				# Get Embedding
				embedding = self.extractEmbedding(face_region)
				embeddings.append({
					'bbox' : (x, y, w, h),
					'embedding' : embedding,
					'face_image' : face_region
				})
		return embeddings