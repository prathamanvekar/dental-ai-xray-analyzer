import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from models_arch import CNNBiLSTM
from rule_engine import ClinicalRuleEngine

class DentalInferencePipeline:
    def __init__(self, models_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.models_dir = models_dir
        self.rule_engine = ClinicalRuleEngine()

        # Class mappings (MUST match training)
        self.JAW_CLASSES = ['Mandible', 'Maxilla']
        self.TOOTH_CLASSES = ['Canine', 'Incisor', 'Molar', 'Premolar']
        self.DISEASE_CLASSES = [
            'Caries', 'Cyst', 'Healthy', 'Impacted_Tooth',
            'Periapical_Lesion', 'Periodontal_Bone_Loss', 'Root_Canal_Treated'
        ]

        self.load_models()

    def load_models(self):
        print("Loading models...")

        # 1. Load YOLO (Object Detection)
        yolo_path = os.path.join(self.models_dir, 'best_yolo.pt')
        self.yolo_model = YOLO(yolo_path)

        # 2. Load Jaw Classifier
        self.jaw_model = self._load_classifier('jaw_model.pth', len(self.JAW_CLASSES))

        # 3. Load Tooth Type Classifier
        self.tooth_model = self._load_classifier('tooth_model.pth', len(self.TOOTH_CLASSES))

        # 4. Load Disease Classifier
        self.disease_model = self._load_classifier('disease_model.pth', len(self.DISEASE_CLASSES))

        print("✅ All models loaded successfully.")

    def _load_classifier(self, filename, num_classes):
        path = os.path.join(self.models_dir, filename)
        model = CNNBiLSTM(num_classes=num_classes)

        # Load weights (map location ensures CPU/GPU compatibility)
        checkpoint = torch.load(path, map_location=self.device)

        # Handle state dict structure (in case it was saved nested)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def preprocess_roi(self, roi_img):
        """
        Preprocess crop for CNN (Resize -> Tensor -> Normalize)
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(roi_img).unsqueeze(0).to(self.device)

    def enhance_image_cv2(self, img):
        """
        Lightweight enhancement using CLAHE (OpenCV)
        Replaces ESRGAN for speed in local deployment.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def analyze_image(self, image_path):
        """
        Full Pipeline: Load -> Enhance -> Detect -> Crop -> Classify -> Recommend
        """
        # 1. Load Image
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            return {"error": "Could not load image"}

        # 2. Enhancement
        enhanced_img = self.enhance_image_cv2(orig_img)

        # 3. YOLO Detection
        results = self.yolo_model(enhanced_img, conf=0.25)[0]

        findings = []

        # 4. Process Detections
        for box in results.boxes:
            # Get Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop ROI
            roi = enhanced_img[y1:y2, x1:x2]
            if roi.size == 0: continue

            roi_tensor = self.preprocess_roi(roi)

            # 5. Classifiers
            with torch.no_grad():
                # Jaw
                jaw_out = self.jaw_model(roi_tensor)
                jaw_idx = torch.argmax(jaw_out, dim=1).item()
                jaw_label = self.JAW_CLASSES[jaw_idx]

                # Tooth Type
                tooth_out = self.tooth_model(roi_tensor)
                tooth_idx = torch.argmax(tooth_out, dim=1).item()
                tooth_label = self.TOOTH_CLASSES[tooth_idx]

                # Disease
                dis_out = self.disease_model(roi_tensor)
                dis_probs = torch.softmax(dis_out, dim=1)
                dis_conf, dis_idx = torch.max(dis_probs, dim=1)
                dis_label = self.DISEASE_CLASSES[dis_idx.item()]
                dis_conf = dis_conf.item()

            # 6. Rule Engine
            finding_data = {
                'bbox': [x1, y1, x2, y2],
                'jaw': jaw_label,
                'tooth_type': tooth_label,
                'disease': dis_label,
                'confidence': dis_conf
            }

            recommendation = self.rule_engine.generate_recommendation(finding_data)
            finding_data['recommendation'] = recommendation

            findings.append(finding_data)

        return {
            "image": orig_img,
            "enhanced": enhanced_img,
            "findings": findings
        }
