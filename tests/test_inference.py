import unittest
import os
import base64

from fastapi.testclient import TestClient

from src.api.config import settings

from tests.config import current_dir

class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        settings.testing = True

    def test_inference(self):

        from src.api.main import app
        client = TestClient(app)

        # health check
        
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

        # no predict type provided
        response = client.post("/generate", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Either 'image' or 'name' must be provided."})

        image_path = os.path.join(current_dir, 'files/sample.jpg')

        # Load and encode test image
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode()
        
        # normal name generation, but no image provided

        payload = {
            "type": "name",
            "name": "Test Name"
        }
        response = client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Image must be provided for name generation."})

        # normal name generation

        payload = {
            "image": encoded_image,
            "diversity": 1.0,
            "min_name_length": 2
        }
        response = client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("name", response.json())
        self.assertTrue(len(response.json()["name"].split()) >= 2)
        self.assertIsInstance(response.json()["name"], str)
        print(response.json()["name"])

        # normal bio generation, but no name provided

        payload = {
            "type": "bio",
            "image": encoded_image
        }

        response = client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Name must be provided for bio generation."})

        # normal bio generation

        payload = {
            "type": "bio",
            "name": "John Smith",
            "diversity": 1.0,
            "max_bio_length": 200,
            "nsfw_on": False
        }
        response = client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("bio", response.json())
        self.assertIsInstance(response.json()["bio"], str)
        print(response.json()["bio"])