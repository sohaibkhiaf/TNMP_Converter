import os

from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "static/generated"
MODEL_PATH = "checkpoints/G_final.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        def down(in_c, out_c, normalize=True):
            layers = [
                nn.Conv2d(in_c, out_c, 4, 2, 1)
            ]

            if normalize:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(nn.LeakyReLU(0.2))

            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]

            if dropout:
                layers.append(nn.Dropout(0.2))

            return nn.Sequential(*layers)

        # Encoder
        self.d1 = down(3, 64, normalize=False)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.d5 = down(512, 512)
        self.d6 = down(512, 512)
        self.d7 = down(512, 512, normalize=False)
        self.d8 = down(512, 512, normalize=False)

        # Decoder
        self.u1 = up(512, 512, dropout=True)
        self.u2 = up(1024, 512, dropout=True)
        self.u3 = up(1024, 512, dropout=True)
        self.u4 = up(1024, 512)
        self.u5 = up(1024, 256)
        self.u6 = up(512, 128)
        self.u7 = up(256, 64)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):

        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        # Decoder + Skip connections
        u1 = self.u1(d8)

        u2 = self.u2(
            torch.cat([u1, d7], 1)
        )

        u3 = self.u3(
            torch.cat([u2, d6], 1)
        )

        u4 = self.u4(
            torch.cat([u3, d5], 1)
        )

        u5 = self.u5(
            torch.cat([u4, d4], 1)
        )

        u6 = self.u6(
            torch.cat([u5, d3], 1)
        )

        u7 = self.u7(
            torch.cat([u6, d2], 1)
        )

        output = self.final(
            torch.cat([u7, d1], 1)
        )

        return self.tanh(output)


# ============================================================
# Load Generator
# ============================================================

G = Generator().to(device)

G.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

G.eval()

print("Generator loaded successfully.")
print("Device:", device)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    # Same size used during training
    image = image.resize((512, 512))

    image = tf.to_tensor(image)

    # Same normalization used during training
    image = tf.normalize(
        image,
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

    image = image.unsqueeze(0)

    return image.to(device)


# ============================================================
# Convert model output to normal-map image
# ============================================================

def tensor_to_image(tensor):

    tensor = tensor.squeeze(0)

    # [-1, 1] → [0, 1]
    tensor = tensor * 0.5 + 0.5

    tensor = tensor.clamp(0, 1)

    # CHW → HWC
    tensor = tensor.permute(1, 2, 0)

    tensor = tensor.cpu()

    # [0,1] → [0,255]
    tensor = (tensor * 255).byte()

    return Image.fromarray(tensor.numpy())


# ============================================================
# Home
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# Generate Normal Map
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():

    if "texture" not in request.files:

        return "No image uploaded", 400

    file = request.files["texture"]

    if file.filename == "":

        return "No file selected", 400

    input_path = os.path.join(
        UPLOAD_FOLDER,
        "texture.png"
    )

    output_path = os.path.join(
        GENERATED_FOLDER,
        "normal_map.png"
    )

    # Save uploaded texture
    file.save(input_path)

    # Open image
    image = Image.open(input_path)

    # Preprocess
    x = preprocess_image(image)

    # Generate
    with torch.no_grad():

        fake_normal = G(x)

    # Convert tensor → image
    normal_image = tensor_to_image(fake_normal)

    # Save
    normal_image.save(output_path)

    return render_template(
        "index.html",
        result="/static/generated/normal_map.png"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )


