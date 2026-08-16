from ultralytics import YOLO
import torch

print("--- HARDWARE CHECK ---")
if torch.cuda.is_available():
    print(f"Success. Found GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. The script will run on your CPU.")
print("----------------------\n")

# STEP 3: Load the Model
model = YOLO('yolo11x.pt')

# STEP 4: Start the Training!
model.train(
    data='data.yaml',    
    epochs=100,
    imgsz=720,
    batch=32,          
    device=0,
    workers=8,
    project='my_yolo_project',
    name='version_1_frc',
    exist_ok=True,
    patience=25,

    # --- AUGMENTATION SETTINGS ---
    mosaic=1.0,          # 1.0 = On for every image. Combines 4 images into 1 grid.
    close_mosaic=10,     # Disables mosaic for the last 10 epochs for "clean" fine-tuning.
    
    # HSV (Color Augmentation)
    hsv_h=0.015,         # Adjusts Hue (helps with different red/blue shades in stadiums)
    hsv_s=0.7,           # Adjusts Saturation (helps with washed out or vivid colors)
    hsv_v=0.4,           # Adjusts Value/Brightness (helps with shadows on the field)
    
    # Extra FRC-friendly boosts
    degrees=10.0,        # Slight rotation for tilted phone/camera angles
    flipud=0.0,          # Robots are never upside down; keep this at 0.0
    fliplr=0.5           # Mirroring is fine; a robot can drive left or right
)

print("\n--- TRAINING COMPLETE ---")
print("Your best FRC model is located at: my_yolo_project/version_1_frc/weights/best.pt")
