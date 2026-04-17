from ultralytics import YOLO
import torch

# STEP 2: The "Health Check" - This prints out if your 9070 XT is detected
print("--- HARDWARE CHECK ---")
if torch.cuda.is_available():
    print(f"Success. Found GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. The script will run on your CPU.")
print("----------------------\n")

# STEP 3: Load the Model
# 'yolo11x.pt' is the Extra Large version. It's the smartest but heaviest.
# It will download automatically the first time you run this.
model = YOLO('yolo11x.pt')

# STEP 4: Start the Training!
model.train(
    data='data.yaml',    # Image dir
    epochs=100,
    imgsz=640,
    batch=16,            # Batch size
    device=0,
    workers=8,
    project='my_yolo_project',
    name='version_1',    # The name of this specific training run
    exist_ok=True        # If you run it again, it won't crash; it just overwrites
)

print("\n--- TRAINING COMPLETE ---")
print("Result: my_yolo_project/version_1/weights/best.pt")