import dotenv
import os

from ls_ext import LabelStudioClient

dotenv.load_dotenv()

client = LabelStudioClient(base_url="https://app.humansignal.com", api_key=os.environ["LS_API_KEY"])
PROJECT_ID = 260156  # from your config.yaml

# find tasks with a "label" (polygon) result present
for task in client.tasks.list(project=PROJECT_ID):
    for ann in task.annotations:
        results = ann["result"]
        has_polygon = any(r["from_name"] == "label" for r in results)
        if has_polygon:
            # keep only the kp results, drop the polygon ones
            cleaned = [r for r in results if r["from_name"] != "label"]
            #client.annotations.update(id=ann["id"], result=cleaned)
            print(f"cleaned task {task.id}, annotation {ann['id']}")
