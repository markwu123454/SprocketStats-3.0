from googleapiclient.discovery import build
import dotenv
import os
import json
import re
from tqdm import tqdm

dotenv.load_dotenv()
YT_API_KEY = os.environ["YT_API_KEY"]
youtube = build("youtube", "v3", developerKey=YT_API_KEY)


def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", url)
    return match.group(1) if match else None


def get_channel_name(video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    if response["items"]:
        return response["items"][0]["snippet"]["channelTitle"]
    return None


with open("matches.json", "r") as f:
    matches = json.load(f)

for match in tqdm(matches, desc="Processing matches", unit="match"):
    videos = match.get("videos", [])

    first_videos = []
    invalid_videos = []

    for url in videos:
        video_id = extract_video_id(url)
        channel = get_channel_name(video_id) if video_id else None

        if channel and "FIRST" in channel:
            first_videos.append(url)
        else:
            invalid_videos.append(url)

    match["videos"] = first_videos
    match["invalid_videos"] = invalid_videos

# Tally categories
no_video = 0
only_invalid = 0
one_valid_no_invalid = 0
one_valid_with_invalid = 0
multi_valid_no_invalid = 0
multi_valid_with_invalid = 0

for match in matches:
    valid = match.get("videos", [])
    invalid = match.get("invalid_videos", [])
    v, i = len(valid), len(invalid)

    if v == 0 and i == 0:
        no_video += 1
    elif v == 0 and i > 0:
        only_invalid += 1
    elif v == 1 and i == 0:
        one_valid_no_invalid += 1
    elif v == 1 and i > 0:
        one_valid_with_invalid += 1
    elif v > 1 and i == 0:
        multi_valid_no_invalid += 1
    elif v > 1 and i > 0:
        multi_valid_with_invalid += 1

with open("matches.json", "w") as f:
    json.dump(matches, f, indent=2)

print(f"No videos at all:                  {no_video}")
print(f"Only invalid videos:               {only_invalid}")
print(f"Exactly 1 valid, no invalid:       {one_valid_no_invalid}")
print(f"Exactly 1 valid, with invalid:     {one_valid_with_invalid}")
print(f"Multiple valid, no invalid:        {multi_valid_no_invalid}")
print(f"Multiple valid, with invalid:      {multi_valid_with_invalid}")
print("Done. matches.json updated.")