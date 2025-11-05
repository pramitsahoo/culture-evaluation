import json
import glob

# Define the filenames and the keys you want them under
files_and_keys = {
    "Architectures.json": "architectures",
    "Drinks.json": "drinks",
    "Places.json": "places",
    "Traditional Games.json": "traditional_games",
    "Arts.json": "arts",
    "Festivals.json": "festivals",
    "Religion.json": "religion",
    "Traditions.json": "traditions",
    "Clothing.json": "clothing",
    "Jwellery.json": "jewellery",
    "Rituals.json": "rituals",
    "Cuisine.json": "cuisine",
    "Languages and Dialects.json": "languages_and_dialects",
    "States and Capitals.json": "states_and_capitals",
    "Dance_Forms.json": "dance_forms",
    "Names.json": "names",
    "Textiles.json": "textiles"
}

merged = {}

for filename, key in files_and_keys.items():
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged[key] = data  # assuming each file's JSON is an array or an object as desired

with open("india_cultural_features.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
