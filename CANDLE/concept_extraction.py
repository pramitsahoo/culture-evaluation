import json

# Input data as a string
# data = """saree (24) • women (10) • sari (10) • wedding (7) • kurti (6) • cloth (6) • dress (6) • fashion (6) • lehenga (5) • wear (5) • salwar kameez (4) • worn (4) • skirt (3) • bridal (3) • sherwani (3) • day (3) • textile (3) • bride (2) • lehenga choli (2) • men (2) • dhoti (2) • culture (2) • city (2) • garment (2) • grace (2) • salwar (2) • mother (2) • woman (2) • designer (2) • include (2) • western (2) • red (2) • pakistani (2) • suit (2) • uniform (2) • traditional (2) • costume (2) • dress code (2) • known (2) • cotton (2) • ethnic wear (1) • print skirts (1) • kurta (1) • pajama (1) • red wedding (1) • dupatta (1) • explore (1) • fashion also explores (1) • also explores feminism (1) • female (1) • style (1) • tee (1) • part (1) • trousseau (1) • top (1) • bottom (1) • like traditional (1) • kurta pajama set (1) • pajama set includes (1) • wash (1) • score (1) • festivity (1) • women enjoyed (1) • saree clad (1) • festivities across various (1) • across various cities (1) • odisha (1) • summation (1) • new logo (1) • hockey team jersey (1) • national hockey team (1) • legging (1) • wardrobe (1) • lend grace (1) • buying (1) • men spend lavishly (1) • sherwani suit (1) • special occasion (1) • seersucker (1) • organized (1) • sweater (1) • parade (1) • show (1) • embroidery (1) • heavy (1) • loved (1) • every (1) • age (1) • ethnic jacquard sarees (1) • silk (1) • apparel (1) • epitome (1) • symbol (1) • lungi (1) • look (1) • tikka (1) • maang (1) • girl (1) • anklet (1) • tree (1) • color (1) • diwali (1) • printed (1) • veshti (1) • mehndi (1) • people wear new (1) • diwali festival (1) • gift (1) • share gifts (1) • manyavar (1) • half saree (1) • saree among (1) • chiffon sari (1) • blazer (1) • room wearing (1) • every player (1) • story like (1) • shalwar kameez (1) • trend (1) • yard (1) • inseparable (1) • turmeric (1) • time (1) • test (1) • coat (1) • japanese fireman (1) • chinese dragon robe (1) • garments range (1) • hang (1) • mumbai (1) • washed clothes (1) • dhobi hangs (1) • mahalaxmi dhobi ghat (1) • party (1) • occasion (1) • made (1) • patiala salwar (1) • attire (1) • biba (1) • brand (1) • ethnic fashion (1) • bohemian (1) • khadi (1) • jersey (1) • student (1) • apparel industry (1) • program (1) • aim (1) • employability (1) • preparing students (1) • white (1) • mangalsutra (1) • kimono (1) • rina dhaka (1) • celebrity (1) • dressed (1) • decade (1) • international celebrities (1) • three decades (1) • draping (1) • poncho (1) • summer (1) • scarf (1) • significant place (1) • traditional value (1) • early (1) • take oil bath (1) • wear new clothes (1) • wonderful dresses (1) • play (1) • future (1) • floral (1) • print (1) • chief (1) • warrior (1) • formerly (1) • head dress (1) • fusion wear (1) • weave (1) • great diversity (1) • term (1) • material (1) • also (1) • tradition (1) • school (1) • party wear sarees (1) • head massage (1) • continue (1) • new attire (1) • several days (1) • generally tend (1) • inspired (1) • punjabi suits (1) • outfit (1) • fabric (1) • wide even (1) • ancient trade (1) • textiles spread far (1) • beauty (1) • modern (1) • famous (1) • high demand (1) • finely created textiles (1) • knitwear capital (1) • tiruppur (1)"""
data = """  
tea (34) • milk (12) • gooseberry (9) • chai (9) • coffee (6) • china (6) • wine (5) • lassi (4) • yogurt (4) • assam (3) • vitamin c (3) • amla (3) • masala chai (3) • lemon (3) • cardamom (3) black tea (3) • alcohol (3) • green tea (3) • juice (2) • summer (2) • antioxidant (2) • pistachio (2) • pale ale (2) • served (2) • paneer (2) • whisky (2) • world (2) • diet (2) • ginger (2) • pomegranate (2) • country (2) • darjeeling (2) • drinking water (2) • aam panna (1) • mango lassi (1) • wedding (1) • tulsi (1) • honey (1) • dahi vada (1) • recipe (1) • blend (1) • carbonated drinks (1) • displayed (1) • year (1) • marked preference (1) • recent years (1) • liquor (1) • golden milk (1) • almond (1) • ice cream (1) • cup (1) • cart (1) • chai carts (1) • little clay cups (1) • bhang (1) • buttermilk (1) • gewurztraminer (1) • rose water (1) • nilgiri (1) • apple (1) • jalebi (1) • clove (1) • toddy (1) • region (1) • raita (1) • cumin (1) • filter coffee (1) • party (1) • masala tea (1) • rum (1) • old monk (1) • ayurvedic (1) • gourd (1) • cheese (1) • marinated (1) • lamb (1) • paper (1) • expression papers (1) • developed (1) • england (1) • colonization (1) • journey (1) • long (1) • withstand (1) • flower (1) • alcoholic drink (1) • used (1) • tropical (1) • bar (1) • turmeric (1) • holi (1) • plum (1) • flute (1) • vitamin (1) • powerhouse (1) • nutrient (1) • undeniably (1) • season (1) • rose milk (1) • south (1) • espresso (1) • river (1) • mango (1) • curd (1) • pumpkin (1) • fenugreek (1) • kheer (1) • hot (1) • coconut milk (1) • chai latte (1) • dahi (1) • cucumber (1) • lime (1) • thali (1) • fruit (1) • tamarind (1) • beer (1) • yoghurt (1) • coconut water (1) • source (1) • ginseng (1) • ashwagandha (1) • largest (1) • chocolate (1) • chaat (1) • mint (1) • arabica (1) • robusta (1) • variety (1) • samosa (1) • basil (1) • grape (1) • abuse (1) • drug (1) • red (1) • biryani (1) • vegetarian (1) • coconut (1) • grow (1) • shade (1) • guava (1) • snack (1) • sri lanka (1) • beverage (1) • popular (1) • festival (1) • grown (1) • city (1) • bangalore (1) • japan (1) • thailand (1) • pakistan (1) • indus (1) • water quality (1) • british (1) • shared (1) • bangladesh (1)
"""
# Parse data into a list of dictionaries
concepts = []
for item in data.split("•"):
    name, freq = item.strip().rsplit("(", 1)
    concepts.append({"name": name.strip(), "count": int(freq.strip(")"))})

# Count total concepts
total_concepts = len(concepts)

# Create the final JSON structure
final_data = [
    {
        "country": "India",
        "facet": "drink",
        "concepts": concepts
    }
]

# Save to JSON file
output_file = "concepts_drink.json"
with open(output_file, "w") as f:
    json.dump(final_data, f, indent=4)

print(f"Total number of concepts: {total_concepts}")
print(f"Data saved to {output_file}.")
