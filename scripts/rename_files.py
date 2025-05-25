import os

# Pokemon name mapping
pokemon_names = {
    # Indigo League Pokemon
    "1": "bulbasaur",
    "2": "ivysaur",
    "4": "charmander",
    "7": "squirtle",
    "8": "wartortle",
    "10": "caterpie",
    "11": "metapod",
    "12": "butterfree",
    "17": "pidgeotto",
    "20": "raticate",
    "22": "fearow",
    "24": "arbok",
    "25": "pikachu",
    "26": "raichu",
    "27": "sandshrew",
    "32": "nidoran♂",
    "35": "clefairy",
    "37": "vulpix",
    "39": "jigglypuff",
    "44": "gloom",
    "45": "vileplume",
    "46": "paras",
    "48": "venonat",
    "50": "diglett",
    "54": "psyduck",
    "57": "primeape",
    "58": "growlithe",
    "59": "arcanine",
    "63": "abra",
    "65": "alakazam",
    "69": "bellsprout",
    "74": "geodude",
    "77": "ponyta",
    "80": "slowbro",
    "81": "magnemite",
    "83": "farfetchd",
    "86": "seel",
    "91": "cloyster",
    "92": "gastly",
    "93": "haunter",
    "94": "gengar",
    "95": "onix",
    "98": "krabby",
    "102": "exeggcute",
    "104": "cubone",
    "107": "hitmonchan",
    "109": "koffing",
    "115": "kangaskhan",
    "116": "horsea",
    "119": "seaking",
    "122": "mr. mime",
    "123": "scyther",
    "124": "jynx",
    "126": "magmar",
    "129": "magikarp",
    "132": "ditto",
    "133": "eevee",
    "141": "kabutops",
    "142": "aerodactyl",
    "143": "snorlax",
    "146": "moltres",
    
    # Johto Journeys Pokemon
    "152": "chikorita",
    "155": "cyndaquil",
    "158": "totodile",
    "161": "sentret",
    "165": "ledyba",
    "167": "spinarak",
    "170": "chinchou",
    "173": "igglybuff",
    "174": "igglybuff",
    "175": "togepi",
    "179": "mareep",
    "183": "marill",
    "187": "hoppip",
    "191": "sunkern",
    "192": "sunflora",
    "193": "yanma",
    "194": "wooper",
    "195": "quagsire",
    "198": "murkrow",
    "200": "misdreavus",
    "203": "girafarig",
    "204": "pineco",
    "207": "gligar",
    "209": "snubbull",
    "211": "qwilfish",
    "213": "shuckle",
    "214": "heracross",
    "215": "sneasel",
    "216": "teddiursa",
    "218": "slugma",
    "220": "swinub",
    "222": "corsola",
    "223": "remoraid",
    "224": "octillery",
    "226": "mantine",
    "228": "houndour",
    "229": "houndoom",
    "231": "phanpy",
    "232": "donphan",
    "233": "porygon2",
    "234": "stantler",
    "235": "smeargle",
    "236": "tyrogue",
    "237": "hitmontop",
    "238": "smoochum",
    "239": "elekid",
    "240": "magby",
    "241": "miltank",
    "242": "blissey",
    "243": "raikou",
    "244": "entei",
    "245": "suicune",
    "246": "larvitar",
    "247": "pupitar",
    "248": "tyranitar",
    "249": "lugia",
    "250": "ho-oh",
    "251": "celebi"
}

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Get the base name without extension
            base_name = filename[:-4]  # Remove .png
            
            # Check if the filename is just a number
            if base_name.isdigit():
                # If it's a number, look up the Pokemon name
                if base_name in pokemon_names:
                    new_name = f"{base_name.zfill(3)}-{pokemon_names[base_name]}.png"
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_name)
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_name}")
                    except Exception as e:
                        print(f"Error renaming {filename}: {str(e)}")
            else:
                # If it's not a number, try to find the Pokemon by name
                base_name = base_name.split('-')[0]  # Remove any -johto suffix
                
                # Find the Pokemon by name
                for dex_num, name in pokemon_names.items():
                    if name == base_name.lower():
                        # Create new filename with format: "001-bulbasaur.png"
                        new_name = f"{dex_num.zfill(3)}-{name}.png"
                        old_path = os.path.join(directory, filename)
                        new_path = os.path.join(directory, new_name)
                        
                        try:
                            os.rename(old_path, new_path)
                            print(f"Renamed: {filename} -> {new_name}")
                        except Exception as e:
                            print(f"Error renaming {filename}: {str(e)}")
                        break

# Rename files in all directories
print("Renaming Indigo League silhouette files...")
rename_files('indigo-league/silhouettes')

print("\nRenaming Indigo League reveal files...")
rename_files('indigo-league/reveals')

print("\nRenaming Johto Journeys files...")
rename_files('johto-journeys')

print("\nRenaming Johto League Champions files...")
rename_files('johto-league-champions')

print("\nRename complete!")