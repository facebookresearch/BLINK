import json
import pdb


all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_ent_id_to_ents = {line['kb_idx']: line['title'] for line in all_wiki_ents if 'kb_idx' in line}
print("Created wikipedia ID -> title map")

with open("scripts/temp2.txt", "w") as wf:
    with open("scripts/temp.txt") as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip().split("    ")
            if line[0] in all_wiki_ent_id_to_ents:
                line[0] = all_wiki_ent_id_to_ents[line[0]]
            new_lines.append("    ".join(line))
            try:
                if line[0] == "":
                    a=wf.write("    ".join(line) + "\n")
                    continue
                pos_start = line[1].find("[") + 1
                pos_end = line[1].find("]")
                if line[0].lower() != line[1][pos_start:pos_end].lower():
                    a=wf.write("    ".join(line) + "\n")
            except:
                print(line)
