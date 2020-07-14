# create right format for BLINK input data
import json
import pdb

all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
print("Parsed wikipedia")
all_wiki_ent_id_to_ents = {line['kb_idx']: i for i, line in enumerate(all_wiki_ents) if 'kb_idx' in line}
print("Created wikidata ID -> idx in saved entities map")

skipped_none = []
skipped_not_inwiki = []
for split in ["train", "test"]:
    print(split)
    with open(
        "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{0}.entities.all_pos.filtered_on_all.no_partials.json".format(
            split
        )
    ) as f:
        f_parsed = json.load(f)
    wf = open("/private/home/belindali/starsem2018-entity-linking/data/graphquestions_all_ents/{0}.jsonl".format(split), "w")
    for i, entry in enumerate(f_parsed):
        assert len(entry["entities_pos"]) == len(entry["entities"])
        new_entity_pos = []
        new_entities = []
        for e, entity_pos in enumerate(entry["entities_pos"]):
            if entity_pos[0] == -1:
                import pdb
                pdb.set_trace()
                print(str(i) + "\t" + entry['utterance'] + "\t" + entry["entities"])
                confirmed = "n"
                while confirmed != "y":
                    mention = input("Copy-paste mention: ")
                    f_parsed[i]["entities_pos"][e][0] = entry['utterance'].find(mention)
                    f_parsed[i]["entities_pos"][e][1] = entity_pos[0] + len(mention)
                    print(entry['utterance'][
                        :f_parsed[i]["entities_pos"][e][0]
                    ] + "[" + entry['utterance'][
                        f_parsed[i]["entities_pos"][e][0]:f_parsed[i]["entities_pos"][e][1]
                    ] + "]" + entry['utterance'][
                        f_parsed[i]["entities_pos"][e][1]:
                    ])
                    confirmed = input("Confirm? y/[n]: ")
                if len(f_parsed[i]["entities_pos"]) > 1:
                    for j in range(len(f_parsed[i]["entities_pos"])):
                        if f_parsed[i]["entities_pos"][j][0] == -1:
                            break
                    f_parsed[i]["entities_pos"][j] = f_parsed[i]["entities_pos"][e]
                else:
                    f_parsed[i]["entities_pos"][0] = f_parsed[i]["entities_pos"][e]
            if entry["entities"][e] is None:
                skipped_none.append(entry['question_id'])
                continue
            if entry["entities"][e] not in all_wiki_ent_id_to_ents:
                skipped_not_inwiki.append(entry['question_id'])
                continue
            new_entity_pos.append(f_parsed[i]["entities_pos"][e])
            new_entities.append(f_parsed[i]["entities"][e])
        entry_new_format = {
            'question_id': entry['question_id'],
            'mention': [entry['utterance'][
                new_entity_pos[e][0]:new_entity_pos[e][1]
            ] for e in range(len(new_entity_pos))],
            'context_left': [entry['utterance'][:new_entity_pos[e][0]] for e in range(len(new_entity_pos))],
            'context_right': [entry['utterance'][new_entity_pos[e][1]:] for e in range(len(new_entity_pos))],
            'label': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['text'] for e in range(len(new_entity_pos))],
            'title': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['title'] for e in range(len(new_entity_pos))],
            'entity': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['entity'] for e in range(len(new_entity_pos))],
            'label_id': [all_wiki_ent_id_to_ents[new_entities[e]] for e in range(len(new_entity_pos))],
            'wikidata_id': [new_entities[e] for e in range(len(new_entity_pos))],
        }
        if len(new_entity_pos) > 1:
            print(entry_new_format["entity"])
            # print(entry_new_format["mention"])
            for i in range(len(new_entity_pos)):
                if "refer to" in entry_new_format["label"][i]:
                    print(entry_new_format["label"][i])
        b = wf.write(json.dumps(entry_new_format) + "\n")
    # json.dump(f_parsed, open(
    #     "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/webqsp.{0}.entities{1}.all_pos.filtered_on_all.json".format(
    #         split, ".with_classes" if split == "test" else ""
    #     ), "w"
    # ))
print(skipped_none)
print(skipped_not_inwiki)


def parse_webqsp_all_entities():
    train_ids = json.load(open(
        "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.train.ids.json"
    ))
    dev_ids = json.load(open(
        "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.dev.ids.json"
    ))
    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    skipped_none = []
    skipped_not_inwiki = []
    for split in ["train", "dev", "test"]:
        print(split)
        with open(
            "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/webqsp.{0}.entities{1}.all_pos.filtered_on_all.json".format(
                split, ".with_classes" if split == "test" else ""
            )
        ) as f:
            f_parsed = json.load(f)
        wf = open("/private/home/belindali/starsem2018-entity-linking/data/WebQSP_all_ents/{0}.jsonl".format(split), "w")
        for i, entry in enumerate(f_parsed):
            if split == "train" and entry["question_id"] not in train_ids:
                continue
            if split == "dev" and entry["question_id"] not in dev_ids:
                continue
            assert len(entry["entities_pos"]) == len(entry["entities"])
            new_entity_pos = []
            new_entities = []
            for e, entity_pos in enumerate(entry["entities_pos"]):
                if entity_pos[0] == -1:
                    import pdb
                    pdb.set_trace()
                    print(str(i) + "\t" + entry['utterance'] + "\t" + entry["entities"])
                    confirmed = "n"
                    while confirmed != "y":
                        mention = input("Copy-paste mention: ")
                        f_parsed[i]["entities_pos"][e][0] = entry['utterance'].find(mention)
                        f_parsed[i]["entities_pos"][e][1] = entity_pos[0] + len(mention)
                        print(entry['utterance'][
                            :f_parsed[i]["entities_pos"][e][0]
                        ] + "[" + entry['utterance'][
                            f_parsed[i]["entities_pos"][e][0]:f_parsed[i]["entities_pos"][e][1]
                        ] + "]" + entry['utterance'][
                            f_parsed[i]["entities_pos"][e][1]:
                        ])
                        confirmed = input("Confirm? y/[n]: ")
                    if len(f_parsed[i]["entities_pos"]) > 1:
                        for j in range(len(f_parsed[i]["entities_pos"])):
                            if f_parsed[i]["entities_pos"][j][0] == -1:
                                break
                        f_parsed[i]["entities_pos"][j] = f_parsed[i]["entities_pos"][e]
                    else:
                        f_parsed[i]["entities_pos"][0] = f_parsed[i]["entities_pos"][e]
                if entry["entities"][e] is None:
                    skipped_none.append(entry['question_id'])
                    continue
                if entry["entities"][e] not in all_wiki_ent_id_to_ents:
                    skipped_not_inwiki.append(entry['question_id'])
                    continue
                new_entity_pos.append(f_parsed[i]["entities_pos"][e])
                new_entities.append(f_parsed[i]["entities"][e])
            entry_new_format = {
                'question_id': entry['question_id'],
                'mention': [entry['utterance'][
                    new_entity_pos[e][0]:new_entity_pos[e][1]
                ] for e in range(len(new_entity_pos))],
                'context_left': [entry['utterance'][:new_entity_pos[e][0]] for e in range(len(new_entity_pos))],
                'context_right': [entry['utterance'][new_entity_pos[e][1]:] for e in range(len(new_entity_pos))],
                'label': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['text'] for e in range(len(new_entity_pos))],
                'title': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['title'] for e in range(len(new_entity_pos))],
                'entity': [all_wiki_ents[all_wiki_ent_id_to_ents[new_entities[e]]]['entity'] for e in range(len(new_entity_pos))],
                'label_id': [all_wiki_ent_id_to_ents[new_entities[e]] for e in range(len(new_entity_pos))],
                'wikidata_id': [new_entities[e] for e in range(len(new_entity_pos))],
            }
            if len(new_entity_pos) > 1:
                print(entry_new_format["entity"])
                # print(entry_new_format["mention"])
                for i in range(len(new_entity_pos)):
                    if "refer to" in entry_new_format["label"][i]:
                        print(entry_new_format["label"][i])
                        import pdb
                        pdb.set_trace()
            b = wf.write(json.dumps(entry_new_format) + "\n")
        # json.dump(f_parsed, open(
        #     "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/webqsp.{0}.entities{1}.all_pos.filtered_on_all.json".format(
        #         split, ".with_classes" if split == "test" else ""
        #     ), "w"
        # ))
    print(skipped_none)
    print(skipped_not_inwiki)


def parse_webqsp_main_entity():
    skipped_none = []
    skipped_not_inwiki = []

    for set in ["train", "test"]:
        with open(
            "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{0}.entities.all_pos.filtered_on_all.no_partials.json".format(
                set,
            )
        ) as f:
            f_parsed = json.load(f)
        wf = open("/private/home/belindali/starsem2018-entity-linking/data/graphquestions/{0}.jsonl".format(set), "w")
        for i, entry in enumerate(f_parsed):
            if entry["main_entity_pos"][0] == -1:
                print(str(i) + "\t" + entry['utterance'] + "\t" + entry["main_entity_tokens"])
                confirmed = "n"
                while confirmed != "y":
                    mention = input("Copy-paste mention: ")
                    f_parsed[i]["main_entity_pos"][0] = entry['utterance'].find(mention)
                    f_parsed[i]["main_entity_pos"][1] = f_parsed[i]["main_entity_pos"][0] + len(mention)
                    print(entry['utterance'][
                        :f_parsed[i]["main_entity_pos"][0]
                    ] + "[" + entry['utterance'][
                        f_parsed[i]["main_entity_pos"][0]:f_parsed[i]["main_entity_pos"][1]
                    ] + "]" + entry['utterance'][
                        f_parsed[i]["main_entity_pos"][1]:
                    ])
                    confirmed = input("Confirm? y/[n]: ")
                if len(f_parsed[i]["entities_pos"]) > 1:
                    for j in range(len(f_parsed[i]["entities_pos"])):
                        if f_parsed[i]["entities_pos"][j][0] == -1:
                            break
                    f_parsed[i]["entities_pos"][j] = f_parsed[i]["main_entity_pos"]
                else:
                    f_parsed[i]["entities_pos"][0] = f_parsed[i]["main_entity_pos"]
            if entry["main_entity"] is None:
                skipped_none.append(entry['question_id'])
                continue
            if entry["main_entity"] not in all_wiki_ent_id_to_ents:
                skipped_not_inwiki.append(entry['question_id'])
                continue
            entry_new_format = {
                'question_id': entry['question_id'],
                'mention': entry['main_entity_tokens'],
                'context_left': entry['utterance'][:entry["main_entity_pos"][0]],
                'context_right': entry['utterance'][entry["main_entity_pos"][1]:],
                'label': all_wiki_ents[all_wiki_ent_id_to_ents[entry["main_entity"]]]['text'],
                'title': all_wiki_ents[all_wiki_ent_id_to_ents[entry["main_entity"]]]['title'],
                'entity': all_wiki_ents[all_wiki_ent_id_to_ents[entry["main_entity"]]]['entity'],
                'label_id': all_wiki_ent_id_to_ents[entry["main_entity"]],
                'wikidata_id': entry["main_entity"],
            }
            try:
                assert entry['main_entity_tokens'].replace(' ', '') == entry['utterance'][
                    entry["main_entity_pos"][0]:entry["main_entity_pos"][1]
                ].replace(' ', '')
            except:
                pdb.set_trace()
            b=wf.write(json.dumps(entry_new_format) + "\n")
        json.dump(f_parsed, open(
            "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/graph.{0}.entities.all_pos.filtered_on_all.json".format(
                set,
            ), "w"
        ))

    print(skipped_none)
    print(skipped_not_inwiki)

parse_webqsp()