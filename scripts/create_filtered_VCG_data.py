import json
import os

graphqs = True
if graphqs:
    sn = "graphquestions"
    prefix = "graph"
    subsets = ["train", "test"]
else:
    sn = "WebQSP"
    prefix = "webqsp"
    subsets = ["train", "dev", "test"]


all_wiki_ents = open("/private/home/belindali/temp/BLINK-Internal/models/entity.jsonl").readlines()
print("Loaded wikipedia")
all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
all_wiki_ent_ids = set([line['kb_idx'] for line in all_wiki_ents if 'kb_idx' in line])
print("Parsed wikipedia IDs")

for subset in subsets:
    '''
    UNFILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.dev.entities.all_pos.json"
    FILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.dev.entities.all_pos.filtered_on_main.json"
    FILTER_ON_MAIN = True
    #'''
    if subset == "test" and not graphqs:
        with_classes = ".with_classes"
    else:
        with_classes = ""
    #'''
    UNFILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/{}/input/{}.{}.entities{}.all_pos.json".format(
        sn, prefix, subset, with_classes)
    FILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/{}/input/{}.{}.entities{}.all_pos.filtered_on_all.json".format(
        sn, prefix, subset, with_classes)
    FILTER_ON_MAIN = False
    REINSTATE_PARTIAL_ENTITIES = True
    #'''
    '''
    UNFILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.test.entities.all_pos.json"
    FILTERED_FILE = "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.test.entities.all_pos.filtered.no_partials.json"
    FILTER_ON_MAIN = False
    REINSTATE_PARTIAL_ENTITIES = True
    # '''

    examples = json.load(open(UNFILTERED_FILE))
    qid_to_ents = {ex['question_id'] : ex['entities'] for ex in examples}
    examples_ents = []
    for ex in examples:
        if FILTER_ON_MAIN:
            examples_ents.append(ex['main_entity'])
        else:
            examples_ents += ex['entities']

    examples_ents = set(examples_ents)
    print("Loaded and parsed examples")

    # these are examples to filter
    in_examples_not_in_wiki = examples_ents - all_wiki_ent_ids
    print("Examples to filter: ")
    print(in_examples_not_in_wiki)

    examples_filtered = []
    partial_entities_list = []
    removed = {}
    for ex in examples:
        if FILTER_ON_MAIN:
            # filter out all examples for which main entity not in wiki IDs
            if ex['main_entity'] not in all_wiki_ent_ids:
                continue
            examples_filtered.append(ex)
        else:
            # if not (len(ex['entities']) == 1 and ex['entities'][0] is None):
            #     assert len(ex['entities_fb']) == len(ex['entities'])
            #     if 'entity_classes' in ex:
            #         assert len(ex['entity_classes']) == len(ex['entities'])
            # first filter out the bad entities
            new_entities_list = []
            # new_entity_fb_list = []
            # new_entity_classes = []
            for i, entity in enumerate(ex['entities']):
                if entity in all_wiki_ent_ids:
                    new_entities_list.append(entity)
                    # new_entity_fb_list.append(ex['entities_fb'][i])
                    # if 'entity_classes' in ex:
                    #     new_entity_classes.append(ex['entity_classes'][i])
                else:
                    removed[ex['question_id']] = entity
            # skip empty entities lists
            if len(new_entities_list) == 0:
                continue
            if len(new_entities_list) < len(ex["entities"]):
                partial_entities_list.append(ex["question_id"])
            if not REINSTATE_PARTIAL_ENTITIES:
                ex['entities'] = new_entities_list
                # ex['entities_fb'] = new_entity_fb_list
                # if 'entity_classes' in ex:
                #     ex['entity_classes'] = new_entity_classes
            examples_filtered.append(ex)

    print("Reinstated {} entities".format(len(partial_entities_list)))

    # save filtered version
    print("Filtered {} - {} = {} examples".format(
        str(len(examples)),
        str(len(examples_filtered)),
        str(len(examples) - len(examples_filtered)),
    ))
    json.dump(examples_filtered, open(FILTERED_FILE, "w"))
