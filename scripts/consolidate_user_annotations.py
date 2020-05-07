import pdb
import json
import os
import numpy
import glob
import copy


# all_wiki_ents = open("/private/home/belindali/BLINK/models/entity.jsonl").readlines()
# print("Loaded wikipedia")
# all_wiki_ents = [json.loads(line) for line in all_wiki_ents]
# print("Parsed wikipedia")
# all_wiki_ent_id_to_ents = {line['kb_idx']: line['title'] for line in all_wiki_ents if 'kb_idx' in line}
# print("Created wikipedia ID -> title map")

NAMES=["belinda", "sewon", "scott", "srini"]
def combine_annotations():
    subsets = ["dev_webqsp", "train_webqsp", "test_webqsp", "test_graphqs", "train_graphqs"]
    for subset in subsets:
        # subset_json_list = []
        subset_json_map = {}
        shared_set_check = {}
        for name in NAMES:
            fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotation_set_{}/annotated".format(name)
            if name == "sewon":
                queries = json.load(open(
                    "/private/home/belindali/BLINK/to_annotate_mention_bound/annotation_set_{}/{}.json".format(
                        name, subset)))
            fn2 = os.path.join(fn, "{}.json".format(subset))
            print(fn2)
            assert os.path.exists(fn2)
            with open(fn2) as f:
                subset_json = json.load(f)
                if name == "sewon":
                    try:
                        assert len(queries) == len(subset_json)
                    except:
                        import pdb
                        pdb.set_trace()
                for i, item in enumerate(subset_json):
                    if name == "sewon":
                        queries[i][1]["question"] = item["question"]
                        assert queries[i][0] == item['id']
                        item = queries[i]
                    id = item[0]
                    if id in subset_json_map:
                        if id not in shared_set_check:
                            shared_set_check[id] = [subset_json_map[id]]
                        shared_set_check[id] += [item[1]]
                        assert len(shared_set_check) < len(NAMES) * 10
                        assert len(shared_set_check[id]) <= len(NAMES)
                    else:
                        subset_json_map[id] = item[1]
        # json.dump(subset_json_list, open(
        #     "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}.json".format(subset), "w",
        # ))
        json.dump(subset_json_map, open(
            "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}.json".format(subset), "w",
        ), indent=2)

# combine_annotations()

FILTER_ENTITY_SET = [
    "Q44148",  # MALE
    "Q43445",  # FEMALE
    "Q8445",  # Marriage
    "Q7390",  # HUman voice
    "Q644357",  # NBA Rookie of the year award
    "Q2842604",  # NCAA Division I
]


# 1182 in test webqsp, changed entity
def get_missing_set():
    webqsp_save_files = []
    graphqs_save_files = []
    webqsp_subsets = {}
    # load webqsp
    unannotated_examples = {}
    # TODO UNCOMMENT!!!!!
    for subset in ["train", "dev", "test"]:
        with_classes = ""
        if subset == "test":
            with_classes = ".with_classes"
        fn = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.{}.entities{}.json".format(subset, with_classes)
        with open(fn) as f:
            webqsp_subset = json.load(f)
        annot_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}.json".format(subset, 'webqsp')
        predicted_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/predicted_bounds_{}_{}.json".format(subset, 'webqsp')
        annotations = json.load(open(annot_fn))
        predicted_annots = json.load(open(predicted_fn))
        webqsp_subsets[subset] = webqsp_subset
        revised_examples = []
        check_annots = {}
        for i in range(len(webqsp_subset)):
            mention_bounds = []
            for j in range(len(webqsp_subset[i]['entities'])):
                target_id = webqsp_subset[i]['entities'][j]
                new_ents_list = copy.deepcopy(webqsp_subset[i]['entities'])
                if target_id in FILTER_ENTITY_SET:
                    new_ents_list.remove(target_id)
                    continue
                if target_id != None and target_id != webqsp_subset[i]['main_entity']:  # and match is not None:
                    if str(i) not in annotations:
                        if str(i) in predicted_annots:
                            # skip, as has exact lexical overlap
                            continue
                        # delete entity
                        print("DELETING {} in example {}: {}".format(target_id, i, webqsp_subset[i]['utterance']))
                        x = None
                        check_annots[i] = {"question": webqsp_subset[i]['utterance'], "target_ID": target_id}
                    
                    elif '[' not in annotations[str(i)]['question']:
                        # delete entity
                        print("IS CORRECT OR TO DELETE?:\n\t{}\n\t{}".format(
                            annotations[str(i)]['question_with_hypothesis'], annotations[str(i)]['target'],
                        ))
                        check_annots[i] = {"question": annotations[str(i)]['question_with_hypothesis'], "target_ID": target_id, "target": annotations[str(i)]['target']}

            if len(new_ents_list) > 0:
                new_ex_copy = copy.deepcopy(webqsp_subset[i])
                new_ex_copy['entities'] = new_ents_list
                new_ex_copy['entities_pos'] = mention_bounds
                revised_examples.append(new_ex_copy)

        # json.dump(check_annots, open(annot_fn[:len(annot_fn)-len(".json")] + "_to_check.json", "w"), indent=2)

    # graphqs
    unannotated_examples = {}
    for subset in ["train", "test"]:
        with_classes = ""
        fn = "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{}.entities.json".format(subset)
        with open(fn) as f:
            graphqs_subset = json.load(f)
        annot_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}.json".format(subset, 'graphqs')
        predicted_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/predicted_bounds_{}_{}.json".format(subset, 'graphqs')
        annotations = json.load(open(annot_fn))
        predicted_annots = json.load(open(predicted_fn))
        revised_examples = []
        check_annots = {}
        unannotated_examples[subset] = {}
        for i in range(len(graphqs_subset)):
            mention_bounds = []
            for j in range(len(graphqs_subset[i]['entities'])):
                target_id = graphqs_subset[i]['entities'][j]
                new_ents_list = copy.deepcopy(graphqs_subset[i]['entities'])
                if target_id in FILTER_ENTITY_SET:
                    new_ents_list.remove(target_id)
                    continue
                if target_id != None:
                    if str(i) not in annotations:
                        if str(i) in predicted_annots:
                            # skip, as has exact lexical overlap
                            continue
                        # delete entity
                        print("DELETING {} in example {}: {}".format(target_id, i, graphqs_subset[i]['utterance']))
                        x = None
                        check_annots[i] = {"question": graphqs_subset[i]['utterance'], "target_ID": target_id}
                    
                    elif '[' not in annotations[str(i)]['question']:
                        annot_q = annotations[str(i)]['question_with_hypothesis']
                        mention = annot_q[annot_q.find('[')+1:annot_q.find(']')]
                        if mention.lower() == annotations[str(i)]['target (FB)'].lower() or mention.lower() == annotations[str(i)]['target'].lower():
                            # skip exact lexical match
                            continue

                        # delete entity
                        print("IS CORRECT OR TO DELETE?:\n\t{}\n\t{}".format(
                            annotations[str(i)]['question_with_hypothesis'], annotations[str(i)]['target'],
                        ))
                        check_annots[i] = {
                            "question": annotations[str(i)]['question_with_hypothesis'],
                            "target_ID": target_id, "target": annotations[str(i)]['target'],
                            "target (FB)": annotations[str(i)]['target (FB)'],
                        }

            if len(new_ents_list) > 0:
                new_ex_copy = copy.deepcopy(graphqs_subset[i])
                new_ex_copy['entities'] = new_ents_list
                new_ex_copy['entities_pos'] = mention_bounds
                revised_examples.append(new_ex_copy)

        # json.dump(check_annots, open(annot_fn[:len(annot_fn)-len(".json")] + "_to_check.json", "w"), indent=2)
    return webqsp_save_files, graphqs_save_files


# get_missing_set()



# 1182 in test webqsp, changed entity
def consolidate_mention_bounds():
    # load webqsp
    for subset in ["train", "dev", "test"]:
        with_classes = ""
        if subset == "test":
            with_classes = ".with_classes"
        fn = "/private/home/belindali/starsem2018-entity-linking/data/WebQSP/input/webqsp.{}.entities{}.json".format(subset, with_classes)
        with open(fn) as f:
            webqsp_subset = json.load(f)
        annot_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}.json".format(subset, 'webqsp')
        predicted_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/predicted_bounds_{}_{}.json".format(subset, 'webqsp')
        missing_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}_to_check.json".format(subset, 'webqsp')
        annotations = json.load(open(annot_fn))
        predicted_annots = json.load(open(predicted_fn))
        missing_predicted_annots = json.load(open(missing_fn))
        revised_examples = []
        nones = []
        for i in range(len(webqsp_subset)):
            mention_bounds = []
            new_ents_list = []
            for j in range(len(webqsp_subset[i]['entities'])):
                target_id = webqsp_subset[i]['entities'][j]
                if target_id == webqsp_subset[i]['main_entity']:
                    mention_bounds.append(webqsp_subset[i]['main_entity_pos'])
                    new_ents_list.append(webqsp_subset[i]['entities'][j])
                elif target_id in FILTER_ENTITY_SET or target_id == None:
                    if target_id == None:
                        nones.append(i)
                    annot_q = None
                elif target_id != webqsp_subset[i]['main_entity']:  # and match is not None:
                    if str(i) not in annotations:
                        if str(i) in predicted_annots:
                            # skip, as has exact lexical overlap
                            annot_q = predicted_annots[str(i)]["question_with_hypothesis"]
                        elif str(i) in missing_predicted_annots:
                            annot_q = missing_predicted_annots[str(i)]["question"]
                        else:
                            # delete entity
                            print("DELETING {} in example {}: {}".format(target_id, i, webqsp_subset[i]['utterance']))
                            x = None
                            while x != "del" and x != "n":
                                x = input("del/n: ")
                            if x == "del":
                                annot_q = None
                            elif x == "n":
                                annot_q = ""
                                while "[" not in annot_q:
                                    annot_q = input("corrected question?: ")
                    else:
                        if '[' in annotations[str(i)]['question']:
                            annot_q = annotations[str(i)]['question']
                        elif str(i) not in missing_predicted_annots:
                            print("DELETING {} in example {}: {}".format(target_id, i, webqsp_subset[i]['utterance']))
                            annot_q = None
                        elif '[' in missing_predicted_annots[str(i)]["question"]:
                            annot_q = missing_predicted_annots[str(i)]["question"]
                        else:
                            raise AssertionError

                    if annot_q is not None:
                        new_ents_list.append(webqsp_subset[i]['entities'][j])
                        mention_bounds.append([annot_q.find('['), annot_q.find(']')-1])
                        assert annot_q[annot_q.find('[')+1:annot_q.find(']')] == webqsp_subset[i]['utterance'][
                            mention_bounds[-1][0]:mention_bounds[-1][1]
                        ]

            try:
                assert len(mention_bounds) == len(new_ents_list)
            except:
                pdb.set_trace()
            if len(new_ents_list) > 0:
                new_ex_copy = copy.deepcopy(webqsp_subset[i])
                new_ex_copy['entities'] = new_ents_list
                new_ex_copy['entities_pos'] = mention_bounds
                revised_examples.append(new_ex_copy)

        print(fn)
        print(nones)
        json.dump(revised_examples, open(fn[:len(fn)-len(".json")] + ".all_pos.json", "w"), indent=2)

    # graphqs
    for subset in ["train", "test"]:
        with_classes = ""
        fn = "/private/home/belindali/starsem2018-entity-linking/data/graphquestions/input/graph.{}.entities.json".format(subset)
        with open(fn) as f:
            graphqs_subset = json.load(f)
        annot_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}.json".format(subset, 'graphqs')
        predicted_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/predicted_bounds_{}_{}.json".format(subset, 'graphqs')
        missing_fn = "/private/home/belindali/BLINK/to_annotate_mention_bound/annotated/{}_{}_to_check.json".format(subset, 'graphqs')
        annotations = json.load(open(annot_fn))
        predicted_annots = json.load(open(predicted_fn))
        missing_predicted_annots = json.load(open(missing_fn))
        revised_examples = []
        for i in range(len(graphqs_subset)):
            mention_bounds = []
            new_ents_list = []
            for j in range(len(graphqs_subset[i]['entities'])):
                target_id = graphqs_subset[i]['entities'][j]
                if target_id in FILTER_ENTITY_SET or target_id == None:
                    if target_id == None:
                        nones.append(i)
                    annot_q = None
                else:
                    if str(i) not in annotations:
                        if str(i) in predicted_annots:
                            # skip, as has exact lexical overlap
                            annot_q = predicted_annots[str(i)]["question_with_hypothesis"]
                        elif str(i) in missing_predicted_annots:
                            annot_q = missing_predicted_annots[str(i)]["question"]
                        else:
                            # delete entity
                            print("DELETING {} in example {}: {}".format(target_id, i, graphqs_subset[i]['utterance']))
                            x = None
                            while x != "del" and x != "n":
                                x = input("del/n: ")
                            if x == "del":
                                annot_q = None
                            elif x == "n":
                                annot_q = ""
                                while "[" not in annot_q:
                                    annot_q = input("corrected question?: ")
                    else:
                        if '[' in annotations[str(i)]['question']:
                            annot_q = annotations[str(i)]['question']
                        elif str(i) not in missing_predicted_annots:
                            if '[' in annotations[str(i)]['question_with_hypothesis']:
                                annot_q = annotations[str(i)]['question_with_hypothesis']
                                mention = annot_q[annot_q.find('[')+1:annot_q.find(']')]
                                # exact lexical match
                                if mention.lower() != annotations[str(i)]['target (FB)'].lower() and mention.lower() != annotations[str(i)]['target'].lower():
                                    print("DELETING {} in example {}: {}".format(target_id, i, graphqs_subset[i]['utterance']))
                                    annot_q = None
                        elif '[' in missing_predicted_annots[str(i)]["question"]:
                            annot_q = missing_predicted_annots[str(i)]["question"]
                        else:
                            raise AssertionError

                    if annot_q is not None:
                        new_ents_list.append(graphqs_subset[i]['entities'][j])
                        mention_bounds.append([annot_q.find('['), annot_q.find(']')-1])
                        assert annot_q[annot_q.find('[')+1:annot_q.find(']')] == graphqs_subset[i]['utterance'][
                            mention_bounds[-1][0]:mention_bounds[-1][1]
                        ]

            try:
                assert len(mention_bounds) == len(new_ents_list)
            except:
                pdb.set_trace()
            if len(new_ents_list) > 0:
                new_ex_copy = copy.deepcopy(graphqs_subset[i])
                new_ex_copy['entities'] = new_ents_list
                new_ex_copy['entities_pos'] = mention_bounds
                revised_examples.append(new_ex_copy)

        print(fn)
        print(nones)
        json.dump(revised_examples, open(fn[:len(fn)-len(".json")] + ".all_pos.json", "w"), indent=2)

consolidate_mention_bounds()
