
def resolve_entity_overlap_beam_search(entities):
    """
    If there are overlapping entity links it will resolve them so that the best linkings are prefered.
    The method is greedy and doesn't guarantee the best solution.

    :param entities: a list of entities as dictionaries (fields: token_ids, drop_score)
    :param single_entity_mode: if True all entities are considered as overlapping and a single entity per list
    is selected
    :return:
    >>> resolve_entity_overlap_beam_search([{'linkings':[{}], 'drop_score':0.7, 'token_ids':[0,1,2,3,4]}, \
                                              {'linkings':[{}], 'drop_score':0.6, 'token_ids':[1,2]}, \
                                              {'linkings':[{}], 'drop_score':0.78, 'token_ids':[3,4]}, \
                                              {'linkings':[{}], 'drop_score':0.8, 'token_ids':[5,6]}])

    """
    sorted_by_position = sorted([el for el in entities if el.get('token_ids')], key=lambda el: min(el['token_ids']))
    if len(sorted_by_position) == 0:
        return []
    e = sorted_by_position.pop(0)
    groups = [[e]]
    while len(sorted_by_position) > 0:
        e2 = sorted_by_position.pop(0)
        tokens_next = set(e2.get('token_ids', []))
        added = False
        for group in groups:
            tokens = set(group[-1].get('token_ids', []))
            if len(tokens & tokens_next) == 0:
                group.append(e2)
                added = True
        if not added:
            groups.append([e2])
    group_drop_scores = []
    for group in groups:
        group_drop_score = 1.0
        for e in group:
            group_drop_score *= e.get("drop_score", 0.0)
        group_drop_scores.append(group_drop_score)

    return_group = sorted(list(zip(groups, group_drop_scores)), key=lambda x: x[1])
    if return_group:
        return return_group[0][0]
    return []