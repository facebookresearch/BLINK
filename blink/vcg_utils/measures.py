import numpy as np

def entity_linking_tp_with_overlap(gold, predicted):
    """

    Partially adopted from: https://yiyangnlp.github.io/downloads/yang-acl-2015-updated.pdf

    :param gold:
    :param predicted:
    :return:
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16), ('Q780394', 24, 32)])
    2
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    1
    >>> entity_linking_tp_with_overlap([(None, 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    0
    >>> entity_linking_tp_with_overlap([(None, 14, 18), (None, )], [(None,)])
    1
    >>> entity_linking_tp_with_overlap([('Q7366', ), ('Q780394', )], [('Q7366', 14, 16)])
    1
    >>> entity_linking_tp_with_overlap([], [('Q7366', 14, 16)])
    0
    """
    if not gold or not predicted:
        return 0
    # Add dummy spans, if no spans are given, everything is overlapping per default
    if any(len(e) != 3 for e in gold):
        gold = [(e[0], 0, 1) for e in gold]
        predicted = [(e[0], 0, 1) for e in predicted]
    # Replace None KB ids with empty strings
    gold = [("",) + e[1:] if e[0] is None else e for e in gold]
    predicted = [("",) + e[1:] if e[0] is None else e for e in predicted]

    gold = sorted(gold, key=lambda x: x[2])
    predicted = sorted(predicted, key=lambda x: x[2])

    lcs_matrix = np.zeros((len(gold), len(predicted)), dtype=np.int16)
    for g_i in range(len(gold)):
        for p_i in range(len(predicted)):
            gm = gold[g_i]
            pm = predicted[p_i]

            if not (gm[1] >= pm[2] or pm[1] >= gm[2]) and (gm[0].lower() == pm[0].lower()):
                if g_i == 0 or p_i == 0:
                    lcs_matrix[g_i, p_i] = 1
                else:
                    lcs_matrix[g_i, p_i] = 1 + lcs_matrix[g_i - 1, p_i - 1]
            else:
                if g_i == 0 and p_i == 0:
                    lcs_matrix[g_i, p_i] = 0
                elif g_i == 0 and p_i != 0:
                    lcs_matrix[g_i, p_i] = max(0, lcs_matrix[g_i, p_i - 1])
                elif g_i != 0 and p_i == 0:
                    lcs_matrix[g_i, p_i] = max(lcs_matrix[g_i - 1, p_i], 0)
                elif g_i != 0 and p_i != 0:
                    lcs_matrix[g_i, p_i] = max(lcs_matrix[g_i - 1, p_i], lcs_matrix[g_i, p_i - 1])

    match_count = lcs_matrix[len(gold) - 1, len(predicted) - 1]
    return match_count