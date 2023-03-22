from numpy import mean
from nltk.translate import meteor_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def bleu_scores(gts, res, n=4):
    results = {}
    scorer = Bleu(n)
    score, scores = scorer.compute_score(gts, res)
    if type(score) == list:
        for i, s in enumerate(score):
            results['BLEU {}'.format(i + 1)] = s
            results['BLEU {} scores'.format(i + 1)] = scores[i]
    else:
        results['BLEU 1'] = score
        results['BLEU 1 scores'] = scores

    return results


def cider_scores(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)

    return {'CIDEr': score, 'CIDEr scores': scores}


# This part of codes contains some bugs we are unable to fix.
# def meteor_scores(gts, res):
#    scorer = Meteor()
#    score, scores = scorer.compute_score(gts, res)
#
#    return {'METEOR': score, 'METEOR scores':scores}


def meteor_scores(gts, res):
    scores = []
    for id in gts.keys():
        score_m = meteor_score.single_meteor_score(gts[id][0], res[id][0])
        scores.append(score_m)
    score = mean(scores)

    return {'METEOR': score, 'METEOR scores': scores}


def rougel_scores(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)

    return {'ROUGE-L': score, 'ROUGE-L scores': scores}


def spice_scores(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)

    return {'SPICE': score, 'SPICE scores': scores}


def show_all_scores(gts, res, n=4, remove_common=False):
    """
    :param gts: list of ground truths
    :param res: list of references
    :return: Language scores
    """
    assert len(gts) == len(res), "ERROR: The length of references do not fit that of ground truths!"
    if remove_common:
        for i in range(len(res)):
            index = res[i].find(".")
            if index >= 0:
                res[i] = res[i][:index+1]
            index = res[i].find("?")
            if index >= 0:
                res[i] = res[i][:index+1]
            index = res[i].find("!")
            if index >= 0:
                res[i] = res[i][:index + 1]

    gts = {i: [txt] for i, txt in enumerate(gts)}
    res = {i: [txt] for i, txt in enumerate(res)}
    bleu_results = bleu_scores(gts, res, n=n)
    meteor_results = meteor_scores(gts, res)
    cider_results = cider_scores(gts, res)
    rouge_results = rougel_scores(gts, res)
    rsum = 0.0
    for i in range(n):
        rsum += bleu_results['BLEU {}'.format(i + 1)]
        print("BLEU {} score: {}".format(i + 1, bleu_results['BLEU {}'.format(i + 1)]))
    rsum += meteor_results['METEOR']
    print("METEOR score: {}".format(meteor_results['METEOR']))
    rsum += cider_results['CIDEr']
    print("CIDEr score: {}".format(cider_results['CIDEr']))
    rsum += rouge_results['ROUGE-L']
    print("ROUGE-L score: {}".format(rouge_results['ROUGE-L']))
    print("rSUM score: {}".format(rsum))

    return bleu_results, meteor_results, cider_results, rouge_results, rsum


def test_stories(end_gen, gold_file="./data/gen/test.txt"):
    with open(gold_file, 'r', encoding='utf-8') as gf:
        gts = [txt.strip('\n') for txt in gf.readlines()]

    res = [txt.lower() for txt in end_gen]
    return show_all_scores(gts, res)


if __name__ == '__main__':
    # Input stories
    gold_file = "./data/gen/test.txt"
    pred_file = "./data/gen/gen_ending.txt"

    with open(gold_file, 'r', encoding='utf-8') as gf:
        gts = [txt.strip('\n') for txt in gf.readlines()]

    with open(pred_file, 'r', encoding='utf-8') as pf:
        res = [txt.lower().strip('\n') for txt in pf.readlines()]

    bleu_results, meteor_results, cider_results, rouge_results, rsum = show_all_scores(gts, res)
