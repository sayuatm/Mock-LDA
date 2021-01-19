import os
import sys
import numpy
import nltk


def build_words():
    word_list = []
    words = {}
    path, dirs, files = next(os.walk('files'))

    for name in files:
        f = open(os.path.join(path, name), 'r')
        content = f.read()
        noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
        tokenized = nltk.word_tokenize(content)
        all_nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if noun_adj(pos)]
        word_list += all_nouns
        words[name] = {i: {} for i in range(len(all_nouns))}
        for i in range(len(all_nouns)):
            words[name][i]['word'] = all_nouns[i]
            words[name][i]['label'] = ''
    distinct_words = list(set(word_list))

    return words, distinct_words


def initial_label(words, k):
    for doc in words:
        for i in range(len(words[doc])):
            words[doc][i]['label'] = numpy.random.randint(0, k)


def gibbs_sampling(words, k, a, b):
    for doc in words:
        for el in words[doc]:
            local_counter = {i: 0 for i in range(k)}
            global_counter = {i: 0 for i in range(k)}
            current_word = words[doc][el]['word']
            words[doc][el]['label'] = ''
            for word in words[doc]:
                label = words[doc][word]['label']
                if label in local_counter:
                    local_counter[label] += 1
            for d in words:
                for word in words[d]:
                    topic_label = words[d][word]['label']
                    if words[d][word]['word'] == current_word and topic_label in global_counter:
                        global_counter[topic_label] += 1
            weight = [-1] * k
            for i in range(k):
                weight[i] = (local_counter[i] + a) * (global_counter[i] + b)
            acc = 0
            box = []
            for w in weight:
                acc += w
                box += [acc]
            # print(acc)
            d = numpy.random.randint(acc)

            index = 0
            for e in box:
                if d > e:
                    index += 1
                else:
                    break
            words[doc][el]['label'] = index


def get_tw_distribution(vocab, words, k):
    # wt_counter = {word: {i: 0 for i in range(k)} for word in vocab}
    tw_counter = {i: {v: 0 for v in vocab} for i in range(k)}
    for doc in words:
        for word in words[doc]:
            # wt_counter[words[doc][word]['word']][words[doc][word]['label']] += 1
            tw_counter[words[doc][word]['label']][words[doc][word]['word']] += 1
    return tw_counter


def main():
    k = int(sys.argv[1])
    word_bag, vocab = build_words()
    alpha = 0.2
    # alpha = 50 / k
    beta = 0.1
    # beta = 200 / len(word_bag)
    initial_label(word_bag, k)
    for i in range(int(sys.argv[2])):
        gibbs_sampling(word_bag, k, alpha, beta)
    # wt_distribution, tw_distribution = get_tw_distribution(vocab, word_bag, k)
    tw_distribution = get_tw_distribution(vocab, word_bag, k)
    for topic in tw_distribution:
        result = dict(sorted(tw_distribution[topic].items(), key=lambda item: item[1], reverse=True)[:5])
        print('topic %r: %r' % (topic, list(result.keys())))


if __name__ == '__main__':
    main()
