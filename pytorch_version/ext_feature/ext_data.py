from tqdm import tqdm
from collections import Counter
from ast import literal_eval


PAD = 0
BOS = 1
EOS = 2
UNK = 3
cate_filename = '../pytorch_version/ext_feature/app_cate.txt'


class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


class Ext_data():
    def __init__(self, max_vocab_size=50):
        self.max_vocab_size = max_vocab_size
        self.cate2app, self.app2cate, self.app_cates = self.load_cates(cate_filename)
        self.cate_counter = self.build_cate_counter(self.app_cates)
        self.senti_sents = list(range(-5, 0)) + list(range(1, 6))
        self.senti_counter = self.build_cate_counter(self.senti_sents)
        self.app_counter = self.build_cate_counter(self.app2cate.keys())
        self.cate_vocab = self.build_rate_vocab(self.cate_counter)
        self.senti_vocab = self.build_rate_vocab(self.senti_counter)
        self.app_vocab = self.build_rate_vocab(self.app_counter)

    def __getitem__(self, index):
        cate_sent = self.cate_vocab.token2id[self.app_cates[index]]
        senti_sent = self.senti_vocab.token2id[self.senti_sents[index]]

        return cate_sent, senti_sent

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq

    def load_cates(self, catepath):
        cate_fr = open(catepath)
        cate2app = literal_eval(cate_fr.readlines()[0])
        cate_fr.close()
        app2cate = {}
        for cate, apps in cate2app.items():
            for app in apps:
                app2cate[app] = cate
        return cate2app, app2cate, cate2app.keys()

    def build_cate_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter[sent] += 1
        return counter

    def build_rate_vocab(self, counter):
        ### Can be
        vocab = AttrDict()
        vocab.token2id = {token: _id for _id, (token, count) in tqdm(enumerate(counter.most_common()))}
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        return vocab

if __name__ == "__main__":
    train_dataset = Ext_data()
    print(train_dataset.senti_vocab)
    print(train_dataset.cate_vocab)
    print(train_dataset.app_vocab)
    print(train_dataset.cate2app)
    print(train_dataset.app2cate)
