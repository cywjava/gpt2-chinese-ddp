import collections
import re

import jieba


class JiebaTokenizer(object):
    """
        使用结巴分词器构造tokens id
        author:chen.yiwan
        date:2023-05-15
        description:1-5 是特殊词汇
                    6-8110 是汉字三级单字表，共8105个
                    8111-8823 是各种符号、和数字
                    8824 开始是（清华开源词表+余总提供的词），去掉了单字且做了剔重后余下的
    """

    def __init__(self, ext_vocab_file):
        """
            Constructs a JiebaTokenizer.
        """
        self._vocab = collections.OrderedDict()
        self._ids_to_tokens = collections.OrderedDict()
        self._unk_token = "[UNK]"
        self._sep_token = "[SEP]"
        self._pad_token = "[PAD]"
        self._cls_token = "[CLS]"
        self._mask_token = "[MASK]"
        self._additional_special_tokens = [self._unk_token, self._sep_token, self._pad_token, self._cls_token,
                                           self._mask_token]
        self._unk_sets = set()
        self._all_special_ids = [0, 1, 2, 3, 4]
        self._special_pattern = ""
        for _item in self._additional_special_tokens:
            if _item == self._additional_special_tokens[len(self._additional_special_tokens) - 1]:
                self._special_pattern += str(_item.replace("[", "\\[").replace("]", "\\]"))
            else:
                self._special_pattern += str(_item.replace("[", "\\[").replace("]", "\\]")) + "|"

        self._jieba = jieba
        self._jieba.load_userdict(ext_vocab_file)
        self._ext_vocab_file = ext_vocab_file
        self._load_vocab()

    def get_vocab_size(self):
        return len(self._vocab)

    def _load_vocab(self):
        """Loads a vocabulary file into a dictionary."""
        with open(self._ext_vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip('\n')
            self._vocab[token] = index
        self._ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self._vocab.items()])

    def get_unk_sets(self):
        return self._unk_sets

    def tokenize(self, text, sub_tokenization: bool = False):
        output = []
        positions = []
        for m in re.finditer(self._special_pattern, text):
            positions.append([m.start(), m.end()])
        texts = []
        if len(positions) > 0:
            texts.append(text[0:positions[0][0]])
            for idx in range(len(positions)):
                texts.append(text[positions[idx][0]: positions[idx][1]])
                if idx < len(positions) - 1:
                    texts.append(text[positions[idx][1]: positions[idx + 1][0]])
            texts.append(text[positions[len(positions) - 1][1]:len(text)])
        else:
            texts.append(text)

        if not sub_tokenization:
            for _text in texts:
                if _text not in self._additional_special_tokens:
                    for token in self._jieba.lcut(_text):
                        output.append(token)
                else:
                    output.append(_text)
        else:
            for _text in texts:
                if _text not in self._additional_special_tokens:
                    temps = self._jieba.lcut(_text)
                    ids = self.convert_tokens_to_ids(temps)
                    for idx in range(len(temps)):
                        if ids[idx] != 4:
                            output.append(temps[idx])
                        else:
                            for char in temps[idx]:
                                if self._convert_tokens_to_id(char) == 4:
                                    self._unk_sets.add(char)
                                output.append(char)
                else:
                    output.append(_text)
        return output

    def _convert_tokens_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        if token == " ":
            return self._vocab.get(token, self._vocab.get(self._pad_token))
        return self._vocab.get(token, self._vocab.get(self._unk_token))

    def convert_tokens_to_ids(self, tokens: []):
        output = []
        for token in tokens:
            output.append(self._convert_tokens_to_id(token))
        return output

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self._unk_token)

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self._all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens


# 测试分词
# tokenizer = JiebaTokenizer("G:\\idea_work2\\GPT2-Chinese\\cache\\jieba_ext_dict.txt")
# str = "一乙基础数据结构元及40"
# tokens = tokenizer.tokenize(str)
# print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
# print("二次分词后的效果：")
# tokens = tokenizer.tokenize(str, sub_tokenization=True)
# print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
# print("打印UNK字符集")
# print(tokenizer.get_unk_sets())
