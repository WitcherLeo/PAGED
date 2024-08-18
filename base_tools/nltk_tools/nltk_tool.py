import nltk


class NLTKTool(object):
    def __init__(self):
        self.tool_name = "nltk"

    def tokenize(self, text: str):
        return nltk.word_tokenize(text)

    def compute_bleu_single(self, pred: str, label: str):
        pred = nltk.word_tokenize(pred)
        label = nltk.word_tokenize(label)
        return nltk.translate.bleu_score.sentence_bleu([label], pred)


if __name__ == '__main__':
    test_tool = NLTKTool()
    test_pred = 'I like the way you smile at me.'
    test_label = 'I like the way you smile at me.'
    print(test_tool.compute_bleu_single(test_pred, test_label))
