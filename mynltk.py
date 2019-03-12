
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Tokenize
"""NLTK, regex, string"""
class preprocessing:

    import nltk

    def doc2sent(document):
        from nltk.tokenize import sent_tokenize
        sent_tokenize(document)

    def doc2words(document):
        from nltk.tokenize import sent_tokenize, word_tokenize
        word_tokenize(sent_tokenize(document))

    def sent2words(sentence):
        from nltk.tokenize import word_tokenize
        word_tokenize(sentence)

    def n_gram(sentence, min=1, max=4):
        from nltk.util import ngrams
        ngram_list = []
        for n in range(min, max):
            for ngram in ngrams(sentence, n):
                # "".join(str(i) for i in ngarm)
                ngram_list.append(ngram)
        return ngram_list

    def non_alphanumeric_dropper(words):
        words = words.str.replace(r"\d+", "")
        words = words.str.replace('[^\w\s]','')
        words = words.str.replace(r"[︰-＠]", "")
        return words


    def spell_correcter(words):
        from autocorrect import spell
        for word in words:
            spell(word)


    def stop_punk_dropper(words):
        from string import punctuation
        from nltk.corpus import stopwords
        from nltk import word_tokenize
        stop_words = stopwords.words('english') + list(punctuation)
        words = word_tokenize(sentence)
        lower_words = [w.lower() for w in words]
        return [w for w in lower_words if w not in stop_words]


    def stemmer(words):
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        for w in words:
            stemmer.stem(w)

    def lemmatizer(words):
        nltk.download('wordnet')
        from nltk.stem.wordnet import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        for w in words:
            lemmatizer.lemmatize(w)


# POS（品詞解析）
"""NLTK"""
class PoS:

    def sent2pos(sentence):
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('words')
        from nltk import word_tokenize, pos_tag
        return pos_tag(word_tokenize(sentence))

    def word2pos(words):
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('words')
        from nltk import pos_tag
        return pos_tag(words)

"""NLTK"""
class ShallowParsing:
    """
    - Working with chunks is way easier than working with full-blown parse trees.
    - It’s an easier task and a shallow parser can be more accurate
    - Chunking is a very similar task to Named-Entity-Recognition. In fact, the same format, IOB-tagging is used
    """
    def chunking(sentence):
        from nltk.chunk import conlltags2tree, tree2conlltags
        iob_tagged = tree2conlltags(sentence)
        chunked_tree =conlltags2tree(iob_tagged)
        return chunked_tree


# Parsing (構文解析)

"""stanfordNLP"""
class DeepParsing:
    """
    - Parsing is Relationship Extraction Technique
    1. Deep Parsing:
    2. Constituency Parsing:
    3. Dependency Parsing:
    # Paper1: https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf
    # Paper2: https://nlp.stanford.edu/software/lex-parser.shtml
    # Qiita: https://qiita.com/To_Murakami/items/d20db917a4e5e1ae0a0c
    # Tutorial1: https://www.commonlounge.com/discussion/1ef22e14f82444b090c8368f9444ba78
    # Tutorial2: http://lab.astamuse.co.jp/entry/corenlp1
    """
    def DeepParser(sentence):
        parser = nltk.ChartParser(groucho_grammar)
        for tree in parser.parse(sent)


    def ConstituencyParser(sentence):

        from nltk.parse.stanford import StanfordParser
        # create parser object
        scp = StanfordParser(path_to_jar='/path/to/stanford-parser.jar', path_to_models_jar='path/to/stanford-parser-models.jar')
        # get parse tree
        result = list(scp.raw_parse(sentence))

    def DependencyParser(sentence):
        from nltk.parse.stanford import StanfordDependencyParser
        # create parser object
        scp = StanfordDependencyParser(path_to_jar='/path/to/stanford-parser.jar', path_to_models_jar='path/to/stanford-parser-models.jar')
        # get parse tree
        result = list(scp.raw_parse(sentence))


# NER（固有表現抽出）
"""NLTK"""
class NER:

    """
    # http://www.informit.com/articles/article.aspx?p=2265404
    """

    import nltk
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.chunk import conlltags2tree, tree2conlltags
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    def iob_tagged_ner(sentence):
        """
        from nltk import word_tokenize, pos_tag, ne_chunk
        from nltk.chunk import conlltags2tree, tree2conlltags
        """
        ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
        iob_tagged = tree2conlltags(ne_tree)
        iob_tagged_ne_tree = conlltags2tree(iob_tagged)
        return iob_tagged_ne_tree

    def sent2ner(sentence):
        """
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        from nltk import word_tokenize, pos_tag, ne_chunk
        """
        return ne_chunk(pos_tag(word_tokenize(sentence)))

    def word2ner(words):
        """
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        from nltk import pos_tag, ne_chunk
        """
        return ne_chunk(pos_tag(words))


# pandas.Seriesに.apply()で適応する
# df["column"].apply(lambda x: tfidf_tokenizer(x))
"""scikit-learn"""
class BoW:

    """
    Bag-of-wordsは文書内の単語の出現回数をベクトルの要素とした分散表現
    弱点: 意味的な表現を学習することがない
    """

    def tfidf_tokenizer(sentence, min_df=0.3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df)
        tfidf = vectorizer.fit_transform(sentence)
        #tfidf_tfidf.get_feature_names()
        return tfidf

    def count_tokenizer(sentence):
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        tf = vectorizer.fit_transform(sentence)
        #tf.get_feature_names()
        return tf


"""Genism, FastText, sklearn"""
class WordEmbedding:

    """
    アウトプットはなに？　周辺後？なんの意味がある？
    """

    """
    1. 同じような意味の単語からは、同じような周辺語が予測されるはず
    2. ある単語の周りに出現する単語を予測する学習を、ニューラルネットワークで行う
    3. 学習の結果、入力層から隠れ層への重みが計算される
    4. 入力がont-hotベクトルなので、入力層から隠れ層への重み行列の各行を、そのまま単語ベクトルと表現してよいだろう
    Keyword: ブラウンクラスタリング
    """

    def word2vec(words, model, size=100, window=5, min_count=5, workers=4):
        """
        * CBOW predicts the word from the context
        * skip-gram predicts the context from the word
        """
        # model=0 is CBOW, 1 is skip-gram
        from gensim.models import Word2Vec
        vectorizer = Word2Vec(sentences=words, size, window, min_count, workers, sg=model)

    def fasttext(words, size=100, window=5, min_count=5, workers=4, sg=1):
        from gensim.models import FastText
        vectorizer = FastText(sentences=words, size, window, min_count, workers, sg)

    """ Word2vec by NN from scratch with Skip-gram, CBOW """
    # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch

    def lda2vec():
        """
        Mark Zuckerberg - Facebook + Amazon = Jeff Bezos
        1. Combining global document themes with local word patterns
        2. Dense word vectors but sparse document vectors
        3. Mixture models for interpretability
        # https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=
        """
        from lda2vec import LDA2Vec()

    def GloVe():
        from glove import Corpus, Glove
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec

    def doc2vec(words, tags):

        """
        - PV-DM(Paragraph Vector Distributed Memory)
         word2vecのCBoWと似ている
         CBoWモデルに文章ベクトルを追加して中間層へ入力
         単語を予測する
        - PV-DBOW(Paragraph Vector Distributed Bag-of-words)
         word2vecのSkip-gramと似ている
         文章ベクトルの入力から使われている単語を予測する
        """
        """
        - コンテンツベースのレコメンド
        - 感情分析
        - 文書分類
        - スパムフィルタリング
        """
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        id2word = TaggedDocument(words=words, tags=[i])
        model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(id2word)
        model.train(id2word, total_examples=model.corpus_count, epochs=model.epochs)
        


    #GloVe

    #LDA2vec
    #doc2vec

"""Genism"""
class TopicModel:

    """
    What does LDA do?
     The quality of text processing.
     The variety of topics the text talks about.
     The choice of topic modeling algorithm.
     The number of topics fed to the algorithm.
     The algorithms tuning parameters.

    Use Case
    - Text Summarization: Resume Summarization
    - Documented clustering: Customer Review
    - Recommender System:
    """

    import gensim
    from gensim import corpora
    import pickle
    from gensim.models import LdaModel, LsiModel
    from gensim.models import CoherenceModel
    """from gensim.models.wrappers import LdaMallet"""
    #rom nltk.corpus import stopwords
    #stop_words = stopwords.words('english')

    # num_topics is for LDAmodel, num_wordsis for print_topics
    """フロー: 1.辞書化, 2.頻度計算, 3.モデル選択（トピック数）, 4.トピック可視化, 5.性能評価"""

    def LDAmodel(words, num_topics=5, num_words=5):
        """
        1. the number of words
        2. the mixture of topics ex: 1/2 the topic “health” and 1/2 the topic “vegetables" etc..
        3. the probability of topic depends on their dominancy
        """
        dictionary = corpora.Dictionary(words)
        # Term Document Frequency
        corpus = [dictionary.doc2bow(word) for word in words]
        # save it!
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')
        # Train model
        ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        # lda_model = LdaModel(corpus=corpus,id2word=id2word,num_topics=20, random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
        topics = ldamodel.print_topics(num_topics=num_topics, num_words=num_words)
        # Validation
        # A measure of how good the model is. lower the better.
        val_perplexity = ldamodel.log_perplexity(corpus)
        # cohherent score
        coherence_ldamodel = CoherenceModel(model=ldamodel, texts=words, dictionary=dictionary, coherence='c_v')
        val_coherence = coherence_ldamodel.get_coherence()

        return topics, val_perplexity, val_coherence


    def LSAmodel(words, num_topics=5, num_words=5):

        dictionary = corpora.Dictionary(words)
        # Term Document Frequency
        corpus = [dictionary.doc2bow(word) for word in words]
        # save it!
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')
        # Train model
        lsimodel = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        # print_topics(num_topics=20, num_words=10)
        topics = lsimodel.print_topics(num_topics=num_topics, num_words=num_words)
        # Validation
        # A measure of how good the model is. lower the better.
        val_perplexity = lsimodel.log_perplexity(corpus)
        # cohherent score
        coherence_lsimodel = CoherenceModel(model=lsimodel, texts=words, dictionary=dictionary, coherence='c_v')
        val_coherence = coherence_lsimodel.get_coherence()

        return topics, val_perplexity, val_coherence

    # def PLSAmodel():

"""Pytorch"""
class SequenceGenerator:

    def seq2seq(sentence):
        from seq2seq import Seq2SeqLSTM  # import the Seq2Seq-LSTM package
        seq2seq = Seq2SeqLSTM()  # create new sequence-to-sequence transformer

class GANs:

    def GANN():
        sfdg

class SentimentAnalysis:

    def SentimentAnalyser():
        sdfgh
