# -*- coding: utf-8 -*-
# !python3

import langdetect
from stanza.server import CoreNLPClient

'''
    For reference 
    Download and Install Stanford CoreNLP: 
    https://stanfordnlp.github.io/CoreNLP/download.html

    Official Python implementation is stanza:
    https://stanfordnlp.github.io/CoreNLP/other-languages.html#python

    Penn Treebank POS tags: 
    https://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports

    Stanza basic usage of CoreNLP client:
    https://stanfordnlp.github.io/stanza/client_usage.html

    Stanford NLP dependencies manual:
    https://nlp.stanford.edu/software/dependencies_manual.pdf

    Setting the CoreNLP root folder as environment variable
    use .bash_profile

    export CLASSPATH=$CLASSPATH:/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0/*:
    CORENLP_HOME="/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0"
    for file in `find $CORENLP_HOME/ -name "*.jar"`;
    do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

    #### if you need to use the CORENLP_HOME path for something:
    corenlp_home = os.environ['CORENLP_HOME']

    ##############################
    #### Annotators explained ####
    ##############################

    # default annotators is all: annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    # tokenize: splits each word
    # ssplit: splits the structure by sentence in a list of sentences.
    # pos: Part of Speech tagging
    # lemma: Lemmatizes the words to a basic conjugation/ dictionary form
    # ner: Named Entity Recognizer
    # parse: Parsing
    # depparse: Dependency Parsing
    # coref: Coreference Resolution


    ########### For English
    # stanford-corenlp-4.1.0-models-english.jar
    # properties are the default

    # Examples of use:
    def example_English():
        text = 'This is a test sentence for the server to handle. I wonder what it will do.'
        with CoreNLPClient(
                    annotators=None,
                        #['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
                    properties=None,
                                #{
                            #'tokenize_pretokenized': True, # Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
                            #'tokenize_no_ssplit':True # Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
                                #}, #You can add more or just use None
                    timeout=150000
                        ) as client:
            ann = client.annotate(text)
        # ann is a Document class, broken down in Sentence objects, which each have Token objects inside.
        # For example, to take the list of tokenized words out:
        len(ann.sentence)
        # 2
        sent_list = [token.word for token in ann.sentence[0].token]
        # ['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle','.']
        sent_list = [token.word for token in ann.sentence[1].token]
        # ['I', 'wonder', 'what', 'it', 'will', 'do','.']

    ########### For Chinese
    # properties from StanfordCoreNLP-chinese.properties

    # Examples of use:
    def example_Chinese():
        text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
        # Taken from stanford-corenlp-4.1.0-models-chinese.jar
        properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
        with CoreNLPClient(
                    annotators=None,
                    properties=properties,  # properties from StanfordCoreNLP-chinese.properties
                    timeout=15000
                        ) as client:
            ann = client.annotate(text)
        sent_list = [token.word for token in ann.sentence[0].token]
        # ['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']
'''


def get_StanfordCoreNLP_chinese_properties(properties=None):
	'''
        Exports properties taken from stanford-corenlp-4.1.0-models-chinese.jar to be able to run the Chinese models with the python client.

        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)

        :return: Properties enabling Chinese language parsing, in addition to any in parameters.

        For example:
            properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
            with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout) as client:
                ann = client.annotate(text)
    '''
	StanfordCoreNLP_chinese_properties = {'annotators': ('tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'),
										  'tokenize.language': 'zh',
										  'segment.model': 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz',
										  'segment.sighanCorporaDict': 'edu/stanford/nlp/models/segmenter/chinese',
										  'segment.serDictionary': 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz',
										  'segment.sighanPostProcessing': True,
										  'ssplit.boundaryTokenRegex': '[.。]|[!?！？]+',
										  'pos.model': 'edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger',
										  'ner.language': 'chinese',
										  'ner.model': 'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz',
										  'ner.applyNumericClassifiers': True, 'ner.useSUTime': False,
										  'ner.fine.regexner.mapping': 'edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab',
										  'ner.fine.regexner.noDefaultOverwriteLabels': 'CITY,COUNTRY,STATE_OR_PROVINCE',
										  'parse.model': 'edu/stanford/nlp/models/srparser/chineseSR.ser.gz',
										  'depparse.model   ': 'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz',
										  'depparse.language': 'chinese',
										  'coref.sieves': 'ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch',
										  'coref.input.type': 'raw', 'coref.postprocessing': True,
										  'coref.calculateFeatureImportance': False, 'coref.useConstituencyTree': True,
										  'coref.maxMentionDistance': 128,
										  'coref.maxMentionDistanceWithStringMatch': 128,
										  'coref.useSemantics': False, 'coref.algorithm': 'neural',
										  'coref.path.word2vec': '', 'coref.language': 'zh',
										  'coref.defaultPronounAgreement': True,
										  'coref.zh.dict': 'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz',
										  'coref.print.md.log': False, 'coref.md.type': 'RULE',
										  'coref.md.liberalChineseMD': False,
										  'kbp.semgrex': 'edu/stanford/nlp/models/kbp/chinese/semgrex',
										  'kbp.tokensregex': 'edu/stanford/nlp/models/kbp/chinese/tokensregex',
										  'kbp.language': 'zh', 'kbp.model': None,
										  'entitylink.wikidict': 'edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz'}
	if properties:
		StanfordCoreNLP_chinese_properties.update(properties)
	return StanfordCoreNLP_chinese_properties


#############################################################################################
#############################################################################################
#############################################################################################
##### Methods to simplify using the CoreNLP client for specific purposes in my projects #####
#############################################################################################
#############################################################################################
#############################################################################################

##########################
##### Segmentation #######
##########################

def Segment_str_langdetect(text,
						   sent_split=True,
						   tolist=True,
						   properties=None,
						   timeout=15000,
						   be_quiet=False,
						   chinese_only=False):
	'''
        Processes a string, detects if it is Chinese or English, and returns list of words nested in lists of sentences, or text split by spaces and newlines depending on parameters.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (bool) be_quiet: CoreNLPClient silent mode
        :param (bool) chinese_only: set to True to ignore English and other languages. Set to False to process English and Chinese.
                                    Ignoring English can save overhead, when faster tools are available.

        :return: segmented text in nested list or string

        Example:

        en_text = 'This is a test sentence for the server to handle. I wonder what it will do.'
        Segment_str_langdetect(en_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle', '.'], ['I', 'wonder', 'what', 'it', 'will', 'do', '.']]

        zh_text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
        Segment_str_langdetect(zh_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各', '地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']]

        Segment_str_langdetect(zh_text, sent_split=True, tolist=False, properties=None, timeout=15000, chinese_only=False)
        >>>'国务院 日前 发出 紧急 通知 ， 要求 各 地 切实 落实 保证 市场 供应 的 各 项 政策 ， 维护 副食品 价格 稳定 。'
    '''
	if sent_split:
		annotators = ['tokenize', 'ssplit']
	else:
		annotators = ['tokenize']
	words = []
	if text != '':
		if not sent_split:
			if not properties:
				properties = {'tokenize_no_ssplit': True}
				# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
			else:
				properties.update({'tokenize_no_ssplit': True})
				# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		##########
		try:
			lang = langdetect.detect(text)
		except langdetect.lang_detect_exception.LangDetectException:
			lang = "undetermined"
		if chinese_only:
			parse_ok = (lang == "zh-cn")
		else:
			parse_ok = (lang == "zh-cn") or (lang == "en")
		if parse_ok:
			if (lang == "zh-cn"):
				properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
			with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout,
							   be_quiet=be_quiet) as client:
				ann = client.annotate(text)
			words = [[token.word for token in sent.token] for sent in ann.sentence]
			segmented_list = [' '.join(wordlist) for wordlist in words]
			if sent_split:
				segmented = '\n'.join(segmented_list)
			else:
				words = [word for sent in words for word in sent]
				segmented = ' '.join(segmented_list)
		else:
			segmented = text
			words = segmented.split()
	else:
		segmented = text
	if tolist:
		return words  # list
	else:
		return segmented  # string


def Segment(text_list,
			sent_split=True,
			tolist=True,
			properties=None,
			timeout=15000,
			verbose=1,
			lang='zh-cn'):
	'''
        Processes a list of Chinese or English strings and returns list of words nested in lists of sentences, or a list of text split by spaces and newlines depending on parameters.
        It starts the server with the same properties for all texts, so all texts must be the same language, setup by the parameter :lang:. Default is Chinese lang='zh-cn'.

        :param (list[str] | tuple[str] | str) text_list: list of strings of raw text for the CoreNLPServer to parse
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (int) verbose: verbose level
                0: CoreNLPClient silent mode, no progress printing
                1: CoreNLPClient silent mode, progress printing
                2: CoreNLPClient silent mode off, no progress printing
                3: CoreNLPClient silent mode off, progress printing
        :param (str) lang: 'zh-cn' for Chinese and 'en' for English

        :return: list of segmented text in nested list or list of strings

        Example:

        en_texts = ['This is a test sentence for the server to handle. I wonder what it will do.']
        Segment(en_texts, sent_split=True, tolist=True, properties=None, timeout=15000, lang='en')
        >>>[[['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle', '.'], ['I', 'wonder', 'what', 'it', 'will', 'do', '.']]]

        zh_texts = ["国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"]
        Segment(zh_texts, sent_split=True, tolist=True, properties=None, timeout=15000, lang='zh-cn')
        >>>[[['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各', '地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']]]

        Segment(zh_texts, sent_split=True, tolist=False, properties=None, timeout=15000, lang='zh-cn')
        >>>[['国务院 日前 发出 紧急 通知 ， 要求 各 地 切实 落实 保证 市场 供应 的 各 项 政策 ， 维护 副食品 价格 稳定 。']]

    '''
	if type(text_list) == type(''):
		text_list = [text_list]
	if not sent_split:
		if not properties:
			properties = {'tokenize_no_ssplit': True}
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		else:
			properties.update({'tokenize_no_ssplit': True})
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
	if (lang == "zh-cn"):
		properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
	if sent_split:
		annotators = ['tokenize', 'ssplit']
	else:
		annotators = ['tokenize']
	if verbose == 0:
		be_quiet = True
		print_progress = False
	elif verbose == 1:
		be_quiet = True
		print_progress = True
	elif verbose == 2:
		be_quiet = False
		print_progress = False
	elif verbose == 3:
		be_quiet = False
		print_progress = True
	else:
		be_quiet = False
		print_progress = True
	result = []
	limit = len(text_list)
	with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout, be_quiet=be_quiet) as client:
		for i, text in enumerate(text_list):
			if print_progress:
				if lang == 'zh-cn':
					print("Segmenting Chinese sentence {} of {}".format(i + 1, limit))
				elif lang == 'en':
					print("SegmentingEnglish sentence {} of {}".format(i + 1, limit))
			if text != '':
				ann = client.annotate(text)
				words = [[token.word for token in sent.token] for sent in ann.sentence]
				segmented_list = [' '.join(wordlist) for wordlist in words]
				if sent_split:
					segmented = '\n'.join(segmented_list)
				else:
					words = [word for sent in words for word in sent]
					segmented = ' '.join(segmented_list)
			else:
				segmented = text
				words = []
			if tolist:
				result.append(words)  # list
			else:
				result.append(segmented)  # string
	return result


#########################
##### POS Tagging #######
#########################

def POS_Tag_str_langdetect(text,
						   sent_split=True,
						   pre_tokenized=True,
						   tolist=True,
						   properties=None,
						   timeout=15000,
						   be_quiet=False,
						   chinese_only=False):
	'''
        Processes a string, detects if it is Chinese or English, and returns a list of words paired in tuples with their tags, nested in lists of sentences;
        or text split by spaces and newlines depending on parameters, tagged delimited by #.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) pre_tokenized: Avoids loading the tokenizer if true. Assumes previously split words by spaces and sentences by newlines.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (bool) be_quiet: CoreNLPClient silent mode
        :param (bool) chinese_only: set to True to ignore English and other languages. Set to False to process English and Chinese.

        POS Tags explanation

        The Chinese tags used by Stanford NLP are the same as Penn Treebank POS Tags

        Penn Treebank POS tags:
        https://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports

        :return: segmented pairs of (word, tag) nested in sentences
            [   [(token, pos_tag), (token, pos_tag)],
                [(token, pos_tag), (token, pos_tag)],
            ]

            or string tagged by #, sentences delimited by newline.

            "token#pos_tag token#pos_tag
            token#pos_tag token#pos_tag"

        Example:

        en_text = 'This is a test sentence for the server to handle. I wonder what it will do.'
        POS_Tag_str_langdetect(en_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN'), ('for', 'IN'), ('the', 'DT'), ('server', 'NN'), ('to', 'TO'), ('handle', 'VB'), ('.', '.')], [('I', 'PRP'), ('wonder', 'VBP'), ('what', 'WP'), ('it', 'PRP'), ('will', 'MD'), ('do', 'VB'), ('.', '.')]]

        zh_text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
        POS_Tag_str_langdetect(zh_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[[('国务院', 'NN'), ('日前', 'NT'), ('发出', 'VV'), ('紧急', 'JJ'), ('通知', 'NN'), ('，', 'PU'), ('要求', 'VV'), ('各', 'DT'), ('地', 'NN'), ('切实', 'AD'), ('落实', 'VV'), ('保证', 'VV'), ('市场', 'NN'), ('供应', 'NN'), ('的', 'DEG'), ('各', 'DT'), ('项', 'M'), ('政策', 'NN'), ('，', 'PU'), ('维护', 'VV'), ('副食品', 'NN'), ('价格', 'NN'), ('稳定', 'NN'), ('。', 'PU')]]

        POS_Tag_str_langdetect(zh_text, sent_split=True, tolist=False, properties=None, timeout=15000, chinese_only=False)
        >>>'国务院#NN 日前#NT 发出#VV 紧急#JJ 通知#NN ，#PU 要求#VV 各#DT 地#NN 切实#AD 落实#VV 保证#VV 市场#NN 供应#NN 的#DEG 各#DT 项#M 政策#NN ，#PU 维护#VV 副食品#NN 价格#NN 稳定#NN 。#PU'
    '''
	annotators = ['pos']
	words = []
	if text != '':
		if pre_tokenized:
			if not properties:
				properties = {'tokenize_pretokenized': True}
				# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
			else:
				properties.update({'tokenize_pretokenized': True})
				# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
		if sent_split == False:
			if not properties:
				properties = {'tokenize_no_ssplit': True}
				# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
			else:
				properties.update({'tokenize_no_ssplit': True})
				# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		##########
		try:
			lang = langdetect.detect(text)
		except langdetect.lang_detect_exception.LangDetectException:
			lang = "undetermined"
		if chinese_only:
			parse_ok = (lang == "zh-cn")
		else:
			parse_ok = (lang == "zh-cn") or (lang == "en")
		if parse_ok:
			if (lang == "zh-cn"):
				properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
			with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout,
							   be_quiet=be_quiet) as client:
				ann = client.annotate(text)
			words = [[(token.word, token.pos) for token in sent.token] for sent in ann.sentence]
			segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
			if sent_split:
				segmented = '\n'.join(segmented_list)
			else:
				words = [(word, pos) for sent in words for word, pos in sent]
				segmented = ' '.join(segmented_list)
		else:
			segmented = text
			words = segmented.split()
	else:
		segmented = text
	if tolist:
		return words  # list
	else:
		return segmented  # string


def POS_Tag(text_list,
			sent_split=True,
			pre_tokenized=True,
			tolist=True,
			properties=None,
			timeout=15000,
			verbose=1,
			lang='zh-cn'):
	'''
        Processes a list of Chinese or English strings and returns lists of words paired in tuples with their tags, nested in lists of sentences, nested in lists of documents in text_list;
        or lists of text split by spaces and newlines depending on parameters, tagged delimited by #.
        It starts the server with the same properties for all texts, so all texts must be the same language, setup by the parameter :lang:. Default is Chinese lang='zh-cn'.

        :param (list[str] | tuple[str] | str) text_list: list of strings of raw text for the CoreNLPServer to parse
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) pre_tokenized: Avoids loading the tokenizer if true. Assumes previously split words by spaces and sentences by newlines.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (int) verbose: verbose level
                0: CoreNLPClient silent mode, no progress printing
                1: CoreNLPClient silent mode, progress printing
                2: CoreNLPClient silent mode off, no progress printing
                3: CoreNLPClient silent mode off, progress printing
        :param (str) lang: 'zh-cn' for Chinese and 'en' for English

        POS Tags explanation

        The Chinese tags used by Stanford NLP are the same as Penn Treebank POS Tags

        Penn Treebank POS tags:
        https://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports

        :return: segmented pairs of (word, tag) nested in sentences, nested in documents (determined at input)
            [[   [(token, pos_tag), (token, pos_tag)],
                [(token, pos_tag), (token, pos_tag)],
                ], # per document
            ...]

            or list of strings tagged by #, sentences delimited by newline.

            ["token#pos_tag token#pos_tag
            token#pos_tag token#pos_tag",...]

        Example:

        en_texts = ['This is a test sentence for the server to handle. I wonder what it will do.']
        POS_Tag(en_texts, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[[[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN'), ('for', 'IN'), ('the', 'DT'), ('server', 'NN'), ('to', 'TO'), ('handle', 'VB'), ('.', '.')], [('I', 'PRP'), ('wonder', 'VBP'), ('what', 'WP'), ('it', 'PRP'), ('will', 'MD'), ('do', 'VB'), ('.', '.')]]]

        zh_texts = ["国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"]
        POS_Tag(zh_texts, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
        >>>[[[('国务院', 'NN'), ('日前', 'NT'), ('发出', 'VV'), ('紧急', 'JJ'), ('通知', 'NN'), ('，', 'PU'), ('要求', 'VV'), ('各', 'DT'), ('地', 'NN'), ('切实', 'AD'), ('落实', 'VV'), ('保证', 'VV'), ('市场', 'NN'), ('供应', 'NN'), ('的', 'DEG'), ('各', 'DT'), ('项', 'M'), ('政策', 'NN'), ('，', 'PU'), ('维护', 'VV'), ('副食品', 'NN'), ('价格', 'NN'), ('稳定', 'NN'), ('。', 'PU')]]]

        POS_Tag(zh_texts, sent_split=True, tolist=False, properties=None, timeout=15000, chinese_only=False)
        >>>['国务院#NN 日前#NT 发出#VV 紧急#JJ 通知#NN ，#PU 要求#VV 各#DT 地#NN 切实#AD 落实#VV 保证#VV 市场#NN 供应#NN 的#DEG 各#DT 项#M 政策#NN ，#PU 维护#VV 副食品#NN 价格#NN 稳定#NN 。#PU']
    '''
	if type(text_list) == type(''):
		text_list = [text_list]
	if pre_tokenized:
		if not properties:
			properties = {'tokenize_pretokenized': True}
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
		else:
			properties.update({'tokenize_pretokenized': True})
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
	if sent_split == False:
		if not properties:
			properties = {'tokenize_no_ssplit': True}
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		else:
			properties.update({'tokenize_no_ssplit': True})
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
	if (lang == "zh-cn"):
		properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
	annotators = ['pos']
	if verbose == 0:
		be_quiet = True
		print_progress = False
	elif verbose == 1:
		be_quiet = True
		print_progress = True
	elif verbose == 2:
		be_quiet = False
		print_progress = False
	elif verbose == 3:
		be_quiet = False
		print_progress = True
	else:
		be_quiet = False
		print_progress = True
	result = []
	limit = len(text_list)
	with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout, be_quiet=be_quiet) as client:
		for i, text in enumerate(text_list):
			if print_progress:
				if lang == 'zh-cn':
					print("POS Tagging Chinese sentence {} of {}".format(i + 1, limit))
				elif lang == 'en':
					print("POS Tagging English sentence {} of {}".format(i + 1, limit))
			if text != '':
				ann = client.annotate(text)
				words = [[(token.word, token.pos) for token in sent.token] for sent in ann.sentence]
				segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
				if sent_split:
					segmented = '\n'.join(segmented_list)
				else:
					words = [(word, pos) for sent in words for word, pos in sent]
					segmented = ' '.join(segmented_list)
			else:
				segmented = text
				words = []
			if tolist:
				result.append(words)  # list
			else:
				result.append(segmented)  # string
	return result


def POS_Tag_str_tolist(pos_tag_str):
	'''
        In case of storing POS tags output from the method POS_Tag() in string form,
        this method returns it to nested list form.

        :param (str) pos_tag_str: POS tags string, sentences delimited by newline, in format:
           "token#pos_tag token#pos_tag
            token#pos_tag token#pos_tag"

        :return: POS tagged text in format:
            [   [(token, pos_tag), (token, pos_tag)],
                [(token, pos_tag), (token, pos_tag)],
            ]
    '''
	pos_tag_sentences = pos_tag_str.split('\n')
	pos_tag_tups = [sent.split(' ') for sent in pos_tag_sentences]
	pos_tags = [[tuple(tup.split('#')) for tup in sent] for sent in pos_tag_tups]
	return pos_tags


################################
##### Dependency Parsing #######
################################
########################
'''
    Dependency parsing is a bit harder, here's an example:

    en_text = 'This is a nice sentence for the server to handle. I wonder what it will do.'

    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'lemma', 'pos', 'depparse'],
                        properties=None, 
                        timeout=15000) as client:
        ann = client.annotate(en_text)

    sentence = ann.sentence[0]

    print(sentence.basicDependencies)

    Now let's explain the result. Each node is a word, the index is its number in the sentence.
    Each edge is a connection between the words. Let's look at the edge between 5 and 4:

    # edge {
    #   source: 5       # which is 'sentence'
    #   target: 4       # which is 'nice'
    #   dep: "amod"     # "amod" means adjective modifier
    #   isExtra: False  # 
    #   sourceCopy: 0
    #   targetCopy: 0
    #   language: UniversalEnglish
    # }
'''


########################


def Dependency_Parse_str_langdetect(text,
									dependency_type='basicDependencies',
									sent_split=False,
									pre_tokenized=True,
									tolist=True,
									output_with_sentence=True,
									properties=None,
									timeout=15000,
									be_quiet=False,
									chinese_only=False):
	'''
        Processes a string, detects if it is Chinese or English, and collects the dependency, source word and target word in a list of tuples nested in a list of sentences.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (str) dependency_type: Choose from the options Stanford NLP has available. Default basicDependencies.
                'alternativeDependencies'
                'basicDependencies'
                'collapsedCCProcessedDependencies'
                'collapsedDependencies'
                'enhancedDependencies'
                'enhancedPlusPlusDependencies'
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) pre_tokenized: Avoids loading the tokenizer if true. Assumes previously split words by spaces and sentences by newlines.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (bool) output_with_sentence: set to True (default) to get the segmented sentence as part of the output on top of the dependencies. Set to False to keep dependencies only.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (bool) be_quiet: CoreNLPClient silent mode
        :param (bool) chinese_only: set to True to ignore English and other languages. Set to False to process English and Chinese.

        Stanford NLP dependencies manual:
        https://nlp.stanford.edu/software/dependencies_manual.pdf

        :return: Tuple of sentence, and dependency list nested in a list of sentences
                if output_with_sentence==True:

                    [   (sentence,
                        [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                        ),
                        (sentence,
                        [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                        ),
                    ...]

                    or Dependency string formatted as follows:
                        sentence
                        dependency(source,target), dependency(source,target), ....

                        sentence
                        dependency(source,target), dependency(source,target), ....

                if output_with_sentence==False:

                    [   [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                        [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                    ...]

                    or Dependency string formatted as follows:
                        dependency(source,target), dependency(source,target), ....
                        dependency(source,target), dependency(source,target), ....

        Example:

        en_text = 'This is a test sentence for the server to handle. I wonder what it will do.'
        Dependency_Parse_str_langdetect(en_text, dependency_type='basicDependencies', sent_split=True, tolist=True, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False)
        >>> [   (['This','is','a','test','sentence','for','the','server','to','handle','.'],
                    [('nsubj', 'sentence', 'This'),
                    ('cop', 'sentence', 'is'),
                    ('det', 'sentence', 'a'),
                    ('compound', 'sentence', 'test'),
                    ('acl', 'sentence', 'handle'),
                    ('punct', 'sentence', '.'),
                    ('det', 'server', 'the'),
                    ('mark', 'handle', 'for'),
                    ('nsubj', 'handle', 'server'),
                    ('mark', 'handle', 'to')]
                ),

                (['I', 'wonder', 'what', 'it', 'will', 'do', '.'],
                    [('obj', 'do', 'what'),
                    ('nsubj', 'do', 'it'),
                    ('aux', 'do', 'will'),
                    ('ccomp', 'wonder', 'do'),
                    ('punct', 'wonder', '.'),
                    ('nsubj', 'wonder', 'I')]
                )
            ]

        print(Dependency_Parse_str_langdetect(en_text, dependency_type='basicDependencies', sent_split=True, tolist=False, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False))
        >>>
        This is a test sentence for the server to handle .
        nsubj(sentence,This), cop(sentence,is), det(sentence,a), compound(sentence,test), acl(sentence,handle), punct(sentence,.), det(server,the), mark(handle,for), nsubj(handle,server), mark(handle,to)

        I wonder what it will do .
        obj(do,what), nsubj(do,it), aux(do,will), ccomp(wonder,do), punct(wonder,.), nsubj(wonder,I)

        zh_text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
        Dependency_Parse_str_langdetect(zh_text, dependency_type='basicDependencies', sent_split=True, tolist=True, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False)
        >>>[(  ['国务院','日前','发出','紧急','通知','，','要求','各','地','切实','落实','保证','市场','供应','的','各','项','政策','，','维护','副食品','价格','稳定','。'],
                  [('nsubj', '发出', '国务院'),
                   ('nmod:tmod', '发出', '日前'),
                   ('dobj', '发出', '通知'),
                   ('punct', '发出', '，'),
                   ('conj', '发出', '要求'),
                   ('punct', '发出', '。'),
                   ('amod', '通知', '紧急'),
                   ('dobj', '要求', '地'),
                   ('ccomp', '要求', '落实'),
                   ('det', '地', '各'),
                   ('advmod', '落实', '切实'),
                   ('ccomp', '落实', '保证'),
                   ('dobj', '保证', '政策'),
                   ('punct', '保证', '，'),
                   ('conj', '保证', '维护'),
                   ('compound:nn', '供应', '市场'),
                   ('case', '供应', '的'),
                   ('mark:clf', '各', '项'),
                   ('det', '政策', '各'),
                   ('nmod:assmod', '政策', '供应'),
                   ('dobj', '维护', '稳定'),
                   ('compound:nn', '稳定', '副食品'),
                   ('compound:nn', '稳定', '价格')]
            )]

        print(Dependency_Parse_str_langdetect(zh_text, dependency_type='basicDependencies', sent_split=True, tolist=False, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False))
        >>>
        国务院 日前 发出 紧急 通知 ， 要求 各 地 切实 落实 保证 市场 供应 的 各 项 政策 ， 维护 副食品 价格 稳定 。
        nsubj(发出,国务院), nmod:tmod(发出,日前), dobj(发出,通知), punct(发出,，), conj(发出,要求), punct(发出,。), amod(通知,紧急), dobj(要求,地), ccomp(要求,落实), det(地,各), advmod(落实,切实), ccomp(落实,保证), dobj(保证,政策), punct(保证,，), conj(保证,维护), compound:nn(供应,市场), case(供应,的), mark:clf(各,项), det(政策,各), nmod:assmod(政策,供应), dobj(维护,稳定), compound:nn(稳定,副食品), compound:nn(稳定,价格)
    '''
	annotators = ['depparse']
	if pre_tokenized:
		if not properties:
			properties = {'tokenize_pretokenized': True}
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
		else:
			properties.update({'tokenize_pretokenized': True})
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
	if sent_split == False:
		if not properties:
			properties = {'tokenize_no_ssplit': True}
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		else:
			properties.update({'tokenize_no_ssplit': True})
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
	if text != '':
		if use_str_langdetect:
			try:
				lang = langdetect.detect(text)
			except langdetect.lang_detect_exception.LangDetectException:
				lang = "undetermined"
			if chinese_only:
				parse_ok = (lang == "zh-cn")
			else:
				parse_ok = (lang == "zh-cn") or (lang == "en")
		else:
			parse_ok = True
			if no_str_langdetect_chinese:
				lang = 'zh-cn'
		if parse_ok:
			if (lang == "zh-cn"):
				properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
			with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout,
							   be_quiet=be_quiet) as client:
				ann = client.annotate(text)
			#######
			deps = []
			if not tolist: deps_strs = []
			for sent in ann.sentence:
				words = dict([(i + 1, token.word) for i, token in enumerate(sent.token)])
				sentence_words = [token.word for token in sent.token]
				if output_with_sentence:
					deps_sent_str = ' '.join(sentence_words) + '\n'
				else:
					deps_sent_str = ''
				if dependency_type == None:
					depTree = sent.basicDependencies
				elif dependency_type == 'alternativeDependencies':
					depTree = sent.alternativeDependencies
				elif dependency_type == 'basicDependencies':
					depTree = sent.basicDependencies
				elif dependency_type == 'collapsedCCProcessedDependencies':
					depTree = sent.collapsedCCProcessedDependencies
				elif dependency_type == 'collapsedDependencies':
					depTree = sent.collapsedDependencies
				elif dependency_type == 'enhancedDependencies':
					depTree = sent.enhancedDependencies
				elif dependency_type == 'enhancedPlusPlusDependencies':
					depTree = sent.enhancedPlusPlusDependencies
				else:
					depTree = sent.basicDependencies
				if output_with_sentence:
					deps_sent = (
					sentence_words, [(edge.dep, words[edge.source], words[edge.target]) for edge in depTree.edge])
				else:
					deps_sent = [(edge.dep, words[edge.source], words[edge.target]) for edge in depTree.edge]
				deps.append(deps_sent)
				if not tolist:
					if output_with_sentence:
						deps_sent_str += ', '.join(
							['{}({},{})'.format(dep_tup[0], dep_tup[1], dep_tup[2]) for dep_tup in deps_sent[1]])
					else:
						deps_sent_str += ', '.join(
							['{}({},{})'.format(dep_tup[0], dep_tup[1], dep_tup[2]) for dep_tup in deps_sent])
					deps_strs.append(deps_sent_str)
			if not tolist:
				if output_with_sentence:
					deps_str = '\n\n'.join(deps_strs)
				else:
					deps_str = '\n'.join(deps_strs)
		else:
			deps = None
			deps_str = ''
	else:
		deps = None
		deps_str = ''
	if tolist:
		return deps
	else:
		return deps_str


def Dependency_Parse(text_list,
					 dependency_type='basicDependencies',
					 sent_split=False,
					 pre_tokenized=True,
					 tolist=True,
					 output_with_sentence=True,
					 properties=None,
					 timeout=15000,
					 verbose=1,
					 lang='zh-cn'):
	'''
        Processes a list of Chinese or English texts and collects the dependency, source word and target word in a list of tuples nested in a list of sentences, in a list of documents.

        :param (list[str] | tuple[str] | str) text_list: list of strings of raw text for the CoreNLPServer to parse
        :param (str) dependency_type: Choose from the options Stanford NLP has available. Default basicDependencies.
                'alternativeDependencies'
                'basicDependencies'
                'collapsedCCProcessedDependencies'
                'collapsedDependencies'
                'enhancedDependencies'
                'enhancedPlusPlusDependencies'
        :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
        :param (bool) pre_tokenized: Avoids loading the tokenizer if true. Assumes previously split words by spaces and sentences by newlines.
        :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
        :param (bool) output_with_sentence: set to True (default) to get the segmented sentence as part of the output on top of the dependencies. Set to False to keep dependencies only.
        :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
        :param (int) timeout: CoreNLP server time before raising exception.
        :param (int) verbose: verbose level
                0: CoreNLPClient silent mode, no progress printing
                1: CoreNLPClient silent mode, progress printing
                2: CoreNLPClient silent mode off, no progress printing
                3: CoreNLPClient silent mode off, progress printing
        :param (str) lang: 'zh-cn' for Chinese and 'en' for English

        Stanford NLP dependencies manual:
            https://nlp.stanford.edu/software/dependencies_manual.pdf

        Since updates, Stanford CoreNLP now uses Universal Dependencies.
        Universal Dependencies website:
        https://universaldependencies.org/#language-

        :return: List per document of: Tuple of sentence, and dependency list nested in a list of sentences
                if output_with_sentence==True:

                    [   [(sentence,
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                            ),
                         (sentence,
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                            ),
                            ...], # per document
                    ...]

                    or List per document of dependency string formatted as follows:
                      ["sentence
                        dependency(source,target), dependency(source,target), ....

                        sentence
                        dependency(source,target), dependency(source,target), ....", #per sentence
                        ...] # per document

                if output_with_sentence==False:

                    [   [   
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                        ...], # per document
                    ...]

                    or List per document of dependency string formatted as follows:
                        [   "dependency(source,target), dependency(source,target), ....
                             dependency(source,target), dependency(source,target), ....", # per document
                        ...]

        Example:

        en_texts = ['This is a test sentence for the server to handle. I wonder what it will do.']
        Dependency_Parse(en_text, dependency_type='basicDependencies', sent_split=True, tolist=True, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False)
        >>> [[   (['This','is','a','test','sentence','for','the','server','to','handle','.'],
                    [('nsubj', 'sentence', 'This'),
                    ('cop', 'sentence', 'is'),
                    ('det', 'sentence', 'a'),
                    ('compound', 'sentence', 'test'),
                    ('acl', 'sentence', 'handle'),
                    ('punct', 'sentence', '.'),
                    ('det', 'server', 'the'),
                    ('mark', 'handle', 'for'),
                    ('nsubj', 'handle', 'server'),
                    ('mark', 'handle', 'to')]
                ),

                (['I', 'wonder', 'what', 'it', 'will', 'do', '.'],
                    [('obj', 'do', 'what'),
                    ('nsubj', 'do', 'it'),
                    ('aux', 'do', 'will'),
                    ('ccomp', 'wonder', 'do'),
                    ('punct', 'wonder', '.'),
                    ('nsubj', 'wonder', 'I')]
                )
            ]]

        Dependency_Parse(en_texts, dependency_type='basicDependencies', sent_split=True, tolist=False, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False))
        >>>["This is a test sentence for the server to handle .
        nsubj(sentence,This), cop(sentence,is), det(sentence,a), compound(sentence,test), acl(sentence,handle), punct(sentence,.), det(server,the), mark(handle,for), nsubj(handle,server), mark(handle,to)

        I wonder what it will do .
        obj(do,what), nsubj(do,it), aux(do,will), ccomp(wonder,do), punct(wonder,.), nsubj(wonder,I)"]

        zh_texts = ["国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"]
        Dependency_Parse(zh_texts, dependency_type='basicDependencies', sent_split=True, tolist=True, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False)
        >>>[[(  ['国务院','日前','发出','紧急','通知','，','要求','各','地','切实','落实','保证','市场','供应','的','各','项','政策','，','维护','副食品','价格','稳定','。'],
                  [('nsubj', '发出', '国务院'),
                   ('nmod:tmod', '发出', '日前'),
                   ('dobj', '发出', '通知'),
                   ('punct', '发出', '，'),
                   ('conj', '发出', '要求'),
                   ('punct', '发出', '。'),
                   ('amod', '通知', '紧急'),
                   ('dobj', '要求', '地'),
                   ('ccomp', '要求', '落实'),
                   ('det', '地', '各'),
                   ('advmod', '落实', '切实'),
                   ('ccomp', '落实', '保证'),
                   ('dobj', '保证', '政策'),
                   ('punct', '保证', '，'),
                   ('conj', '保证', '维护'),
                   ('compound:nn', '供应', '市场'),
                   ('case', '供应', '的'),
                   ('mark:clf', '各', '项'),
                   ('det', '政策', '各'),
                   ('nmod:assmod', '政策', '供应'),
                   ('dobj', '维护', '稳定'),
                   ('compound:nn', '稳定', '副食品'),
                   ('compound:nn', '稳定', '价格')]
            )]]

        Dependency_Parse(zh_texts, dependency_type='basicDependencies', sent_split=True, tolist=False, output_with_sentence=True, pre_tokenized=False, properties=None, timeout=15000, chinese_only=False))
        >>> 
        ["国务院 日前 发出 紧急 通知 ， 要求 各 地 切实 落实 保证 市场 供应 的 各 项 政策 ， 维护 副食品 价格 稳定 。
        nsubj(发出,国务院), nmod:tmod(发出,日前), dobj(发出,通知), punct(发出,，), conj(发出,要求), punct(发出,。), amod(通知,紧急), dobj(要求,地), ccomp(要求,落实), det(地,各), advmod(落实,切实), ccomp(落实,保证), dobj(保证,政策), punct(保证,，), conj(保证,维护), compound:nn(供应,市场), case(供应,的), mark:clf(各,项), det(政策,各), nmod:assmod(政策,供应), dobj(维护,稳定), compound:nn(稳定,副食品), compound:nn(稳定,价格)"]
    '''
	if type(text_list) == type(''):
		text_list = [text_list]
	if pre_tokenized:
		if not properties:
			properties = {'tokenize_pretokenized': True}
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
		else:
			properties.update({'tokenize_pretokenized': True})
			# Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
	if sent_split == False:
		if not properties:
			properties = {'tokenize_no_ssplit': True}
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
		else:
			properties.update({'tokenize_no_ssplit': True})
			# Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
	if lang == "zh-cn":
		properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
	annotators = ['depparse']
	if verbose == 0:
		be_quiet = True
		print_progress = False
	elif verbose == 1:
		be_quiet = True
		print_progress = True
	elif verbose == 2:
		be_quiet = False
		print_progress = False
	elif verbose == 3:
		be_quiet = False
		print_progress = True
	else:
		be_quiet = False
		print_progress = True
	result = []
	limit = len(text_list)
	with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout, be_quiet=be_quiet) as client:
		for i, text in enumerate(text_list):
			if print_progress:
				if lang == 'zh-cn':
					print("Dependency Parsing Chinese sentence {} of {}".format(i + 1, limit))
				elif lang == 'en':
					print("Dependency Parsing English sentence {} of {}".format(i + 1, limit))
			if text != '':
				ann = client.annotate(text)
				#######
				deps = []
				if not tolist: deps_strs = []
				for sent in ann.sentence:
					words = dict([(i + 1, token.word) for i, token in enumerate(sent.token)])
					sentence_words = [token.word for token in sent.token]
					if output_with_sentence:
						deps_sent_str = ' '.join(sentence_words) + '\n'
					else:
						deps_sent_str = ''
					if dependency_type == None:
						depTree = sent.basicDependencies
					elif dependency_type == 'alternativeDependencies':
						depTree = sent.alternativeDependencies
					elif dependency_type == 'basicDependencies':
						depTree = sent.basicDependencies
					elif dependency_type == 'collapsedCCProcessedDependencies':
						depTree = sent.collapsedCCProcessedDependencies
					elif dependency_type == 'collapsedDependencies':
						depTree = sent.collapsedDependencies
					elif dependency_type == 'enhancedDependencies':
						depTree = sent.enhancedDependencies
					elif dependency_type == 'enhancedPlusPlusDependencies':
						depTree = sent.enhancedPlusPlusDependencies
					else:
						depTree = sent.basicDependencies
					if output_with_sentence:
						deps_sent = (
						sentence_words, [(edge.dep, words[edge.source], words[edge.target]) for edge in depTree.edge])
					else:
						deps_sent = [(edge.dep, words[edge.source], words[edge.target]) for edge in depTree.edge]
					deps.append(deps_sent)
					if not tolist:
						if output_with_sentence:
							deps_sent_str += ', '.join(
								['{}({},{})'.format(dep_tup[0], dep_tup[1], dep_tup[2]) for dep_tup in deps_sent[1]])
						else:
							deps_sent_str += ', '.join(
								['{}({},{})'.format(dep_tup[0], dep_tup[1], dep_tup[2]) for dep_tup in deps_sent])
						deps_strs.append(deps_sent_str)
				if not tolist:
					if output_with_sentence:
						deps_str = '\n\n'.join(deps_strs)
					else:
						deps_str = '\n'.join(deps_strs)
			else:
				if output_with_sentence:
					deps = [([None], [(None, None, None)])]
				else:
					deps = [[(None, None, None)]]
				deps_str = ''
			if tolist:
				result.append(deps)
			else:
				result.append(deps_str)
	return result


# sometimes returns unshapely tuples, maybe broken by punctuation as words
def Dependency_Parse_str_tolist(dep_parse_str, output_with_sentence=True):
	'''
        In case of storing Dependency_Parse() output in string form,
        this method returns it to nested list form.

        :param (str) dep_parse_str: Dependency_Parse() output, sentences delimited by newline or double newline
            if output_with_sentence==True, in format:
               "sentence
                dependency(source,target), dependency(source,target),...

                sentence
                dependency(source,target), dependency(source,target),..."
            if output_with_sentence==False, in format:
               "dependency(source,target), dependency(source,target),...
                dependency(source,target), dependency(source,target),..."  
        :param (bool) output_with_sentence: set to True (default) to have the segmented sentence as part of the input on top of the dependencies. Set to False to input dependencies only.

        :return: Dependency_Parse() tolist output:
            if output_with_sentence==True, in format:
                [   (sentence, 
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                            ),
                            (sentence, 
                            [(dependency, source_word, target_word),(dependency, source_word, target_word)]
                            ),
                ...]
            if output_with_sentence==False, in format:
                [   [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                    [(dependency, source_word, target_word),(dependency, source_word, target_word)],
                ...]
    '''
	deps = []
	if output_with_sentence:
		parse_sentences = dep_parse_str.split('\n\n')
		for sent in parse_sentences:
			parse_tup_per_sent = tuple(sent.split('\n'))
			sentence_words = parse_tup_per_sent[0].split(' ')
			tup_list = parse_tup_per_sent[1].split(', ')
			tup_new_list = [tuple(tup.replace('(', ' ').replace(',', ' ').replace(')', '').split(' ')) for tup in
							tup_list]
			ins = (sentence_words, tup_new_list)
			deps.append(ins)
	else:
		parse_sentences = dep_parse_str.split('\n')
		for sent in parse_sentences:
			tup_list = sent.split(', ')
			tup_new_list = [tuple(tup.replace('(', ' ').replace(',', ' ').replace(')', '').split(' ')) for tup in
							tup_list]
			deps.append(tup_new_list)
	return deps


if __name__ == '__main__':
	pass
