import ddparser
import json
from extract import FineGrainedInfo
from extract import CoarseGrainedInfo
import time
import argparse
import re
import sys
import copy
import random


# import jieba
# import jieba.posseg as pseg

def filter_rel_list(rellist, reltype='SVO', threshold=2000):
	kept_rels = []
	kept_rel_idxs = []
	for rel_idx, rel in enumerate(rellist):
		if rel[1] == reltype:
			kept_rels.append(copy.deepcopy(rel))
			kept_rel_idxs.append(rel_idx)
	assert len(kept_rels) == len(kept_rel_idxs)
	if len(kept_rels) > threshold:
		print(f"Number of kept rels is {len(kept_rels)}, larger than {threshold}! Sampled to {threshold} meet computational need!")
		new_kept_rels = []
		new_kept_rel_idxs = []
		all_nums = [i for i in range(len(kept_rels))]
		sampled_nums = random.sample(all_nums, k=threshold)
		for num in sampled_nums:
			new_kept_rel_idxs.append(kept_rel_idxs[num])
			new_kept_rels.append(kept_rels[num])
		kept_rel_idxs = new_kept_rel_idxs
		kept_rels = new_kept_rels

	assert len(kept_rels) == len(kept_rel_idxs)
	assert len(kept_rels) <= threshold
	return kept_rel_idxs, kept_rels


def serialize_rel(rel):
	assert len(rel) == 3
	assert len(rel[0]) == 3
	assert len(rel[2]) == 3
	string = f"{rel[0][0]}::{rel[0][1]}::{rel[0][2]}::::{rel[1]}::::{rel[2][0]}::{rel[2][1]}::{rel[2][2]}"
	return string


def acquire_coarse_span(arg_idx, ddp_sent_res):
	arg_idx_from_1 = arg_idx + 1
	for widx in range(len(ddp_sent_res['word'])):
		if ddp_sent_res['head'][widx] != arg_idx_from_1:
			continue


# reformat from (([S_id, S], [V_id, V], [O_id, O]), 'SVO') into ((S, V, O), 'SVO', (S_id, V_id, O_id))
def reformat_cur_rels(cur_rels):
	reformatted = []
	for sent_id, sent_rels in enumerate(cur_rels):
		sent_reformatted = []
		for rel in sent_rels:
			new_rel_0 = []
			new_rel_2 = []
			for item in rel[0]:
				if item is None:
					new_rel_0.append(None)
					new_rel_2.append(None)
				else:
					new_rel_0.append(item[1])
					new_rel_2.append(item[0])
			new_rel = (new_rel_0, rel[1], new_rel_2)

			sent_reformatted.append(new_rel)
		reformatted.append(sent_reformatted)

	return reformatted


def find_children_nodes_from_0(ddp_sent_res, node_idx_from_0):
	node2_idx_from_0_list = []
	node_idx_from_1 = node_idx_from_0 + 1
	for node2_idx_from_0, node_2_head in enumerate(ddp_sent_res['head']):
		if node_2_head == node_idx_from_1:
			node2_idx_from_0_list.append(node2_idx_from_0)
	return node2_idx_from_0_list


def find_bei_from_children(ddp_sent_res, head_idx_from_0):
	children_list_from_0 = find_children_nodes_from_0(ddp_sent_res, head_idx_from_0)
	for ch_idx in children_list_from_0:
		if ddp_sent_res['word'][ch_idx] in ['被']:
			return True
	return False


# re-segment the sentence to acquire the POS tags with jieba
'''
def reseg_and_postag(sent):
	raise AssertionError
	words = pseg.cut(sent)
	ret = {}
	for (word, flag) in words:
		if word not in ret:
			ret[word] = []
		ret[word].append(flag)
	for key in ret:
		ret[key] = list(set(ret[key]))
	return ret


# a wrapper class for corenlp annotator, provides information on NER mentions/spans and POS tags
class CorenlpAnnotator():
	def __init__(self, stats_out_fn):
		self.ner_token_bucket = {}
		self.total_count_special_token_misplaced_spans = 0
		self.client = CoreNLPClient(
			annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
			properties=StanfordCoreNLP_chinese_properties,
			timeout=30000,
			memory='16G')
		self.stats_out_fn = stats_out_fn

	# borrowed from StanfordCoreNLP_Chinese/main_ner.py
	def fetch_corenlp_annotation(self, sentences):
		corenlp_ner_mentions = []
		corenlp_ner_spans = []
		corenlp_pos_tags = []  # List(Dict(word: List(pos_tags)))
		for sent_id, sent in enumerate(doc['splitted_text']):
			ann = self.client.annotate(sent)
			sent_ner_mentions = []
			sent_ner_spans = []
			sent_pos_tags = {}
			for ann_sentence in ann.sentence:
				for token in ann_sentence.token:
					if token.ner not in ['O', 'PERCENT']:
						if token.ner not in self.ner_token_bucket:
							self.ner_token_bucket[token.ner] = 1
						else:
							self.ner_token_bucket[token.ner] += 1
						span = [token.beginChar, token.endChar]
						if token.word != sent[span[0]:span[1]]:
							print("sentence: ", sent)
							for _tok in ann_sentence.token:
								print(
									f"{_tok.word}; [{_tok.beginChar}, {_tok.endChar}]; {sent[_tok.beginChar:_tok.endChar]}")
							print(f"token.word: {token.word}")
							print(f"sent[span[0]:span[1]]: {sent[span[0]:span[1]]}")
							self.total_count_special_token_misplaced_spans += 1
							# raise AssertionError
						mention = {'word': token.word, 'label': token.ner, 'span': span}
						sent_ner_mentions.append(mention)
						sent_ner_spans.append(span)
					if token.word not in sent_pos_tags:
						sent_pos_tags[token.word] = []
					sent_pos_tags[token.word].append(token.pos)
			for word in sent_pos_tags:
				sent_pos_tags[word] = list(set(sent_pos_tags[word]))

			corenlp_ner_mentions.append(sent_ner_mentions)
			corenlp_ner_spans.append(sent_ner_spans)
			corenlp_pos_tags.append(sent_pos_tags)
		return corenlp_ner_mentions, corenlp_ner_spans, corenlp_pos_tags

	def print_stats(self):
		print("ner_token_bucket: ")
		print(self.ner_token_bucket)
		print("total_count_special_token_misplaced_spans: ")
		print(self.total_count_special_token_misplaced_spans)
		with open(self.stats_out_fn, 'w', encoding='utf8') as fp:
			json.dump(self.ner_token_bucket, fp, ensure_ascii=False)


'''


# examines whether a token is of acceptable part-of-speech in at least one mention of it in this sentence.
def examine_postags(obj_token, pos_tags, acceptable_list):
	if pos_tags is None:
		return False
	if obj_token not in pos_tags:
		return False
	flags = pos_tags[obj_token]
	for flag in flags:
		if flag in acceptable_list:
			return True
	return False


# OBSOLETE
# add a dummy () into ((S, V, O), 'SVO', ()) in order to maintain same structure as fine_rels
'''
def reformat_coarse_rels(coarse_rels):
	reformatted = []
	for sent_id, sent_rels in enumerate(coarse_rels):
		sent_reformatted = []
		for rel in sent_rels:
			new_rel = [rel[0], rel[1], ()]
			sent_reformatted.append(new_rel)
		reformatted.append(sent_reformatted)
	return reformatted
'''


# merge entries in dict_2 into dict_1
def merge_dict(dict_1, dict_2):
	for key in dict_2:
		if key not in dict_1:
			dict_1[key] = dict_2[key]
		else:
			dict_1[key] += dict_2[key]
	return


# disgard all relation tuples except those triples for SVO relations; pure output for downstream linking & typing.
def only_keep_svo(cur_rels, entry_threshold=3000):
	pruned_cur_rels = []
	for sent_id, sent_cur_rels in enumerate(cur_rels):
		sent_pruned_cur_rels = []
		for rel in sent_cur_rels:
			if rel[1] == 'SVO':
				sent_pruned_cur_rels.append(rel)

		if len(sent_pruned_cur_rels) <= entry_threshold:
			return_sent_pruned_cur_rels = sent_pruned_cur_rels
		else:
			return_sent_pruned_cur_rels = random.sample(sent_pruned_cur_rels, k=entry_threshold)
			print(f"sentid {sent_id}; returning SVO triples pruned from {len(pruned_cur_rels)} to threshold {entry_threshold}")

		pruned_cur_rels.append(return_sent_pruned_cur_rels)

	return pruned_cur_rels


# for each DOB relation, add another two SVO relations to the relation list (split this DOB up into 2)
def translate_nary_to_binaries(cur_rels):
	translated_cur_rels = []
	for sent_id, sent_cur_rels in enumerate(cur_rels):
		sent_translated_cur_rels = []
		for rel in sent_cur_rels:
			if rel[1] != 'DOB':
				sent_translated_cur_rels.append(rel)
			else:
				rel1 = [[rel[0][0], rel[0][1], rel[0][2]], 'SVO', [rel[2][0], rel[2][1], rel[2][2]]]
				rel2 = [[rel[0][0], rel[0][1], rel[0][3]], 'SVO', [rel[2][0], rel[2][1], rel[2][3]]]
				sent_translated_cur_rels.append(rel1)
				sent_translated_cur_rels.append(rel2)
				sent_translated_cur_rels.append(rel)
		translated_cur_rels.append(sent_translated_cur_rels)
	return translated_cur_rels


def no_chinese_char(string):
	if string is None or len(string) == 0:
		return False
	chinese_character_list = re.findall(r'[\u4e00-\u9fff]+', string)
	chinese_character_list = ''.join(chinese_character_list)
	if len(chinese_character_list) > 0:
		return False
	else:
		return True


# input: fine_grained and coarse_grained relations in a document, plus some globally-maintained statistics
#
# output: filtered fine and coarse grained relations in a document (a list of sentences), plus some globally-maintained
# statistics
def filter_triples_stopwords(cur_rels, stop_word_list, cur_stop_word_count_bucket=None, cur_number_count=0,
							 MUST_INCLUDE_CHINESE_flag=True, arg_len_threshold=20):
	filtered_cur_rels = []

	def all_digits(string):
		if string is None or len(string) == 0:
			return False
		digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
		for c in string:
			if c not in digits:
				return False
		return True

	for sid, sent_cur_rels in enumerate(cur_rels):
		filtered_sent_cur_rels = []
		for rel in sent_cur_rels:
			if rel[1] != 'SVO':
				filtered_sent_cur_rels.append(rel)
			else:
				subj = rel[0][0]
				pred = rel[0][1]
				obj = rel[0][2]
				skip = False
				subj_null_flag = False
				obj_null_flag = False
				if subj is not None and len(subj) > arg_len_threshold:
					skip = True
				if subj in stop_word_list:
					skip = True
					if cur_stop_word_count_bucket is not None:
						cur_stop_word_count_bucket[subj] += 1
				elif all_digits(subj) or (MUST_INCLUDE_CHINESE_flag and no_chinese_char(subj)):
					skip = True
					cur_number_count += 1
				elif subj is not None and len(subj.strip()) == 0:
					subj_null_flag = True

				if obj is not None and len(obj) > arg_len_threshold:
					skip = True
				if obj in stop_word_list:
					skip = True
					if cur_stop_word_count_bucket is not None:
						cur_stop_word_count_bucket[obj] += 1
				elif all_digits(obj) or (MUST_INCLUDE_CHINESE_flag and no_chinese_char(subj)):
					skip = True
					cur_number_count += 1
				elif obj is not None and len(obj.strip()) == 0:
					obj_null_flag = True

				if len(pred.strip()) == 0:
					skip = True

				if subj_null_flag and obj_null_flag:
					skip = True

				if skip is False:
					assert len(rel) == 3
					if subj_null_flag and rel[0][2] is not None:
						filtered_sent_cur_rels.append([[None, rel[0][1], rel[0][2]], rel[1], [None, rel[2][1], rel[2][2]]])
					elif obj_null_flag and rel[0][0] is not None:
						filtered_sent_cur_rels.append([[rel[0][0], rel[0][1], None], rel[1], [rel[2][0], rel[2][1], None]])
					else:
						filtered_sent_cur_rels.append(rel)
		filtered_cur_rels.append(filtered_sent_cur_rels)

	return filtered_cur_rels, cur_number_count


class Token_Normalizer:
	def __init__(self, remove_from_args):
		self.coarse_pred_change_count = 0
		self.fine_pred_change_count = 0
		self.coarse_args_change_count = 0
		self.fine_args_change_count = 0
		self.remove_from_args = remove_from_args

	def rel_token_normalize(self, fine_rels, coarse_rels):
		assert len(fine_rels) == len(coarse_rels)
		for sent_id in range(len(fine_rels)):
			for f_rel in fine_rels[sent_id]:
				if f_rel[1] != 'SVO':
					continue
				if self.remove_from_args and f_rel[0][0] is not None and '·' in f_rel[0][0]:
					print("· in fine-grained relation!")
					f_rel[0][0] = f_rel[0][0].replace('·', '')
					self.fine_args_change_count += 1
				if f_rel[0][1] is not None and '·' in f_rel[0][1]:
					print("· in fine-grained relation!")
					f_rel[0][1] = f_rel[0][1].replace('·', '')
					self.fine_pred_change_count += 1
				if self.remove_from_args and f_rel[0][2] is not None and '·' in f_rel[0][2]:
					print("· in fine-grained relation!")
					f_rel[0][2] = f_rel[0][2].replace('·', '')
					self.fine_args_change_count += 1
			for c_rel in coarse_rels[sent_id]:
				if c_rel[1] != 'SVO':
					continue
				if self.remove_from_args and c_rel[0][0] is not None and '·' in c_rel[0][0]:
					print("· in coarse-grained relation!")
					c_rel[0][0] = c_rel[0][0].replace('·', '')
					self.coarse_args_change_count += 1
				if c_rel[0][1] is not None and '·' in c_rel[0][1]:
					print("· in coarse-grained relation!")
					c_rel[0][1] = c_rel[0][1].replace('·', '')
					self.coarse_pred_change_count += 1
				if self.remove_from_args and c_rel[0][2] is not None and '·' in c_rel[0][2]:
					print("· in coarse-grained relation!")
					c_rel[0][2] = c_rel[0][2].replace('·', '')
					self.coarse_args_change_count += 1

	def print_stats(self):
		print(f"coarse pred change count: {self.coarse_pred_change_count};")
		print(f"fine pred change count: {self.fine_pred_change_count};")
		print(f"coarse args change count: {self.coarse_args_change_count};")
		print(f"fine args change count: {self.fine_args_change_count}!")


def merge_orig_amend_prune(orig, amend, prune_idxs, discard_residue):
	merged = []
	for oid, orel in enumerate(orig):
		if (oid not in prune_idxs) or (not discard_residue):
			merged.append(orel)
		else:
			print("!")
	merged += amend
	return merged


def merge_orig_amend_noprune(orig, amend, prune_idxs, discard_residue):
	merged = orig + amend
	return merged


merge_orig_amend = merge_orig_amend_noprune


# build amendment relations for documents, each containing k sentences
def build_amendment_relations(ddp_res, fine_rels, coarse_rels, pos_tags, DEBUG, discard_residue, ignore_postag=False):

	nightly = False

	POB_flag = True
	MT_flag = True
	VV_flag = True
	VCMP_flag = True
	HEAD_NEG_flag = False
	ATT_OBJ_flag = True
	ATT_POB_flag = True
	ATT_SUB_flag = True  # sub-structure for argument-modifier structures.
	ATT_ATT_flag = True
	VOB_VOB_flag = True
	COP_SBJ_flag = True
	COP_ADJ_flag = True
	ALL_ADV_pivot_flag = False
	Prep_nary_flag = True  # whether or not to include the relation with prep-object argument pair for trinary relations resulting from prepositions: Barack Obama visited Hawaii last Friday.
	Discard_POB_where_ADV_rel_not_found_flag = False
	progressive_flag = True  # whether the result of a previous amendment is used in later amendments.

	amendment_counts = {'pob': 0, 'mt': 0, 'vv': 0, 'vcmp': 0, 'headneg': 0, 'attobj': 0, 'attsub': 0, 'attatt': 0,
						'vobvob': 0, 'copsbj': 0, 'attpob': 0, 'copadj': 0}

	pivot_adv_list = ['与', '和', '跟', '同',  # conjunction
					  '在', '于', '自', '自从', '从', '当', '由', '趁', '趁着', '随着', '到',  # time
					  '按照', '通过', '按', '比', '拿', '以', '凭', '凭借',  # method
					  '为', '为了', '由于', '因为',  # reason
					  '对', '对于', '关于', '除了', '除', '让', '给', '使得', '使',
					  '朝', '向', '沿', '沿着', '顺', '顺着']  # for POB

	cop_pred_set = {'是', '也是', '就是', '而是', '正是', '才是', '都是', '仍是', '既是', '又是', '却是', '只是', '算是', '竟是',
					'便是', '无疑是', '乃是', '并且是', '达', '高达'}  # '为'}

	# acceptable_postag_list = ['n', 'f', 's', 't', 'nr', 'ns', 'nt', 'nw', 'nz', 'v', 'vd', 'vn', 'r', 'PER', 'LOC', 'ORG',
	# 						  'TIME']
	acceptable_postag_list = ['FW', 'NN', 'NR', 'NT', 'PN', 'VV']
	noun_postag_list = ['FW', 'NN', 'NR', 'NT', 'PN']
	modif_noun_list = ['FW', 'NN', 'NR']
	all_postag_list = ['AD', 'CC', 'CD', 'DT', 'FW', 'JJ', 'LC', 'M', 'NN', 'NR', 'NT', 'OD', 'P', 'PN', 'PU', 'SP',
					   'VA', 'VV']
	elided_ATT_list = ['对', '']
	punctuation_list = ['。', '？', '！', '，', '、', '；', '：', '"', '（', '）', '《', '》', '【', '】', '……', '“', '”',
						'「', '」', '～']

	vcmp_dict = {}

	amend_fine_rels = []  # the amended fine-grained relations of the whole document
	amend_coarse_rels = []  # the amended coarse-grained relations of the whole document
	pruned_fine_rels = []
	pruned_coarse_rels = []
	possible_fine_rels = []
	assert len(ddp_res) == len(fine_rels)
	assert len(ddp_res) == len(coarse_rels)
	for sent_id in range(len(ddp_res)):
		# Example ddp:
		# [{'word': ['张三', '急匆匆', '地', '赶往', '机场', '。'], 'head': [4, 4, 2, 0, 4, 4], 'deprel': ['SBV', 'ADV', 'MT', 'HED', 'VOB', 'MT']}]
		r = ddp_res[sent_id]
		f = fine_rels[sent_id]
		c = coarse_rels[sent_id]

		p = pos_tags[sent_id] if pos_tags is not None else None  # pos_tags per sentence
		a_f = []  # sentence-wise amended fine_grained tokens
		a_c = []  # sentence-wise amended coarse_grained tokens
		possible_f = []
		j_f = f
		j_c = c
		# These indices below actually refer to the j_f and j_c instead of pure f and c,
		# but j_f and j_c are just appending to the f and c, thus should not influence the previous indices.
		f_residue_idxs = set()
		c_residue_idxs = set()

		# POB
		# [{'word': ['中国', '与', '印度', '接壤'], 'head': [4, 4, 2, 0], 'deprel': ['SBV', 'ADV', 'POB', 'HED']}]
		# Idea: if we find nouns with POB relation, headed by an ADV or CMP, and with SVO head Vs as grandparents,
		# then if this SVO relations has empty object position (or if the POB token is nominal or verbal in its nature),
		# that means this object-of-preposition (POB) should in fact be the object of this SVO relation, and we amend this;
		# the relation is not always symmetrical, so we do not add another relation swapping the subject and the object.
		# class 2: [{'word': ['李四光', '为', '科学', '做出', '了', '贡献', '。'], 'head': [4, 4, 2, 0, 4, 4, 4], 'deprel': ['SBV', 'ADV', 'POB', 'HED', 'MT', 'VOB', 'MT']}]
		# class 2: [(([2, '科学'], [3, '做出']), 'ADV_V'), (([0, '李四光'], [3, '做出'], [5, '贡献']), 'SVO')]
		#
		# Careful! In the second condition, pairs of
		if POB_flag is True:
			for lbl_idx, lbl in enumerate(r['deprel']):
				if lbl == 'POB':
					obj_token = r['word'][lbl_idx]
					adv_idx = r['head'][lbl_idx] - 1
					adv_token = r['word'][adv_idx]
					obj_token_is_nominal = ignore_postag or examine_postags(obj_token, p, acceptable_postag_list)
					# if the adverb token does not lie in the list where their POB can be linked with their parent verb:
					if adv_token not in pivot_adv_list and not ALL_ADV_pivot_flag:
						continue
					if adv_idx >= 0 and r['deprel'][adv_idx] in ['ADV',
																 'CMP']:  # if the POB-labelled token is controlled by an ADV
						verb_idx = r['head'][adv_idx] - 1
						# if there exists a relation triple heading with this verb grandparent of the POB,
						# and the relation has empty object, then put the POB in the object position
						if verb_idx < 0:
							continue
						verb_token = r['word'][verb_idx]

						# Process the fine-grained relations.
						for f_rel_idx, f_rel in enumerate(j_f):
							if f_rel[1] != 'SVO':
								continue
							# if the SVO relation it matches lacks an object, put this to the object position.
							if f_rel[2][1] == verb_idx and f_rel[0][2] is None:
								if f_rel[0][1] != verb_token:  # This should not happen here, since POB is the first amendment!
									if DEBUG:
										print(f"POB Diff tokens! {f_rel[0][1]}; {verb_token}")
								POB_matching_flag = False
								for f_rel_idx_ref, f_rel_ref in enumerate(j_f):
									if f_rel_ref[1] == 'ADV_V' and f_rel_ref[0][1] == verb_token and \
											obj_token in f_rel_ref[0][0]:
										POB_matching_flag = True
								if POB_matching_flag is False and Discard_POB_where_ADV_rel_not_found_flag:
									continue

								assert f_rel[0][1].count('·X·') == 0
								f_rel_new = ((f_rel[0][0], adv_token + '·X·' + f_rel[0][1], obj_token), f_rel[1],
											 (f_rel[2][0], f_rel[2][1], lbl_idx))
								if DEBUG:
									print(f"POB:		Instance found (fine)!")
									print(f"POB: 		POB_matching_flag: {POB_matching_flag}")
									print(f"POB:		{f_rel_new[0][0]}; {f_rel_new[0][1]}; {f_rel_new[0][2]}")
									print(f"POB:		{r['word']}")
								a_f.append(f_rel_new)
								f_residue_idxs.add(f_rel_idx)
								amendment_counts['pob'] += 1
							# elsewise, if the SVO relation has an object, but the POB token is indeed a Nominal
							# or Verbal construction, put this to the object position.
							elif f_rel[2][1] == verb_idx and obj_token_is_nominal:
								if f_rel[0][1] != verb_token:  # This should not happen here, since POB is the first amendment!
									if DEBUG:
										print(f"POB Diff tokens! {f_rel[0][1]}; {verb_token}")
								POB_matching_flag = False
								for f_rel_idx_ref, f_rel_ref in enumerate(j_f):
									if f_rel_ref[1] == 'ADV_V' and f_rel_ref[0][1] == verb_token and \
											obj_token in f_rel_ref[0][0]:
										POB_matching_flag = True
								if POB_matching_flag is False and Discard_POB_where_ADV_rel_not_found_flag:
									continue
								assert f_rel[0][1].count('·X·') == 0
								f_rel_new_1 = (
									(f_rel[0][0], adv_token + '·X·' + f_rel[0][1] + '·' + f_rel[0][2], obj_token),
									f_rel[1],
									(f_rel[2][0], f_rel[2][1], lbl_idx))
								f_rel_new_2 = (
									(f_rel[0][0], adv_token+'·X·'+f_rel[0][1], obj_token),
									f_rel[1],
									(f_rel[2][0], f_rel[2][1], lbl_idx))
								if DEBUG:
									print(f"POB:		Instance found (fine)!")
									print(f"POB: 		POB_matching_flag: {POB_matching_flag}")
									print(f"POB:		{f_rel_new_1[0][0]}; {f_rel_new_1[0][1]}; {f_rel_new_1[0][2]}")
									print(f"POB:		{f_rel_new_2[0][0]}; {f_rel_new_2[0][1]}; {f_rel_new_2[0][2]}")
									print(f"POB:		{r['word']}")
								a_f.append(f_rel_new_1)
								a_f.append(f_rel_new_2)
								# f_residue_idxs.append(f_rel_idx)
								amendment_counts['pob'] += 2

						# process the coarse-grained relations.
						for c_rel_idx, c_rel in enumerate(j_c):
							if c_rel[1] != 'SVO':
								continue
							# if the SVO relation it matches lacks an object, put this to the object position.
							if c_rel[2][1] == verb_idx and c_rel[0][2] is None:
								if c_rel[0][
									1] != verb_token:  # This should not happen here, since POB is the first amendment!
									if DEBUG:
										print(f"POB Diff tokens! {c_rel[0][1]}; {verb_token}")
								coarse_obj_token = None
								for c_rel_idx_ref, c_rel_ref in enumerate(j_c):
									if c_rel_ref[1] == 'ADV_V' and c_rel_ref[0][1] == verb_token and \
											obj_token in c_rel_ref[0][0]:
										coarse_obj_token = c_rel_ref[0][0]
								if coarse_obj_token is None:
									if Discard_POB_where_ADV_rel_not_found_flag:
										continue
									else:
										coarse_obj_token = obj_token
								if adv_token in c_rel[0][
									1]:  # then the adverb is subsumed in the coarse-grained predicate.
									continue
								assert '·X·' not in c_rel[0][1]
								c_rel_new = ((c_rel[0][0], adv_token + '·X·' + c_rel[0][1], coarse_obj_token), c_rel[1],
											 (c_rel[2][0], c_rel[2][1], lbl_idx))
								if DEBUG:
									print(f"POB:		Instance found (coarse)!")
									print(f"POB:		coarse_obj_token: {coarse_obj_token}")
									print(f"POB:		{c_rel_new[0][0]}; {c_rel_new[0][1]}; {c_rel_new[0][2]}")
									print(f"POB:		{r['word']}")
								a_c.append(c_rel_new)
								c_residue_idxs.add(c_rel_idx)
								amendment_counts['pob'] += 1

							# elsewise, if the SVO relation has an object, but the POB token is indeed a Nominal
							# or Verbal construction, put this to the object position.
							elif c_rel[2][1] == verb_idx and obj_token_is_nominal:
								if c_rel[0][
									1] != verb_token:  # This should not happen here, since POB is the first amendment!
									if DEBUG:
										print(f"POB Diff tokens! {c_rel[0][1]}; {verb_token}")
								coarse_obj_token = None
								for c_rel_idx_ref, c_rel_ref in enumerate(j_c):
									if c_rel_ref[1] == 'ADV_V' and c_rel_ref[0][1] == verb_token and \
											obj_token in c_rel_ref[0][0]:
										coarse_obj_token = c_rel_ref[0][0]
								if coarse_obj_token is None:
									if Discard_POB_where_ADV_rel_not_found_flag:
										continue
									else:
										coarse_obj_token = obj_token
								assert '·X·' not in c_rel[0][1]
								c_rel_new_1 = (
									(c_rel[0][0], adv_token+'·X·'+c_rel[0][1]+'·'+c_rel[0][2], coarse_obj_token),
									c_rel[1], (c_rel[2][0], c_rel[2][1], lbl_idx))
								c_rel_new_2 = (
									(c_rel[0][0], adv_token+'·X·'+c_rel[0][1], coarse_obj_token),
									c_rel[1], (c_rel[2][0], c_rel[2][1], lbl_idx))
								if DEBUG:
									print(f"POB:		Instance found (coarse)!")
									print(f"POB:		coarse_obj_token: {coarse_obj_token}")
									print(f"POB:		{c_rel_new_1[0][0]}; {c_rel_new_1[0][1]}; {c_rel_new_1[0][2]}")
									print(f"POB:		{c_rel_new_2[0][0]}; {c_rel_new_2[0][1]}; {c_rel_new_2[0][2]}")
									print(f"POB:		{r['word']}")
								a_c.append(c_rel_new_1)
								a_c.append(c_rel_new_2)
								# c_residue_idxs.append(c_rel_idx)
								amendment_counts['pob'] += 2

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# [{'word': ['张三', '在', '李四家', '玩', '。'], 'head': [4, 3, 4, 0, 4], 'deprel': ['SBV', 'MT', 'ADV', 'HED', 'MT']}]
		# [(([0, '张三'], [3, '玩'], None), 'SVO'), (([2, '李四家'], [3, '玩']), 'ADV_V')]
		# this is a quirky account in DDParser, it puts prepositions on lower position in the dependency tree than its object.
		if MT_flag is True:
			for lbl_idx, lbl in enumerate(r['deprel']):
				prep_token = r['word'][lbl_idx]
				if lbl == 'MT' and prep_token in pivot_adv_list:
					lbl_head_from_0 = r['head'][lbl_idx] - 1
					lbl_head_token = r['word'][lbl_head_from_0]
					lbl_grandparent_from_0 = r['head'][lbl_head_from_0] - 1
					lbl_grandparent_token = r['word'][lbl_grandparent_from_0]

					for f_rel_idx, f_rel in enumerate(j_f):
						if f_rel[1] != 'SVO':
							continue
						if lbl_grandparent_from_0 == f_rel[2][1]:
							if lbl_grandparent_token != f_rel[0][1]:
								if DEBUG:
									print(f"MT mismatch! {lbl_grandparent_token}; {f_rel[0][1]}")
							MT_matching_flag = False
							for f_rel_idx_ref, f_rel_ref in enumerate(j_f):
								if f_rel_ref[1] == 'ADV_V' and f_rel_ref[0][1] == lbl_grandparent_token and \
										lbl_head_token in f_rel_ref[0][0]:
									POB_matching_flag = True
							if MT_matching_flag is False and Discard_POB_where_ADV_rel_not_found_flag:
								continue

							if '·X·' in f_rel[0][1]:
								assert f_rel[0][1].count('·X·') == 1
								if DEBUG:
									print("MT meeting predicates including ·X· !")
									print('f_rel: ', f_rel)
									print('ddp result: ', r)
								continue
							f_rel_new = ((f_rel[0][0], prep_token + '·X·' + f_rel[0][1], lbl_head_token), f_rel[1],
										 (f_rel[2][0], f_rel[2][1], lbl_head_from_0))
							a_f.append(f_rel_new)
							if f_rel[0][2] is not None and f_rel[2][2] != lbl_head_from_0:
								f_rel_new_2 = ((f_rel[0][0], prep_token+'·X·'+f_rel[0][1]+'·'+f_rel[0][2], lbl_head_token),
										   		f_rel[1], (f_rel[2][0], f_rel[2][1], lbl_head_from_0))
								a_f.append(f_rel_new_2)
							# f_residue_idxs.append(f_rel_idx)
							amendment_counts['mt'] += 1
							if Prep_nary_flag is True:
								f_rel_new_3 = (
									(lbl_head_token, prep_token + '·X·' + f_rel[0][1] + '·【介宾】', f_rel[0][2]), f_rel[1],
									(lbl_head_from_0, f_rel[2][1], f_rel[2][2]))
								a_f.append(f_rel_new_3)
								amendment_counts['mt'] += 1

					for c_rel_idx, c_rel in enumerate(j_c):
						if c_rel[1] != 'SVO':
							continue
						if lbl_grandparent_from_0 == c_rel[2][1]:
							if lbl_grandparent_token != c_rel[0][1]:
								if DEBUG:
									print(f"MT mismatch! {lbl_grandparent_token}; {c_rel[0][1]}")
							coarse_obj_token = None
							for c_rel_idx_ref, c_rel_ref in enumerate(j_c):
								if c_rel_ref[1] == 'ADV_V' and c_rel_ref[0][1] == lbl_grandparent_token and \
										lbl_head_token in c_rel_ref[0][0]:
									coarse_obj_token = c_rel_ref[0][0]
							if coarse_obj_token is None:
								if Discard_POB_where_ADV_rel_not_found_flag:
									continue
								else:
									coarse_obj_token = lbl_head_token
							if '·X·' in c_rel[0][1]:
								assert c_rel[0][1].count('·X·') == 1
								if DEBUG:
									print("MT meeting predicates including ·X· !")
									print('c_rel: ', c_rel)
									print('ddp result: ', r)
								continue
							c_rel_new = ((c_rel[0][0], prep_token + '·X·' + c_rel[0][1], coarse_obj_token), c_rel[1],
										 (c_rel[2][0], c_rel[2][1], lbl_head_from_0))
							a_c.append(c_rel_new)
							# c_residue_idxs.append(c_rel_idx)
							amendment_counts['mt'] += 1
							if Prep_nary_flag is True:
								c_rel_new_2 = (
									(coarse_obj_token, prep_token + '·X·' + c_rel[0][1] + '·【介宾】', c_rel[0][2]),
									c_rel[1],
									(lbl_head_from_0, c_rel[2][1], c_rel[2][2]))
								a_c.append(c_rel_new_2)
								amendment_counts['mt'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# VV (or other cases where (subj, pred, None), (None, pred, obj))
		# Dealing with conjuncted verbs. In seen examples, when faced with verb conjunction, they are failing to link
		# the object with the VV-inspired (subj, V) structure, and failing to link the subject to the local (V, obj)
		# structure. So if there exists a verb heading two SVO structures, one without subject, the other without object,
		# and the verb is itself bearing a VV dependency label (projected most probably from the verb bearing HEAD label),
		# then we consider the two relations as should have been merged into one.
		#
		# should only check the indices, not the surface names, to include cases such as "(经理, 提供, None)" "(None, 为·提供·帮助, 旅客)"
		# case for VV: 我 去 法国 旅游; between 去 and 旅游; 我 去 诊所 打 疫苗 (a covert 'to')
		# case for COO: 他们 奔跑 、 跳跃 在 广阔的 马赛马拉大草原 (a coordination)
		if VV_flag is True:
			# f_rel_1 is to the left of f_rel_2 <- deprecated: no order needs to be specified, the order of who provides the subject is already there.
			j_f_svos_idxs, j_f_svos = filter_rel_list(j_f, 'SVO')
			j_c_svos_idxs, j_c_svos = filter_rel_list(j_c, 'SVO')

			for f_rel_idx_1, f_rel_1 in zip(j_f_svos_idxs, j_f_svos):
				if f_rel_1[1] != 'SVO':
					raise AssertionError
				elif r['deprel'][f_rel_1[2][1]] not in ['VV', 'COO']:
					continue
				elif r['deprel'][f_rel_1[2][1]] == 'COO':
					hed_vb = r['head'][f_rel_1[2][1]] - 1
					if r['deprel'][hed_vb] not in ['HED', 'IC']:
						continue
				amend_type = r['deprel'][f_rel_1[2][1]]
				for f_rel_idx_2, f_rel_2 in zip(j_f_svos_idxs, j_f_svos):
					if f_rel_2[1] != 'SVO' or f_rel_idx_1 == f_rel_idx_2:
						continue
					bei_in_pred_children = find_bei_from_children(r, f_rel_2[2][1])
					# if the two 'SVO' relations have the same 'VV' labelled predicate (word and index), and f_rel_1's object
					# is None and f_rel_2's subject is None:
					if f_rel_1[2][1] == f_rel_2[2][1]:
						# decide the predicate name when conflict.
						if f_rel_1[0][1] != f_rel_2[0][1]:
							if DEBUG:
								print(f"VV Diff tokens! {f_rel_1[0][1]}; {f_rel_2[0][1]}")
							pred_name = None
							if '·' in f_rel_1[0][1]:
								pred_name = f_rel_1[0][1]
							elif '·' in f_rel_2[0][1]:
								pred_name = f_rel_2[0][1]
							else:
								print("Impossible situation @ VV fine!", file=sys.stderr)
								pred_name = f_rel_2[0][1]
						else:
							pred_name = f_rel_1[0][1]
						#if f_rel_1[0][2] is None and f_rel_2[0][0] is None:
						if f_rel_2[0][0] is None and f_rel_1[0][0] is not None:
							f_rel_new = ((f_rel_1[0][0], pred_name, f_rel_2[0][2]), 'SVO',
										 (f_rel_1[2][0], f_rel_1[2][1], f_rel_2[2][2]))
							if DEBUG:
								print("VV:		Instance found (fine)!")
								print(f"VV:		{f_rel_new[0][0]}; {f_rel_new[0][1]}; {f_rel_new[0][2]}")
								print(f"VV:		{r['word']}")
							a_f.append(f_rel_new)
							# f_residue_idxs.append(f_rel_idx_1)
							f_residue_idxs.add(f_rel_idx_2)
							amendment_counts['vv'] += 1
						elif f_rel_2[0][2] is None and bei_in_pred_children:  # passives
							f_rel_new = ((f_rel_2[0][0], pred_name, f_rel_1[0][0]), 'SVO',
										 (f_rel_2[2][0], f_rel_1[2][1], f_rel_1[2][0]))
							if DEBUG:
								print("VV:		Instance found (fine)!")
								print(f"VV:		{f_rel_new[0][0]}; {f_rel_new[0][1]}; {f_rel_new[0][2]}")
								print(f"VV:		{r['word']}")
							a_f.append(f_rel_new)
							# f_residue_idxs.append(f_rel_idx_1)
							f_residue_idxs.add(f_rel_idx_2)
							amendment_counts['vv'] += 1

			# c_rel_1 is to the left of c_rel_2 <- deprecated: no order needs to be specified, the order of who provides the subject is already there.
			for c_rel_idx_1, c_rel_1 in zip(j_c_svos_idxs, j_c_svos):
				if c_rel_1[1] != 'SVO':
					raise AssertionError
				elif r['deprel'][c_rel_1[2][1]] not in ['VV', 'COO']:
					continue
				elif r['deprel'][c_rel_1[2][1]] == 'COO':
					hed_vb = r['head'][c_rel_1[2][1]] - 1
					if r['deprel'][hed_vb] not in ['HED', 'IC']:
						continue
				for c_rel_idx_2, c_rel_2 in zip(j_c_svos_idxs, j_c_svos):
					if c_rel_2[1] != 'SVO' or c_rel_idx_1 == c_rel_idx_2:
						continue
					bei_in_pred_children = find_bei_from_children(r, c_rel_2[2][1])
					# if the two 'SVO' relations have the same 'VV' labelled head (word and index), and c_rel_1's object
					# is None and c_rel_2's subject is None:
					if c_rel_1[2][1] == c_rel_2[2][1]:
						if c_rel_1[0][1] != c_rel_2[0][1]:
							if DEBUG:
								print(f"VV Diff tokens! {c_rel_1[0][1]}; {c_rel_2[0][1]}")
							if '·' in c_rel_1[0][1]:
								pred_name = c_rel_1[0][1]
							elif '·' in c_rel_2[0][1]:
								pred_name = c_rel_2[0][1]
							else:
								print("Impossible situation @ VV fine!", file=sys.stderr)
								pred_name = c_rel_2[0][1]
						else:
							pred_name = c_rel_2[0][1]
						#if c_rel_1[0][2] is None and c_rel_2[0][0] is None:
						if c_rel_2[0][0] is None and c_rel_1[0][0] is not None:
							c_rel_new = ((c_rel_1[0][0], pred_name, c_rel_2[0][2]), 'SVO',
										 (c_rel_1[2][0], c_rel_1[2][1], c_rel_2[2][2]))
							if DEBUG:
								print("VV:		 Instance found (coarse)!")
								print(f"VV:		{c_rel_new[0][0]}; {c_rel_new[0][1]}; {c_rel_new[0][2]}")
								print(f"VV:		{r['word']}")
							a_c.append(c_rel_new)
							# c_residue_idxs.append(c_rel_idx_1)
							c_residue_idxs.add(c_rel_idx_2)
							amendment_counts['vv'] += 1
						elif c_rel_2[0][2] is None and bei_in_pred_children:  # passives
							c_rel_new = ((c_rel_2[0][0], pred_name, c_rel_1[0][0]), 'SVO',
										 (c_rel_2[2][0], c_rel_1[2][1], c_rel_1[2][0]))
							if DEBUG:
								print("VV:		 Instance found (coarse)!")
								print(f"VV:		{c_rel_new[0][0]}; {c_rel_new[0][1]}; {c_rel_new[0][2]}")
								print(f"VV:		{r['word']}")
							a_c.append(c_rel_new)
							# c_residue_idxs.append(c_rel_idx_1)
							c_residue_idxs.add(c_rel_idx_2)
							amendment_counts['vv'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# V_CMP: ((subj, pred_1, None), 'SVO'), ((pred_1, pred_2), 'V_CMP'), ((None, pred_2, obj), 'SVO')
		# f_rel_1: ((None, to, library), 'SVO', (None, 3, 4))
		# f_rel_2: ((walk, to), 'V_CMP', (2, 3));
		# f_rel_3: ((I, walk, None), 'SVO', (1, 2, None));
		# Turned into: ((I, walk to, library), 'SVO', (1, 2, 4))
		# Combine Vs with particles to create new predicates: When a SVO head is itself a complement in V_CMP relation,
		# and its parent in the V_CMP relation is involved in an SVO with None object, and it itself is involved in an
		# SVO relation with None subject, then the V_CMP-related two words should be in one predicate, and the subject
		# and object should be merged into one relation.
		if VCMP_flag is True:
			j_f_svos_idxs, j_f_svos = filter_rel_list(j_f, 'SVO')
			j_c_svos_idxs, j_c_svos = filter_rel_list(j_c, 'SVO')

			j_f_vcmp_idxs, j_f_vcmps = filter_rel_list(j_f, 'V_CMP')
			j_c_vcmp_idxs, j_c_vcmps = filter_rel_list(j_c, 'V_CMP')

			for f_rel_idx_1, f_rel_1 in zip(j_f_svos_idxs, j_f_svos):
				if f_rel_1[1] != 'SVO':
					raise AssertionError
				if f_rel_1[0][0] is not None:  # f_rel_1 has to be SVO with empty subject
					continue
				for f_rel_idx_2, f_rel_2 in zip(j_f_vcmp_idxs, j_f_vcmps):
					assert f_rel_2[1] == 'V_CMP'
					# f_rel_2 must be 'V_CMP', and the predicate of f_rel_1 is the dependent of f_rel_2
					if f_rel_2[1] == 'V_CMP' and f_rel_2[0][1] == f_rel_1[0][1] and f_rel_2[2][1] == f_rel_1[2][1]:
						if f_rel_2[0][1] != f_rel_1[0][1]:
							if DEBUG:
								print(f"VCMP 1-2 mismatch! This should not happen! {f_rel_2[0][1]}; {f_rel_1[0][1]}",
									  file=sys.stderr)
						complement_token = f_rel_2[0][1]
						for f_rel_idx_3, f_rel_3 in zip(j_f_svos_idxs, j_f_svos):
							assert f_rel_3[1] == 'SVO'
							# f_rel_3 must be 'SVO', and the predicate of f_rel_3 is the head of the V_CMP in f_rel_2,
							# and f_rel_3 must have empty object.
							if f_rel_3[1] == 'SVO' and f_rel_3[2][1] == f_rel_2[2][0] and f_rel_3[0][2] is None:
								if f_rel_3[0][1] != f_rel_2[0][0]:
									if DEBUG:
										print(
											f"VCMP 3-2 mismatch! This should not have happened! {f_rel_3[0][1]} {f_rel_2[0][0]}",
											file=sys.stderr)
								f_rel_new = ((f_rel_3[0][0], f_rel_2[0][0] + '·' + f_rel_2[0][1], f_rel_1[0][2]), 'SVO',
											 (f_rel_3[2][0], f_rel_2[2][0], f_rel_1[2][2]))
								if DEBUG:
									print("V_CMP	Instance found (fine)!")
									print(f"V_CMP:		{f_rel_new[0][0]}; {f_rel_new[0][1]}; {f_rel_new[0][2]}")
									print(f"V_CMP:		{r['word']}")
								a_f.append(f_rel_new)
								f_residue_idxs.add(f_rel_idx_1)
								f_residue_idxs.add(f_rel_idx_3)
								amendment_counts['vcmp'] += 1
								if complement_token not in vcmp_dict:
									vcmp_dict[complement_token] = 1
								else:
									vcmp_dict[complement_token] += 1
								continue  # there could be multiple co-ordinated subjects
						break  # but there could be only one V_CMP structure for each CMP (since we're assuming this to be a dependecy tree, not a graph).

			for c_rel_idx_1, c_rel_1 in zip(j_c_svos_idxs, j_c_svos):
				if c_rel_1[1] != 'SVO':
					raise AssertionError
				if c_rel_1[0][0] is not None:  # c_rel_1 has to be SVO with empty subject
					continue
				for c_rel_idx_2, c_rel_2 in zip(j_c_vcmp_idxs, j_c_vcmps):
					assert c_rel_2[1] == 'V_CMP'
					# c_rel_2 must be 'V_CMP', and the predicate of c_rel_1 is the dependent of c_rel_2
					if c_rel_2[1] == 'V_CMP' and c_rel_2[2][1] == c_rel_1[2][1]:
						if c_rel_2[0][1] != c_rel_1[0][1]:
							if DEBUG:
								print(f"VMP 1-2 mismatch! This should not happen! {c_rel_2[0][1]}; {c_rel_1[0][1]}",
									  file=sys.stderr)
						complement_token = c_rel_2[0][1]
						for c_rel_idx_3, c_rel_3 in zip(j_c_svos_idxs, j_c_svos):
							# c_rel_3 must be 'SVO', and the predicate of c_rel_3 is the head of the V_CMP in c_rel_2,
							# and c_rel_3 must have empty object.
							assert c_rel_3[1] == 'SVO'
							if c_rel_3[1] == 'SVO' and c_rel_3[2][1] == c_rel_2[2][0] and c_rel_3[0][2] is None:
								if c_rel_3[0][1] != c_rel_2[0][0]:
									if DEBUG:
										print(
											f"VMP 3-2 mismatch! This should not happen! {c_rel_3[0][1]}; {c_rel_2[0][0]}",
											file=sys.stderr)
								c_rel_new = ((c_rel_3[0][0], c_rel_2[0][0] + '·' + c_rel_2[0][1], c_rel_1[0][2]), 'SVO',
											 (c_rel_3[2][0], c_rel_2[2][0], c_rel_1[2][2]))
								if DEBUG:
									print("V_CMP	Instance found (coarse)!")
									print(f"V_CMP:		{c_rel_new[0][0]}; {c_rel_new[0][1]}; {c_rel_new[0][2]}")
									print(f"V_CMP:		{r['word']}")
								a_c.append(c_rel_new)
								c_residue_idxs.add(c_rel_idx_1)
								c_residue_idxs.add(c_rel_idx_3)
								amendment_counts['vcmp'] += 1
								if complement_token not in vcmp_dict:
									vcmp_dict[complement_token] = 1
								else:
									vcmp_dict[complement_token] += 1
								continue  # there could be multiple co-ordinated subjects
						break  # but there could be only one V_CMP structure for each CMP (since we're assuming this to be a dependecy tree, not a graph).

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# ATT_OBJ: in many cases such as "咽炎(SBV) 成为(HED) 发热(ATT) 的(MT) 原因(VOB)", the ATT token affiliated to the
		# VOB might itself be an object to a composite predicate "成为·的·原因"
		# [{'word': ['咽炎', '成为', '发热', '的', '原因', '。'], 'head': [2, 0, 5, 3, 2, 2], 'deprel': ['SBV', 'HED', 'ATT', 'MT', 'VOB', 'MT']}]
		# [(([0, '咽炎'], [1, '成为'], [4, '原因']), 'SVO'), (([2, '发热'], [4, '原因']), 'ATT_N')]
		# [(([0, '咽炎'], [1, '成为'], [4, '发热的原因']), 'SVO'), (([2, '发热'], [4, '原因']), 'ATT_N')]
		# We DO NOT do this amendment for coarse-grained relations, because in coarse setting, the surface object contains the entire phrase "reason of fever".
		if ATT_OBJ_flag is True and (pos_tags is not None or ignore_postag):
			attobj_cnt = 0

			j_f_attn_idxs, j_f_attns = filter_rel_list(j_f, 'ATT_N')
			j_c_attn_idxs, j_c_attns = filter_rel_list(j_c, 'ATT_N')

			for f_rel_idx_1, f_rel_1 in enumerate(j_f):
				if f_rel_1[1] != 'SVO':
					continue
				for f_rel_idx_2, f_rel_2 in zip(j_f_attn_idxs, j_f_attns):
					if f_rel_2[1] != 'ATT_N':
						raise AssertionError
					ATT_word = f_rel_2[0][0]
					ATT_head = f_rel_2[0][1]
					if ATT_head in ['的']:  # set of stop words
						continue
					ATT_is_nominal = ignore_postag or examine_postags(ATT_word, p, acceptable_postag_list)
					if f_rel_1[2][2] is not None and f_rel_1[2][2] == f_rel_2[2][1] and ATT_is_nominal:
						if f_rel_1[0][2] != f_rel_2[0][1]:
							if DEBUG:
								print(f"ATT_OBJ surface object mismatch! {f_rel_1[0][2]}; {f_rel_2[0][1]}")
						if '·X·' in f_rel_1[0][1]:
							if f_rel_1[0][1].count('·X·') > 1:
								print(f_rel_1)
								print(r)
								raise AssertionError
							new_pred = f_rel_1[0][1].replace('·X·', f'·X·的·{f_rel_2[0][1]}·')
						else:
							new_pred = f_rel_1[0][1]+'·X·的·'+f_rel_2[0][1]

						if nightly:
							for tok_bei_idx in range(len(r['word'])):
								if r['word'][tok_bei_idx] == '被' and r['deprel'][tok_bei_idx] == 'POB' and r['head'][tok_bei_idx] == f_rel_2[2][0]+1:
									new_pred = new_pred.replace('·X·', '·被·X·')
									break

						f_rel_new = (
							(f_rel_1[0][0], new_pred, f_rel_2[0][0]), f_rel_1[1],
							(f_rel_1[2][0], f_rel_1[2][1], f_rel_2[2][0]))
						a_f.append(f_rel_new)
						# f_residue_idxs.append(f_rel_idx_1)
						amendment_counts['attobj'] += 1
			for c_rel_idx_1, c_rel_1 in enumerate(j_c):
				if c_rel_1[1] != 'SVO':
					continue
				for c_rel_idx_2, c_rel_2 in zip(j_c_attn_idxs, j_c_attns):
					if c_rel_2[1] != 'ATT_N':
						raise AssertionError
					if c_rel_2[0][0] == '不':
						continue
					ATT_word = c_rel_2[0][0]
					ATT_head = c_rel_2[0][1]
					if ATT_head in ['的']:  # set of stop words
						continue
					ATT_is_nominal = ignore_postag or examine_postags(ATT_word, p, acceptable_postag_list)
					if c_rel_1[0][2] == c_rel_2[0][1] and ATT_is_nominal:
						if '·X·' in c_rel_1[0][1]:
							if c_rel_1[0][1].count('·X·') > 1:
								print(c_rel_1)
								print(r)
								raise AssertionError
							new_pred = c_rel_1[0][1].replace('·X·', f'·X·的·{c_rel_2[0][1]}')
						else:
							new_pred = c_rel_1[0][1]+'·X·的·'+c_rel_2[0][1]
						c_rel_new = (
							(c_rel_1[0][0], new_pred, c_rel_2[0][0]), c_rel_1[1],
							(c_rel_1[2][0], c_rel_1[2][1], c_rel_2[2][0])
						)
						# if the new object is actually contained in the predicate, then don't add this entry
						if c_rel_new[0][2] in c_rel_new[0][1]:
							continue
						a_c.append(c_rel_new)
						amendment_counts['attobj'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# 张三发出关于那次事故的报道
		if ATT_POB_flag is True and (pos_tags is not None or ignore_postag):
			j_f_svos_idxs, j_f_svos = filter_rel_list(j_f, 'SVO')
			j_c_svos_idxs, j_c_svos = filter_rel_list(j_c, 'SVO')
			j_f_attn_idxs, j_f_attns = filter_rel_list(j_f, 'ATT_N')
			j_c_attn_idxs, j_c_attns = filter_rel_list(j_c, 'ATT_N')

			for f_rel_idx_1, f_rel_1 in zip(j_f_svos_idxs, j_f_svos):
				if f_rel_1[1] != 'SVO':
					continue
				for f_rel_idx_2, f_rel_2 in zip(j_f_attn_idxs, j_f_attns):
					if f_rel_2[1] != 'ATT_N':
						# continue
						raise AssertionError
					if f_rel_1[2][2] is not None and f_rel_1[2][2] == f_rel_2[2][1]:
						ATT_word = f_rel_2[0][0]
						ATT_idx_from_1 = f_rel_2[2][0] + 1

						for bei_idx in range(len(r['word'])):
							if r['deprel'][bei_idx] == 'POB' and r['head'][bei_idx] == ATT_idx_from_1 and \
								r['word'][bei_idx] == '被':
								ATT_word = '被' + ATT_word
								break

						att_pob_found = None
						for tok_idx in range(len(r['word'])):
							if r['deprel'][tok_idx] == 'POB' and r['head'][tok_idx] == ATT_idx_from_1 and \
									r['word'][tok_idx] != '被':
								obj_token = r['word'][tok_idx]
								#if '被' in obj_token:
								#	print(r)
								#	print(obj_token)
								if '·X·' in f_rel_1[0][1]:
									if f_rel_1[0][1].count('·X·') > 1:
										print(f_rel_1)
										print(r)
										raise AssertionError
									new_pred = f_rel_1[0][1].replace('·X·', f'·{ATT_word}·X·的·{f_rel_1[0][2]}·')
								else:
									new_pred = f_rel_1[0][1]+f'·{ATT_word}·X·的·{f_rel_1[0][2]}'
								f_rel_new = (
									(f_rel_1[0][0], new_pred, obj_token), f_rel_1[1],
									(f_rel_1[2][0], f_rel_1[2][1], tok_idx))
								if att_pob_found is not None:
									if DEBUG:
										print("DOUBLE ATT_POB FOUND (FINE)!")
										print(f"new rel 1: {att_pob_found}")
										print(f"new rel 2: {f_rel_new}")
										print(f"ddp result: {r}")
								# print(f_rel_new)
								a_f.append(f_rel_new)
								amendment_counts['attpob'] += 1
								att_pob_found = f_rel_new

			for c_rel_idx_1, c_rel_1 in zip(j_c_svos_idxs, j_c_svos):
				if c_rel_1[1] != 'SVO':
					continue
				for c_rel_idx_2, c_rel_2 in zip(j_c_attn_idxs, j_c_attns):
					if c_rel_2[1] != 'ATT_N':
						# continue
						raise AssertionError
					if c_rel_1[2][2] is not None and c_rel_1[2][2] == c_rel_2[2][1]:
						ATT_word = c_rel_2[0][0]
						ATT_idx_from_1 = c_rel_2[2][0] + 1
						if c_rel_1[0][2] != c_rel_2[0][1]:
							if DEBUG:
								print(f"ATT_POB surface object mismatch! {c_rel_1[0][2]}; {c_rel_2[0][1]}")
						att_pob_found = None
						for tok_idx in range(len(r['word'])):
							if r['deprel'][tok_idx] == 'POB' and r['head'][tok_idx] == ATT_idx_from_1 and \
								r['word'][tok_idx] not in ['被']:
								obj_token = r['word'][tok_idx]  # supposedly will be coarsified later.
								if '·X·' in c_rel_1[0][1]:
									if c_rel_1[0][1].count('·X·') > 1:
										print(c_rel_1)
										print(r)
										raise AssertionError
									new_pred = c_rel_1[0][1].replace('·X·', f'·{ATT_word}·X·的·{c_rel_1[0][2]}·')
								else:
									new_pred = c_rel_1[0][1]+f'·{ATT_word}·X·的·{c_rel_1[0][2]}'
								c_rel_new = (
									(c_rel_1[0][0], new_pred, obj_token), c_rel_1[1],
									(c_rel_1[2][0], c_rel_1[2][1], tok_idx))
								# if the new object is actually contained in the predicate, then don't add this entry
								if c_rel_new[0][2] in c_rel_new[0][1]:
									continue
								if att_pob_found is not None:
									print("DOUBLE ATT_POB FOUND (COARSE)!")
									print(f"new rel 1: {att_pob_found}")
									print(f"new rel 2: {c_rel_new}")
									print(f"ddp result: {r}")
								print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
								a_c.append(c_rel_new)
								amendment_counts['attpob'] += 1
								att_pob_found = c_rel_new

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# deals with copular subjects (copular objects have been dealt with in ATT_OBJ amendment)
		# 北京大学的校长是郝平。 -> (校长, 是, 郝平) -> (北京大学, 校长·是, 郝平)
		if COP_SBJ_flag is True and (pos_tags is not None or ignore_postag):
			j_f_attn_idxs, j_f_attns = filter_rel_list(j_f, 'ATT_N')
			j_c_attn_idxs, j_c_attns = filter_rel_list(j_c, 'ATT_N')

			for f_rel_idx_1, f_rel_1 in enumerate(j_f):
				if f_rel_1[1] != 'SVO':
					continue
				pred_head_word = r['word'][f_rel_1[2][1]]
				if pred_head_word not in cop_pred_set:
					continue
				if pred_head_word == '属于':
					print(f_rel_1)
				for f_rel_idx_2, f_rel_2 in zip(j_f_attn_idxs, j_f_attns):
					if f_rel_2[1] != 'ATT_N':
						# continue
						raise AssertionError
					ATT_word = f_rel_2[0][0]
					ATT_is_nominal = ignore_postag or examine_postags(ATT_word, p, noun_postag_list)
					if f_rel_1[2][0] is not None and f_rel_1[2][0] == f_rel_2[2][1] and ATT_is_nominal:
						if f_rel_1[0][0] != f_rel_2[0][1]:
							if DEBUG:
								print(f"COP_SBJ surface object mismatch! {f_rel_1[0][0]}; {f_rel_2[0][1]}")
						if '·X·' in f_rel_1:
							print('COP_SBJ predicate with ·X· !')
							print("f_rel_1: ", f_rel_1)
							print("ddp result: ", r)
							raise AssertionError
						f_rel_new = (
							(f_rel_2[0][0], f_rel_1[0][0]+'·'+f_rel_1[0][1], f_rel_1[0][2]), f_rel_1[1],
							(f_rel_2[2][0], f_rel_1[2][1], f_rel_1[2][2]))
						a_f.append(f_rel_new)
						# f_residue_idxs.append(f_rel_idx_1)
						amendment_counts['copsbj'] += 1
			for c_rel_idx_1, c_rel_1 in enumerate(j_c):
				if c_rel_1[1] != 'SVO':
					continue
				pred_head_word = r['word'][c_rel_1[2][1]]
				if pred_head_word not in cop_pred_set:
					continue
				for c_rel_idx_2, c_rel_2 in zip(j_c_attn_idxs, j_c_attns):
					if c_rel_2[1] != 'ATT_N':
						# continue
						raise AssertionError
					ATT_word = c_rel_2[0][0]
					ATT_is_nominal = ignore_postag or examine_postags(ATT_word, p, noun_postag_list)
					if c_rel_1[2][0] is not None and c_rel_1[2][0] == c_rel_2[2][1] and ATT_is_nominal and c_rel_1[0][0] == r['word'][c_rel_1[2][0]]:
						if c_rel_1[0][0] != c_rel_2[0][1]:
							if DEBUG:
								print(f"COPSBJ surface object mismatch! {c_rel_1[0][0]}; {c_rel_2[0][1]}")
						if '·X·' in c_rel_1:
							print('COP_SBJ predicate with ·X· !')
							print("c_rel_1: ", c_rel_1)
							print("ddp result: ", r)
							raise AssertionError
						c_rel_new = (
							(c_rel_2[0][0], c_rel_1[0][0] + '·' + c_rel_1[0][1], c_rel_1[0][2]), c_rel_1[1],
							(c_rel_2[2][0], c_rel_1[2][1], c_rel_1[2][2]))
						a_c.append(c_rel_new)
						# c_residue_idxs.append(c_rel_idx_1)
						amendment_counts['copsbj'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# MT variant: [玉米 是 从 美国 引进 的 。] [SBV HED MT ADV ATT VOB] [2, 0, 4, 5, 6, 2]
		# Bare ADV variant: [设备 是 木头 做 的 。] [SBV, HED, ADV, ATT, VOB] [2, 0, 4, 5, 2] (only when ADV is a noun.)
		# ADV POB variant: [设备 是 用 木头 做 的 。] [SBV, HED, ADV, POB, ATT, VOB] [2, 0, 5, 3, 6, 2]
		# SBV variant: [语言 是 埃及人 说 的 。] [SBV, HED, SBV, ATT, VOB] [2, 0, 4, 5, 2]
		if COP_ADJ_flag is True:
			for f_rel_idx, f_rel in enumerate(j_f):
				if f_rel[1] != 'SVO' or f_rel[0][2] != '的':
					continue
				pred_head_word = r['word'][f_rel[2][1]]
				if pred_head_word not in cop_pred_set:
					continue
				elif '·X·' in f_rel[0][1]:
					continue
				de_idx_from_1 = f_rel[2][2] + 1
				de_in_vobvob_flag = False  # whether the De token actually hosts another object after it.
				for vobvob_idx in range(len(r['word'])):
					if r['head'][vobvob_idx] == de_idx_from_1 and r['deprel'][vobvob_idx] == 'VOB':
						de_in_vobvob_flag = True
						break
				if de_in_vobvob_flag:
					continue
				for att_idx in range(len(r['word'])):
					if r['head'][att_idx] == de_idx_from_1 and r['deprel'][att_idx] == 'ATT':
						att_tok = r['word'][att_idx]
						att_idx_from_1 = att_idx + 1
						for bei_idx in range(len(r['word'])):
							if r['head'][bei_idx] == att_idx_from_1 and r['deprel'][bei_idx] == 'POB' and \
									r['word'][bei_idx] == '被':
								att_tok = '被' + att_tok
								break
						for cmp_idx in range(len(r['word'])):
							if r['head'][cmp_idx] == att_idx_from_1 and r['deprel'][cmp_idx] == 'CMP' \
									and cmp_idx+1 > att_idx_from_1:
								att_tok = att_tok + r['word'][cmp_idx]

						for adv_idx in range(len(r['word'])):
							if r['head'][adv_idx] == att_idx_from_1 and r['deprel'][adv_idx] == 'ADV':
								adv_tok = r['word'][adv_idx]
								adv_idx_from_1 = adv_idx + 1
								adv_child_found = False
								for prep_idx in range(len(r['word'])):
									if r['head'][prep_idx] == adv_idx_from_1 and r['deprel'][prep_idx] == 'MT':
										mt_token = r['word'][prep_idx]
										if mt_token in punctuation_list:
											continue
										new_pred = f_rel[0][1] + '·' + mt_token + '·X·' + att_tok + '·的'
										f_rel_new = ((f_rel[0][0], new_pred, adv_tok), f_rel[1],
													 (f_rel[2][0], f_rel[2][1], adv_idx))
										if not (prep_idx+1 < adv_idx_from_1 < att_idx_from_1 < de_idx_from_1):
											if DEBUG:
												print("UNEXPECTED BEHAVIOR! NOT FOLLOWING MT ADV ATT OBV ORDER!")
												print(r)
												print(f_rel)
												print(f_rel_new)
												print("--------------------------------------")
												print("")
											continue
										a_f.append(f_rel_new)
										f_residue_idxs.add(f_rel_idx)
										amendment_counts['copadj'] += 1
										adv_child_found = True
									elif r['head'][prep_idx] == adv_idx_from_1 and r['deprel'][prep_idx] == 'POB':
										pob_tok = r['word'][prep_idx]
										if pob_tok == '被':
											pass
										else:
											new_pred = f_rel[0][1] + '·' + adv_tok + '·X·' + att_tok + '·的'
											f_rel_new = ((f_rel[0][0], new_pred, pob_tok), f_rel[1],
														 (f_rel[2][0], f_rel[2][1], prep_idx))
											if not (adv_idx_from_1 < prep_idx+1 < att_idx_from_1 < de_idx_from_1):
												if DEBUG:
													print("UNEXPECTED BEHAVIOR! NOT FOLLOWING ADV POB ATT OBV ORDER!")
													print(r)
													print(f_rel)
													print(f_rel_new)
													print("--------------------------------------")
													print("")
												continue
											a_f.append(f_rel_new)
											f_residue_idxs.add(f_rel_idx)
											amendment_counts['copadj'] += 1
											adv_child_found = True

								if not adv_child_found and examine_postags(adv_tok, p, noun_postag_list):
									new_pred = f_rel[0][1] + '·X·' + att_tok + '·的'
									f_rel_new = ((f_rel[0][0], new_pred, adv_tok), f_rel[1],
												 (f_rel[2][0], f_rel[2][1], adv_idx))
									if not (adv_idx_from_1 < att_idx_from_1 < de_idx_from_1):
										if DEBUG:
											print("UNEXPECTED BEHAVIOR! NOT FOLLOWING ADV ATT OBV ORDER!")
											print(r)
											print(f_rel)
											print(f_rel_new)
											print("--------------------------------------")
											print("")
										continue
									a_f.append(f_rel_new)
									f_residue_idxs.add(f_rel_idx)
									amendment_counts['copadj'] += 1

							elif r['head'][adv_idx] == att_idx_from_1 and r['deprel'][adv_idx] == 'SBV':
								sbv_tok = r['word'][adv_idx]

								new_pred = f_rel[0][1] + '·X·' + att_tok + '·的'
								f_rel_new = ((f_rel[0][0], new_pred, sbv_tok), f_rel[1],
											 (f_rel[2][0], f_rel[2][1], adv_idx))
								if att_idx_from_1 > de_idx_from_1:
									if DEBUG:
										print("UNEXPECTED BEHAVIOR! ATT IDX > DE IDX!")
										print(r)
										print(f_rel)
										print(f_rel_new)
										print("--------------------------------------")
										print("")
									continue
								a_f.append(f_rel_new)
								f_residue_idxs.add(f_rel_idx)
								amendment_counts['copadj'] += 1

			for c_rel_idx, c_rel in enumerate(j_c):
				pass

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# ATT_POB_OBJ: in cases such as "他们(SBV) 成为(HED) 建设(ATT) 家乡(VOB) 的(MT) 力量(VOB)"
		# [{'word': ['他们', '成为', '建设', '家园', '的', '力量', '。'], 'head': [2, 0, 6, 3, 3, 2, 2], 'deprel': ['SBV', 'HED', 'ATT', 'VOB', 'MT', 'VOB', 'MT']}]
		# [(([0, '他们'], [1, '成为'], [5, '力量']), 'SVO'), ((None, [2, '建设'], [3, '家园']), 'SVO'), (([2, '建设'], [5, '力量']), 'ATT_N')]
		# [(([0, '他们'], [1, '成为'], [5, '建设家园的力量']), 'SVO'), ((None, [2, '建设'], [3, '家园']), 'SVO'), (([2, '建设家园'], [5, '力量']), 'ATT_N')]
		# Aim: extract (他们, 成为, 力量), (他们, 成为·力量, 建设), (力量, 建设, 家乡)
		# Careful! Mind the cases such as: "他们(SBV) 成为(HED) 建设(ATT) 家乡(VOB) 的(MT) 动力(VOB)", where "他们" perhaps means "the workers' wives and kids".
		# we don't want (动力, 建设, 家乡)
		# For similar reasons as above, we do not to this amendment for coarse-grained relations as well.
		if ATT_SUB_flag is True:
			j_f_attn_idxs, j_f_attns = filter_rel_list(j_f, 'ATT_N')

			for f_rel_idx_1, f_rel_1 in enumerate(j_f):
				if f_rel_1[1] != 'SVO':
					continue
				if f_rel_1[0][0] is not None and f_rel_1[0][
					2] is not None:  # if the sub-relation has no empty slot, then there's no need.
					continue
				arg_seq_names = [f_rel_1[0][1]]  # contains the predicate
				arg_seq_idxs = [f_rel_1[2][1]]  # contains the predicate
				hit_N_flag = False  # to cope with situations with "领导小组成立以来会议内容"
				while not hit_N_flag:
					hit_N_flag = True
					for f_rel_idx_next, f_rel_next in zip(j_f_attn_idxs, j_f_attns):
						if f_rel_next[1] != 'ATT_N':
							# continue
							raise AssertionError
						if f_rel_next[2][0] is not None and f_rel_next[2][0] == arg_seq_idxs[-1]:
							if f_rel_next[0][0] != arg_seq_names[-1]:
								if DEBUG:
									print(
										f"ATT_SUB sub-relation predicate mismatch! {f_rel_next[0][0]}; {arg_seq_names[-1]}")  # "他们 成为 在 和平时期 建设 家乡 的 力量。"
							arg_seq_names.append(f_rel_next[0][1])
							arg_seq_idxs.append(f_rel_next[2][1])
							hit_N_flag = False

				assert len(arg_seq_names) == len(arg_seq_idxs)
				for i in range(1, len(arg_seq_names)):
					pred_name = '·'.join(arg_seq_names[:i])
					pred_idx = arg_seq_idxs[0]
					arg_name = arg_seq_names[i]
					arg_idx = arg_seq_idxs[i]
					if f_rel_1[0][0] is None:  # if the sub-relation has an empty subject: 他们成为被媒体忽视的力量。
						f_rel_new = (
							(arg_name, pred_name, f_rel_1[0][2]), f_rel_1[1], (arg_idx, pred_idx, f_rel_1[2][2]))
					elif f_rel_1[0][2] is None:
						f_rel_new = (
							(f_rel_1[0][0], pred_name, arg_name), f_rel_1[1], (f_rel_1[2][0], pred_idx, arg_idx))
					else:
						raise AssertionError
					a_f.append(f_rel_new)
					# f_residue_idxs.append(f_rel_idx_1)
					amendment_counts['attsub'] += 1

			for c_rel_idx, c_rel in enumerate(j_c):
				pass

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# all three tokens in this modification construction should be nominal in nature
		if ATT_ATT_flag is True:  # 德国总理默克尔 -> (默克尔，是·X·的总理，德国)
			for tok_1_idx in range(len(r['word'])):
				if r['deprel'][tok_1_idx] == 'ATT':
					tok_1_name = r['word'][tok_1_idx]
					tok_1_is_nominal = ignore_postag or examine_postags(tok_1_name, p, modif_noun_list)
					tok_2_idx = r['head'][tok_1_idx] - 1  # starting from 0
					if tok_2_idx - tok_1_idx != 1 or not tok_1_is_nominal:  # only take those contiguous strings of (ATT ATT N)
						continue
					if r['deprel'][tok_2_idx] == 'ATT':
						tok_2_name = r['word'][tok_2_idx]
						tok_2_is_nominal = ignore_postag or examine_postags(tok_2_name, p, modif_noun_list)
						# only when the second ATT is a noun! Otherwise there'd be cases like "手续办理时效" and "旅客购票速度"
						if not tok_2_is_nominal:
							continue
						tok_3_idx = r['head'][tok_2_idx] - 1  # starting from 0
						if tok_3_idx - tok_2_idx != 1:  # only take those contiguous strings of (ATT ATT N)
							continue
						if r['deprel'][tok_3_idx] in ['SBV', 'VOB', 'POB', 'HED', 'COO', 'IC', 'DOB']:
							tok_3_name = r['word'][tok_3_idx]
							tok_3_is_nominal = ignore_postag or examine_postags(tok_3_name, p, modif_noun_list)
							if not tok_3_is_nominal:
								continue
							f_rel_new = ((tok_3_name, '是·X·的·' + tok_2_name, tok_1_name), 'SVO',
								(tok_3_idx, tok_2_idx, tok_1_idx))
							if len(tok_3_name) == 1 or no_chinese_char(tok_3_name) or tok_3_name == '自己':  # heuristic filter
								if DEBUG:
									print(f"Modification construction filtered by heuristics: {f_rel_new}")
								continue
							elif f_rel_new[2][0] is not None and f_rel_new[2][1] is not None and abs(f_rel_new[2][0]-f_rel_new[2][1]) > 1:
								print(f"Modification construction filtered by distance: {f_rel_new}")
								continue
							possible_f.append(f_rel_new)
							amendment_counts['attatt'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		# for cases like "他被控试图杀死李四。"
		if VOB_VOB_flag is True:
			for f_rel_idx, f_rel in enumerate(j_f):
				if f_rel[1] != 'SVO' or f_rel[0][2] is None or f_rel[2][2] is None:
					continue
				vob_chain = [[f_rel[0][2]], [f_rel[2][2]]]
				hit_N_flag = False
				while not hit_N_flag:
					hit_N_flag = True
					obj_idx_from_1 = vob_chain[1][-1] + 1
					for tok_idx in range(len(r['word'])):
						if r['head'][tok_idx] == obj_idx_from_1 and r['deprel'][tok_idx] == 'VOB':
							vob_chain[0].append(r['word'][tok_idx])
							vob_chain[1].append(tok_idx)
							hit_N_flag = False
				assert len(vob_chain[0]) == len(vob_chain[1])
				for nidx in range(1, len(vob_chain[0])):
					if '·X·' in f_rel[0][1]:
						assert f_rel[0][1].count('·X·') == 1
						new_pred = f_rel[0][1].replace('·X·', '·'+'·'.join(vob_chain[0][:nidx])+'·X·')
					else:
						new_pred = f_rel[0][1]+'·'+'·'.join(vob_chain[0][:nidx])
					f_rel_new = ((f_rel[0][0], new_pred, vob_chain[0][nidx]), f_rel[1], (f_rel[2][0], f_rel[2][1], vob_chain[1][nidx]))
					a_f.append(f_rel_new)
					# f_residue_idxs.append(f_rel_idx)
					amendment_counts['vobvob'] += 1

		if progressive_flag:
			j_f = merge_orig_amend(f, a_f, f_residue_idxs, discard_residue)
			j_c = merge_orig_amend(c, a_c, c_residue_idxs, discard_residue)

		amend_fine_rels.append(a_f)
		amend_coarse_rels.append(a_c)
		possible_fine_rels.append(possible_f)
		p_f = []  # pruned_f
		p_c = []  # pruned_c

		# [f/c]_residue_idxs may contain out-of-range idxs for f/c, they just won't be matched whatsoever.
		for f_rel_idx, f_rel in enumerate(f):
			if (f_rel_idx not in f_residue_idxs) or (not discard_residue):
				p_f.append(f_rel)
		for c_rel_idx, c_rel in enumerate(c):
			if (c_rel_idx not in c_residue_idxs) or (not discard_residue):
				p_c.append(c_rel)
		if not discard_residue:
			assert len(c) == len(p_c)
			assert len(f) == len(p_f)
		pruned_fine_rels.append(p_f)
		pruned_coarse_rels.append(p_c)

	return amend_fine_rels, amend_coarse_rels, pruned_fine_rels, pruned_coarse_rels, possible_fine_rels, vcmp_dict, amendment_counts


def check_rel_pred_tense(rel, sent_rels, sent_ddp):
	past_mts = ['了', '过']
	past_advs = ['曾经', '曾', '从前', '以前', '昨天', '上周', '近日', '去年']
	future_advs = ['将', '将要', '将会', '必将', '明天', '下周', '明年', '后天']
	pred_name = rel[0][1]
	pred_head_idx_from_1 = rel[2][1] + 1
	past_flag = False
	future_flag = False
	for tok_id in range(len(sent_ddp['word'])):
		if sent_ddp['head'][tok_id] == pred_head_idx_from_1:
			if sent_ddp['deprel'][tok_id] == 'ADV':
				if sent_ddp['word'][tok_id] in future_advs:
					future_flag = True
				if sent_ddp['word'][tok_id] in past_advs:
					past_flag = True
			elif sent_ddp['deprel'][tok_id] == 'MT':
				if sent_ddp['word'][tok_id] in past_mts:
					past_flag = True
	if future_flag and past_flag:
		print("Both past tense and future tense are detected for the same predicate!")
		print("Rel: ", rel)
		print("Sentence: ", ''.join(sent_ddp['word']))
	if future_flag:
		return 'future'
	elif past_flag:
		return 'past'
	else:
		return 'present'


def check_rel_pred_modal(rel, sent_rels, sent_ddp):
	# basic value set: 能、该、会、要、敢
	modal_list = {'能': '能', '能够': '能', '该': '该', '应该': '该', '会': '会', '要': '要', '需要': '要', '敢': '敢',
				  '敢于': '敢', '可以': '能', '必须': '要', '肯': '会'}

	pred_head_idx_from_1 = rel[2][1] + 1
	pred_modals = []
	for tok_id in range(len(sent_ddp['word'])):
		if sent_ddp['head'][tok_id] == pred_head_idx_from_1:
			if sent_ddp['deprel'][tok_id] == 'ADV' and sent_ddp['word'][tok_id] in modal_list:
				modal_value = modal_list[sent_ddp['word'][tok_id]]
				if modal_value not in pred_modals:
					pred_modals.append(modal_value)
	if len(pred_modals) == 0:
		return None
	else:
		return ''.join([f'【{m}】' for m in pred_modals])


def trace_antecedent(desc_idx, ante_idx, sent_ddp):
	ante_idx_from_1 = ante_idx + 1
	child_idx = desc_idx
	parent_idx_from_1 = desc_idx + 1
	while parent_idx_from_1 != 0:
		if ante_idx_from_1 == parent_idx_from_1:
			return True
		parent_idx_from_1 = sent_ddp['head'][child_idx]
		child_idx = parent_idx_from_1 - 1
	return False


def check_nct(cur_rels, ddp_res, DEBUG=False):
	CONVERT_COPULAR = False
	CONVERT_TENSE = False
	CONVERT_MODAL = False
	neg_adv_list = ['不', '未能', '不会', '不能', '没有', '无法', '难以',]
	neg_cmp_list = ['失败']
	cop_pred_set = {'是', '也是', '就是', '而是', '正是', '才是', '都是', '仍是', '既是', '又是', '却是', '只是', '算是', '竟是',
					'便是', '无疑是', '乃是', '并且是'}
	new_rels = []
	for (sent_rels, sent_ddp) in zip(cur_rels, ddp_res):
		new_sent_rels = []
		for rel in sent_rels:
			if rel[1] != 'SVO':
				new_sent_rels.append(rel)
				continue
			# the type of negation included in the predicate is not covered here with '否·', since they're already
			# disambiguated from the positive versions of the same predicates.
			hed_idx_from_1 = rel[2][1] + 1
			vb_toks = rel[0][1].split('·')
			vb_idxs_from_1 = []
			for tok_id in range(len(sent_ddp['word'])):
				if trace_antecedent(tok_id, hed_idx_from_1-1, sent_ddp):
					if sent_ddp['word'][tok_id] in vb_toks and sent_ddp['word'][tok_id] != sent_ddp['word'][rel[2][1]]:
						vb_idxs_from_1.append(tok_id+1)
					elif tok_id == rel[2][1]:
						vb_idxs_from_1.append(tok_id + 1)
			neg_mark_count = 0
			for tok_id in range(len(sent_ddp['word'])):
				if sent_ddp['head'][tok_id] in vb_idxs_from_1:
					if (sent_ddp['deprel'][tok_id] == 'ADV' and sent_ddp['word'][tok_id] in neg_adv_list) or \
							(sent_ddp['deprel'][tok_id] == 'CMP' and sent_ddp['word'][tok_id] in neg_cmp_list):
						# ((f_rel_1[0][0], pred_name, arg_name), f_rel_1[1], (f_rel_1[2][0], pred_idx, arg_idx))
						neg_mark_count += 1
			if neg_mark_count > 2:
				if DEBUG:
					print(sent_ddp)
					print(vb_idxs_from_1)
					print(rel)
					print("neg_mark_count: ", neg_mark_count)
			if neg_mark_count % 2 == 1:
				new_sent_rels.append(((rel[0][0], '否·' + rel[0][1], rel[0][2]), rel[1], rel[2]))
			elif neg_mark_count % 2 == 0:
				new_sent_rels.append(rel)
			else:
				raise AssertionError

		new_rels.append(new_sent_rels)

	if CONVERT_COPULAR:
		negged_rels = copy.deepcopy(new_rels)
		new_rels = []
		for (sent_rels, sent_ddp) in zip(negged_rels, ddp_res):
			new_sent_rels = []
			for rel in sent_rels:
				if rel[1] != 'SVO':
					new_sent_rels.append(rel)
					continue
				# copular constructions.
				# pred_set = set(rel[0][1].split('·'))
				# the condition below means there are at least one piece of predicate that match one of the copular indicators.
				# if len(pred_set.intersection(cop_pred_set)) > 0:
				elif rel[0][1] in cop_pred_set:
					new_sent_rels.append(((rel[0][0], rel[0][1] + '·' + rel[0][2], None), rel[1], rel[2]))
					continue
				else:
					new_sent_rels.append(rel)
			new_rels.append(new_sent_rels)

	if CONVERT_TENSE:
		prev_rels = copy.deepcopy(new_rels)
		new_rels = []
		for (sent_rels, sent_ddp) in zip(prev_rels, ddp_res):
			new_sent_rels = []
			for rel in sent_rels:
				if rel[1] != 'SVO':
					new_sent_rels.append(rel)
				else:
					pred_tense = check_rel_pred_tense(rel, sent_rels, sent_ddp)
					if pred_tense == 'past':
						new_sent_rels.append(((rel[0][0], '【过去式】·'+rel[0][1], rel[0][2]), rel[1], rel[2]))
					elif pred_tense == 'future':
						new_sent_rels.append(((rel[0][0], '【将来式】·' + rel[0][1], rel[0][2]), rel[1], rel[2]))
					elif pred_tense == 'present':
						new_sent_rels.append(rel)
					else:
						raise AssertionError
			new_rels.append(new_sent_rels)

	if CONVERT_MODAL:
		prev_rels = copy.deepcopy(new_rels)
		new_rels = []
		for (sent_rels, sent_ddp) in zip(prev_rels, ddp_res):
			new_sent_rels = []
			for rel in sent_rels:
				if rel[1] != 'SVO':
					new_sent_rels.append(rel)
				else:
					pred_modal = check_rel_pred_modal(rel, sent_rels, sent_ddp)
					if pred_modal is not None:
						new_sent_rels.append(((rel[0][0], pred_modal+'·'+rel[0][1], rel[0][2]), rel[1], rel[2]))
						continue
					else:
						new_sent_rels.append(rel)
			new_rels.append(new_sent_rels)

	assert len(cur_rels) == len(new_rels)
	for (sent_cur_rels, sent_new_rels) in zip(cur_rels, new_rels):
		assert len(sent_cur_rels) == len(sent_new_rels)
	return new_rels


# initializing global registers for filterring triples with stopword/number arguments
stop_word_list = ['有', '没有', '还有', '还', '是', '你', '我', '他', '她', '它', '他们', '她们', '它们', '带', '的', '任',
				  '这', '那', '这些', '那些', '哪', '哪些', '这个', '那个', '这里', '那里', '里', '可能', '之', '个',
				  '能', '内', '外', '等', '下', '上']


def coarsify(coarse_infos, rel_res, subj_flag, obj_flag, keep_same_flag):
	new_rel_res = []
	for sent_id, sent_res in enumerate(rel_res):
		new_sent_res = []
		for rid in range(len(sent_res)):
			rel = sent_res[rid]
			new_subj = rel[0][0]
			new_obj = rel[0][2]
			pred_splitted = rel[0][1].split('·')
			if subj_flag and rel[2][0] is not None:
				subj_node = coarse_infos[sent_id].nodes[rel[2][0]]
				new_subj = coarse_infos[sent_id].process_sub_term(subj_node)
				# only replace the arguments with their projection when it does not mess up with the predicate
				for pred_chunk in pred_splitted:
					if pred_chunk not in ['的', '在'] and pred_chunk in new_subj:
						new_subj = rel[0][0]
						break
				if rel[0][0] not in new_subj:
					# print("rel[0][0]: ", rel[0][0])
					# print("new_subj: ", new_subj)
					if new_subj not in rel[0][0]:
						# print("rel[0][0]: ", rel[0][0])
						# print("new_subj: ", new_subj)
						# raise AssertionError
						pass
					new_subj = rel[0][0]
			if obj_flag and rel[2][2] is not None:
				obj_node = coarse_infos[sent_id].nodes[rel[2][2]]
				new_obj = coarse_infos[sent_id].process_sub_term(obj_node)
				# only replace the arguments with their projection when it does not mess up with the predicate
				for pred_chunk in pred_splitted:
					if pred_chunk not in ['的', '在'] and pred_chunk in new_obj:
						new_obj = rel[0][2]
						break
				if rel[0][2] not in new_obj:
					# print("rel[0][2]: ", rel[0][2])
					# print("new obj: ", new_obj)
					if new_obj not in rel[0][2]:
						# print("rel[0][2]: ", rel[0][2])
						# print("new obj: ", new_obj)
						# raise AssertionError
						pass
					new_obj = rel[0][2]  # if the coarsified argument is not as long as the original one, then we don't change that
			new_rel = ((new_subj, rel[0][1], new_obj), rel[1], rel[2])
			if keep_same_flag or new_subj != rel[0][0] or new_obj != rel[0][2]:
				new_sent_res.append(new_rel)
		new_rel_res.append(new_sent_res)
	assert len(rel_res) == len(new_rel_res)
	return new_rel_res


def post_processing(fine_res, coarse_res, ddp_res, corenlp_pos_tags, token_normalizer, coarse_infos, vcmp_bucket=None, amendment_counts=None,
					fine_stop_word_count_bucket=None, fine_digit_excluded_count=0, MUST_INCLUDE_CHINESE_flag=True,
					coarse_stop_word_count_bucket=None, coarse_digit_excluded_count=0, KEEP_ONLY_SVO=True, DEBUG=True,
					ignore_postag=False):
	ADD_HALF_COARSE = False
	# reformat from (([S_id, S], [V_id, V], [O_id, O]), 'SVO') into ((S, V, O), 'SVO', (S_id, V_id, O_id))
	if vcmp_bucket is None:
		vcmp_bucket = {}
	if amendment_counts is None:
		amendment_counts = {}
	fine_res = reformat_cur_rels(fine_res)
	coarse_res = reformat_cur_rels(coarse_res)
	reformatted_t = time.time()
	# split n-ary relations into sets of binary relations (still keeps the original n-ary relations (DOB) in the list)
	fine_res = translate_nary_to_binaries(fine_res)
	coarse_res = translate_nary_to_binaries(coarse_res)
	translated_t = time.time()

	token_normalizer.rel_token_normalize(fine_res, coarse_res)
	amend_fine_res, amend_coarse_res, fine_res, coarse_res, possible_res, cur_vcmp_dict, cur_amendment_counts = \
		build_amendment_relations(ddp_res, fine_res, coarse_res, corenlp_pos_tags, DEBUG, discard_residue=True,
								  ignore_postag=ignore_postag)

	merge_dict(vcmp_bucket, cur_vcmp_dict)
	merge_dict(amendment_counts, cur_amendment_counts)
	amendment_t = time.time()

	fine_res, fine_digit_excluded_count = filter_triples_stopwords(fine_res, stop_word_list,
																   fine_stop_word_count_bucket,
																   fine_digit_excluded_count, MUST_INCLUDE_CHINESE_flag)
	coarse_res, coarse_digit_excluded_count = filter_triples_stopwords(coarse_res, stop_word_list,
																	   coarse_stop_word_count_bucket,
																	   coarse_digit_excluded_count,
																	   MUST_INCLUDE_CHINESE_flag)

	amend_fine_res, _ = filter_triples_stopwords(amend_fine_res, stop_word_list,
												 MUST_INCLUDE_CHINESE_flag=MUST_INCLUDE_CHINESE_flag)
	amend_coarse_res, _ = filter_triples_stopwords(amend_coarse_res, stop_word_list,
												   MUST_INCLUDE_CHINESE_flag=MUST_INCLUDE_CHINESE_flag)
	possible_res, _ = filter_triples_stopwords(possible_res, stop_word_list,
											   	MUST_INCLUDE_CHINESE_flag=MUST_INCLUDE_CHINESE_flag)

	filtered_t = time.time()

	fine_res = check_nct(fine_res, ddp_res)
	coarse_res = check_nct(coarse_res, ddp_res)
	amend_fine_res = check_nct(amend_fine_res, ddp_res)
	amend_coarse_res = check_nct(amend_coarse_res, ddp_res)
	if KEEP_ONLY_SVO:
		fine_res = only_keep_svo(fine_res)
		coarse_res = only_keep_svo(coarse_res)
		amend_fine_res = only_keep_svo(amend_fine_res)
		amend_coarse_res = only_keep_svo(amend_coarse_res)

	amend_coarse_res = coarsify(coarse_infos, amend_coarse_res, subj_flag=True, obj_flag=True, keep_same_flag=True)

	if ADD_HALF_COARSE:
		amend_crossed_res_1 = coarsify(coarse_infos, amend_fine_res, subj_flag=True, obj_flag=False, keep_same_flag=False)
		amend_crossed_res_2 = coarsify(coarse_infos, amend_fine_res, subj_flag=False, obj_flag=True, keep_same_flag=False)
		crossed_res_1 = coarsify(coarse_infos, fine_res, subj_flag=True, obj_flag=False, keep_same_flag=False)
		crossed_res_2 = coarsify(coarse_infos, fine_res, subj_flag=False, obj_flag=True, keep_same_flag=False)
		amend_crossed_res = [x+y for x, y in zip(amend_crossed_res_1, amend_crossed_res_2)]
		crossed_res = [x+y for x, y in zip(crossed_res_1, crossed_res_2)]
	else:
		amend_crossed_res = [[] for x in amend_fine_res]
		crossed_res = [[] for x in fine_res]
	return fine_res, coarse_res, amend_fine_res, amend_coarse_res, crossed_res, amend_crossed_res, possible_res, \
		   reformatted_t, translated_t, filtered_t, amendment_t, \
		   fine_digit_excluded_count, coarse_digit_excluded_count


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--data_entry_filename', default='./webhose_data_entries_with_corenlp_ner_and_parse.json',
						type=str)
	parser.add_argument('-o', '--output_data_filename',
						default='./webhose_data_entries_with_corenlp_ner_and_parse_%d.json', type=str)
	parser.add_argument('-j', '--json_stats_filename', default='./rel_counts_%d.json')
	parser.add_argument('-s', '--slice_id', type=int)
	parser.add_argument('-n', '--ner_stats',
						default='./webhose_corenlpner_stats.json')
	parser.add_argument('--total_slices', type=int, default=8)
	parser.add_argument('--ready_ddp', type=int, default=0)

	parser.add_argument('--debug', type=int, default=0)

	args = parser.parse_args()
	args.ready_ddp = True if args.ready_ddp > 0 else False

	fine_stop_word_count_bucket = {}
	coarse_stop_word_count_bucket = {}
	for stop_word in stop_word_list:
		fine_stop_word_count_bucket[stop_word] = 0
		coarse_stop_word_count_bucket[stop_word] = 0
	fine_digit_excluded_count = 0  # the count of arguments that appear to be pure numbers in fine-grained extracted relations
	coarse_digit_excluded_count = 0  # the count of arguments that appear to be pure numbers in coarse-grained extracted relations

	vcmp_bucket = {}
	amendment_counts = {'pob': 0, 'mt': 0, 'vv': 0, 'vcmp': 0, 'headneg': 0, 'attobj': 0, 'attsub': 0, 'attatt': 0,
						'vobvob': 0, 'copsbj': 0}

	amend_fine_count = 0
	amend_coarse_count = 0
	crossed_count = 0
	amend_crossed_count = 0

	KEEP_ONLY_SVO = True
	DEBUG = (args.debug > 0)
	MUST_INCLUDE_CHINESE_flag = True

	extract_time_sum = 0
	reformat_time_sum = 0
	translate_time_sum = 0
	filter_time_sum = 0
	amendment_time_sum = 0

	# read in the data input.
	data_entry_filename = args.data_entry_filename
	output_data_filename = args.output_data_filename % args.slice_id
	json_stats_filename = args.json_stats_filename % args.slice_id
	print(f"Working on slice number {args.slice_id}!")

	with open(data_entry_filename, 'r', encoding='utf8') as input_fp:
		total_length = 0
		for line in input_fp:
			total_length += 1

	slice_standard_size = (total_length // args.total_slices)
	slice_start = args.slice_id * slice_standard_size
	slice_end = (args.slice_id + 1) * slice_standard_size if (args.slice_id != args.total_slices - 1) else total_length

	print(f"Total entries: {total_length}; attending to entries from {slice_start} to {slice_end}!")

	ddp = ddparser.DDParser(encoding_model='transformer')
	# annotator = CorenlpAnnotator(args.ner_stats)
	token_normalizer = Token_Normalizer(remove_from_args=False)

	st = time.time()

	input_fp = open(data_entry_filename, 'r', encoding='utf8')
	output_fp = open(output_data_filename, 'w', encoding='utf8')
	for ent_idx, in_line in enumerate(input_fp):
		if ent_idx < slice_start:
			continue

		# if ent_idx < 2347600:
		#	continue

		if ent_idx >= slice_end:
			print(f"Last index (not included): {ent_idx}!")
			break

		entry = json.loads(in_line)

		if ent_idx % 100 == 0 and ent_idx > 0:
			ct = time.time()
			dur = ct - st
			dur_h = int(dur) / 3600
			dur_m = (int(dur) % 3600) / 60
			dur_s = int(dur) % 60
			print(ent_idx, 'time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
			print(f"extract: {extract_time_sum}; reformat: {reformat_time_sum}; "
				  f"trans: {translate_time_sum}; filter: {filter_time_sum}; amend: {amendment_time_sum}.")
			# annotator.print_stats()
			if ent_idx % 1000 == 0:
				token_normalizer.print_stats()
				print("Number of amendments per amendment rule: ")
				print(amendment_counts)

		entry_st = time.time()
		sentences = entry['splitted_text']
		corenlp_pos_tags = entry['corenlp_pos_tags']

		if args.ready_ddp:
			ddp_res = entry['ddp_lbls']
			for sid in range(len(ddp_res)):
				if len(''.join(ddp_res[sid]['word'])) != len(sentences[sid]):
					print("ddpres words not the same as sentences!", file=sys.stderr)
					print(ddp_res[sid])
					print(sentences[sid])
		else:
			ddp_res = ddp.parse(sentences)

		assert len(ddp_res) == len(entry['splitted_text'])
		for sent_ddp, sent_spl in zip(ddp_res, entry['splitted_text']):
			if len(''.join(sent_ddp['word'])) != len(sent_spl):
				print(sent_ddp)
				print(sent_spl)
				raise AssertionError

		fine_res = []
		coarse_res = []
		coarse_infos = []
		for r in ddp_res:
			fine_info = FineGrainedInfo(r)
			individual_fine_res = fine_info.parse()
			for rel in individual_fine_res:
				for a in rel[0]:
					assert a is None or len(a) == 2
			fine_res.append(individual_fine_res)

			coarse_info = CoarseGrainedInfo(r)
			coarse_infos.append(coarse_info)
			individual_coarse_res = coarse_info.parse()
			for rel in individual_fine_res:
				for a in rel[0]:
					assert a is None or len(a) == 2
			coarse_res.append(individual_coarse_res)

		extracted_t = time.time()

		fine_res, coarse_res, amend_fine_res, amend_coarse_res, crossed_res, amend_crossed_res, possible_res, \
		reformatted_t, translated_t, filtered_t, amendment_t, \
		fine_digit_excluded_count, coarse_digit_excluded_count = post_processing(fine_res, coarse_res, ddp_res,
																				 corenlp_pos_tags, token_normalizer,
																				 coarse_infos=coarse_infos,
																				 vcmp_bucket=vcmp_bucket,
																				 amendment_counts=amendment_counts,
																				 fine_stop_word_count_bucket=fine_stop_word_count_bucket,
																				 fine_digit_excluded_count=fine_digit_excluded_count,
																				 MUST_INCLUDE_CHINESE_flag=MUST_INCLUDE_CHINESE_flag,
																				 coarse_stop_word_count_bucket=coarse_stop_word_count_bucket,
																				 coarse_digit_excluded_count=coarse_digit_excluded_count,
																				 KEEP_ONLY_SVO=KEEP_ONLY_SVO,
																				 DEBUG=DEBUG)

		entry['fine_rels'] = fine_res
		entry['coarse_rels'] = coarse_res
		entry['ddp_lbls'] = ddp_res
		entry['amend_fine_rels'] = amend_fine_res
		entry['amend_coarse_rels'] = amend_coarse_res
		entry['crossed_rels'] = crossed_res
		entry['amend_crossed_rels'] = amend_crossed_res
		entry['possible_rels'] = possible_res

		amend_fine_count += sum([len(x) for x in amend_fine_res])
		amend_coarse_count += sum([len(x) for x in amend_coarse_res])
		crossed_count += sum([len(x) for x in crossed_res])
		amend_crossed_count += sum([len(x) for x in amend_crossed_res])

		out_line = json.dumps(entry, ensure_ascii=False)
		output_fp.write(out_line + '\n')

		extract_time_sum += (extracted_t - entry_st)
		reformat_time_sum += (reformatted_t - extracted_t)
		translate_time_sum += (translated_t - reformatted_t)
		amendment_time_sum += (amendment_t - translated_t)
		filter_time_sum += (filtered_t - amendment_t)
		ent_idx += 1

	print("Fine grained relations bucket of filtered-out relations according to stop words: ")
	print(fine_stop_word_count_bucket)
	print("")

	print("Coarse grained relations bucket of filtered-out relations according to stop words: ")
	print(coarse_stop_word_count_bucket)
	print("")

	print(f"Number of fine grained relations filtered out because of number arguments: {fine_digit_excluded_count}")
	print(f"Number of coarse grained relations filtered out because of number arguments: {coarse_digit_excluded_count}")

	print(f"Number of fine-grained relations additionally found: {amend_fine_count}")
	print(f"Number of coarse-grained relations additionally found: {amend_coarse_count}")
	print(f"Number of crossed relations found: {crossed_count}")
	print(f"Number of crossed relations frond from additional relations: {amend_crossed_count}")

	print("Bucket of complements involved in V_CMP amendment: ")
	print(vcmp_bucket)

	stats_dict = {'fine_stop_word_count_bucket': fine_stop_word_count_bucket,
				  'coarse_stop_word_count_bucket': coarse_stop_word_count_bucket,
				  'fine_digit_excluded_count': fine_digit_excluded_count,
				  'coarse_digit_excluded_count': coarse_digit_excluded_count,
				  'amend_fine_count': amend_fine_count,
				  'amend_coarse_count': amend_coarse_count,
				  'crossed_count': crossed_count,
				  'amend_crossed_count': amend_crossed_count,
				  'vcmp_bucket': vcmp_bucket}
	with open(json_stats_filename, 'w', encoding='utf8') as stats_fp:
		json.dump(stats_dict, stats_fp, indent=4, ensure_ascii=False)

	input_fp.close()
	output_fp.close()

	print("Finished!")


def parse_coref():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--data_entry_filename', default='./webhose_data_entries_with_corenlp_ner_and_parse.json',
						type=str)
	parser.add_argument('-o', '--output_data_filename',
						default='./webhose_data_entries_with_corenlp_ner_and_withcoref_parse_%d.json', type=str)
	parser.add_argument('-j', '--json_stats_filename', default='./rel_counts_coref_%d.json')
	parser.add_argument('-s', '--slice_id', type=int)
	parser.add_argument('--total_slices', type=int, default=8)

	parser.add_argument('--debug', type=int, default=0)

	args = parser.parse_args()

	fine_stop_word_count_bucket = {}
	coarse_stop_word_count_bucket = {}
	for stop_word in stop_word_list:
		fine_stop_word_count_bucket[stop_word] = 0
		coarse_stop_word_count_bucket[stop_word] = 0
	fine_digit_excluded_count = 0  # the count of arguments that appear to be pure numbers in fine-grained extracted relations
	coarse_digit_excluded_count = 0  # the count of arguments that appear to be pure numbers in coarse-grained extracted relations

	vcmp_bucket = {}
	amendment_counts = {'pob': 0, 'mt': 0, 'vv': 0, 'vcmp': 0, 'headneg': 0, 'attobj': 0, 'attsub': 0, 'attatt': 0,
						'vobvob': 0, 'copsbj': 0}

	amend_fine_count = 0
	amend_coarse_count = 0
	crossed_count = 0
	amend_crossed_count = 0

	diff_when_corefed_count = 0
	coref_addrel_entry_count = 0

	KEEP_ONLY_SVO = True
	DEBUG = (args.debug > 0)
	MUST_INCLUDE_CHINESE_flag = True

	extract_time_sum = 0
	reformat_time_sum = 0
	translate_time_sum = 0
	filter_time_sum = 0
	amendment_time_sum = 0

	# read in the data input.
	data_entry_filename = args.data_entry_filename
	output_data_filename = args.output_data_filename % args.slice_id
	json_stats_filename = args.json_stats_filename % args.slice_id
	print(f"Working on slice number {args.slice_id}!")

	with open(data_entry_filename, 'r', encoding='utf8') as input_fp:
		total_length = 0
		for line in input_fp:
			total_length += 1

	slice_standard_size = (total_length // args.total_slices)
	slice_start = args.slice_id * slice_standard_size
	slice_end = (args.slice_id + 1) * slice_standard_size if (args.slice_id != args.total_slices - 1) else total_length

	print(f"Total entries: {total_length}; attending to entries from {slice_start} to {slice_end}!")

	ddp = ddparser.DDParser(encoding_model='transformer')
	token_normalizer = Token_Normalizer(remove_from_args=False)

	st = time.time()

	input_fp = open(data_entry_filename, 'r', encoding='utf8')
	output_fp = open(output_data_filename, 'w', encoding='utf8')
	for ent_idx, in_line in enumerate(input_fp):
		if ent_idx < slice_start:
			continue
		if ent_idx >= slice_end:
			print(f"Last index (not included): {ent_idx}!")
			break

		entry = json.loads(in_line)

		if ent_idx % 100 == 0 and ent_idx > 0:
			ct = time.time()
			dur = ct - st
			dur_h = int(dur) / 3600
			dur_m = (int(dur) % 3600) / 60
			dur_s = int(dur) % 60
			print(ent_idx, 'time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
			print(f"extract: {extract_time_sum}; reformat: {reformat_time_sum}; "
				  f"trans: {translate_time_sum}; filter: {filter_time_sum}; amend: {amendment_time_sum}.")
			# annotator.print_stats()
			if ent_idx % 1000 == 0:
				token_normalizer.print_stats()
				print("Number of amendments per amendment rule: ")
				print(amendment_counts)
				print(f"diff_when_corefed_count: {diff_when_corefed_count};")
				print(f"coref_addrel_entry_count: {coref_addrel_entry_count};")

		entry_st = time.time()
		sentences = entry['splitted_text_corefed']
		corenlp_pos_tags = entry['corenlp_pos_tags_corefed']
		coref_orig_same_flag = True
		for corefed_sent, orig_sent in zip(sentences, entry['splitted_text']):
			if corefed_sent != orig_sent:
				coref_orig_same_flag = False
				break

		if sentences is not None and not coref_orig_same_flag:
			diff_when_corefed_count += 1
			has_addrel_flag = False

			ddp_res = ddp.parse(sentences)
			fine_res = []
			coarse_res = []
			coarse_infos = []
			for r in ddp_res:
				fine_info = FineGrainedInfo(r)
				individual_fine_res = fine_info.parse()
				for rel in individual_fine_res:
					for a in rel[0]:
						assert a is None or len(a) == 2
				fine_res.append(individual_fine_res)

				coarse_info = CoarseGrainedInfo(r)
				coarse_infos.append(coarse_info)
				individual_coarse_res = coarse_info.parse()
				for rel in individual_fine_res:
					for a in rel[0]:
						assert a is None or len(a) == 2
				coarse_res.append(individual_coarse_res)

			extracted_t = time.time()

			fine_res, coarse_res, amend_fine_res, amend_coarse_res, crossed_res, amend_crossed_res, possible_res, \
			reformatted_t, translated_t, filtered_t, amendment_t, \
			fine_digit_excluded_count, coarse_digit_excluded_count = post_processing(fine_res, coarse_res, ddp_res,
																					 corenlp_pos_tags, token_normalizer,
																					 coarse_infos=coarse_infos,
																					 vcmp_bucket=vcmp_bucket,
																					 amendment_counts=amendment_counts,
																					 fine_stop_word_count_bucket=fine_stop_word_count_bucket,
																					 fine_digit_excluded_count=fine_digit_excluded_count,
																					 MUST_INCLUDE_CHINESE_flag=MUST_INCLUDE_CHINESE_flag,
																					 coarse_stop_word_count_bucket=coarse_stop_word_count_bucket,
																					 coarse_digit_excluded_count=coarse_digit_excluded_count,
																					 KEEP_ONLY_SVO=KEEP_ONLY_SVO,
																					 DEBUG=DEBUG)

			tuples = [['fine_rels', fine_res], ['coarse_rels', coarse_res], ['amend_fine_rels', amend_fine_res],
					  ['amend_coarse_rels', amend_coarse_res], ['crossed_rels', crossed_res],
					  ['amend_crossed_rels', amend_crossed_res], ['possible_rels', possible_res]]
			for key, cand in tuples:
				if entry[key] is None:
					entry[key + '_corefed'] = cand
				else:
					additional_rels = []
					for sid, sent_rels in enumerate(cand):
						additional_sent_rels = []
						serialized_sent_rels = [serialize_rel(r) for r in entry[key][sid]]
						for rel in sent_rels:
							if serialize_rel(rel) not in serialized_sent_rels:
								additional_sent_rels.append(rel)
						additional_rels.append(additional_sent_rels)
						if len(additional_sent_rels) > 0:
							has_addrel_flag = True
					entry[key+'_corefed'] = additional_rels
					assert len(entry[key]) == len(entry[key+'_corefed'])

			entry['ddp_lbls_corefed'] = ddp_res

			extract_time_sum += (extracted_t - entry_st)
			reformat_time_sum += (reformatted_t - extracted_t)
			translate_time_sum += (translated_t - reformatted_t)
			amendment_time_sum += (amendment_t - translated_t)
			filter_time_sum += (filtered_t - amendment_t)

			amend_fine_count += sum([len(x) for x in amend_fine_res])
			amend_coarse_count += sum([len(x) for x in amend_coarse_res])
			crossed_count += sum([len(x) for x in crossed_res])
			amend_crossed_count += sum([len(x) for x in amend_crossed_res])

			if has_addrel_flag:
				coref_addrel_entry_count += 1

		else:
			entry['fine_rels_corefed'] = None
			entry['coarse_rels_corefed'] = None
			entry['amend_fine_rels_corefed'] = None
			entry['amend_coarse_rels_corefed'] = None
			entry['crossed_rels_corefed'] = None
			entry['amend_crossed_rels_corefed'] = None
			entry['possible_rels_corefed'] = None

		out_line = json.dumps(entry, ensure_ascii=False)
		output_fp.write(out_line + '\n')

		ent_idx += 1

	print("Fine grained relations bucket of filtered-out relations according to stop words: ")
	print(fine_stop_word_count_bucket)
	print("")

	print("Coarse grained relations bucket of filtered-out relations according to stop words: ")
	print(coarse_stop_word_count_bucket)
	print("")

	print(f"Number of fine grained relations filtered out because of number arguments: {fine_digit_excluded_count}")
	print(f"Number of coarse grained relations filtered out because of number arguments: {coarse_digit_excluded_count}")

	print(f"Number of fine-grained relations additionally found: {amend_fine_count}")
	print(f"Number of coarse-grained relations additionally found: {amend_coarse_count}")
	print(f"Number of crossed relations found: {crossed_count}")
	print(f"Number of crossed relations frond from additional relations: {amend_crossed_count}")

	print("Bucket of complements involved in V_CMP amendment: ")
	print(vcmp_bucket)

	stats_dict = {'fine_stop_word_count_bucket': fine_stop_word_count_bucket,
				  'coarse_stop_word_count_bucket': coarse_stop_word_count_bucket,
				  'fine_digit_excluded_count': fine_digit_excluded_count,
				  'coarse_digit_excluded_count': coarse_digit_excluded_count,
				  'amend_fine_count': amend_fine_count,
				  'amend_coarse_count': amend_coarse_count,
				  'crossed_count': crossed_count,
				  'amend_crossed_count': amend_crossed_count,
				  'vcmp_bucket': vcmp_bucket}
	with open(json_stats_filename, 'w', encoding='utf8') as stats_fp:
		json.dump(stats_dict, stats_fp, indent=4, ensure_ascii=False)

	input_fp.close()
	output_fp.close()

	print("Finished!")


'''
def prune_parsed_file():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--parsed_entries', default='./webhose_data_entries_with_parse.json', type=str)
	parser.add_argument('-o', '--pruned_entries', default='./webhose_data_entries_with_parse.json', type=str)

	args = parser.parse_args()

	stop_word_list = ['有', '没有', '还有', '还', '是', '你', '我', '他', '她', '它', '他们', '她们', '它们', '带', '的', '任',
					  '这', '那', '这些', '那些', '哪', '哪些', '这个', '那个', '这里', '那里', '里', '可能']

	data_entries = []
	with open(args.parsed_entries, 'r', encoding='utf8') as fp:
		lidx = 0
		for line in fp:
			if lidx % 1000 == 0 and lidx > 0:
				print(lidx)
			doc = json.loads(line)
			doc['fine_rels'], _ = filter_triples_stopwords(doc['fine_rels'], stop_word_list,
														MUST_INCLUDE_CHINESE_flag=True)
			doc['coarse_rels'], _ = filter_triples_stopwords(doc['coarse_rels'], stop_word_list,
														  MUST_INCLUDE_CHINESE_flag=True)
			doc['amend_fine_rels'], _ = filter_triples_stopwords(doc['amend_fine_rels'], stop_word_list,
															  MUST_INCLUDE_CHINESE_flag=True)
			doc['amend_coarse_rels'], _ = filter_triples_stopwords(doc['amend_coarse_rels'], stop_word_list,
																MUST_INCLUDE_CHINESE_flag=True)
			data_entries.append(doc)
			lidx += 1

	print("Dumping......")
	with open(args.pruned_entries, 'w', encoding='utf8') as fp:
		for doc in data_entries:
			fp.write(json.dumps(doc, ensure_ascii=False)+'\n')
'''

if __name__ == '__main__':
	main()
# prune_parsed_file()
