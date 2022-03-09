import json
from StanfordCoreNLP import *
import argparse
import time
import copy
import sys


def ner_pos_per_sentence(client, sent, ner_token_bucket=None, total_count_special_token_misplaced_spans=0):
	if ner_token_bucket is None:
		ner_token_bucket = {}

	ann = client.annotate(sent)
	sent_ner_mentions = []
	sent_ner_spans = []
	sent_pos_tags = {}
	for ann_sentence in ann.sentence:
		for token in ann_sentence.token:
			if token.ner not in ['O', 'PERCENT']:
				if token.ner not in ner_token_bucket:
					ner_token_bucket[token.ner] = 1
				else:
					ner_token_bucket[token.ner] += 1
				span = [token.beginChar, token.endChar]
				if token.word != sent[span[0]:span[1]]:
					print("sentence: ", sent)
					for _tok in ann_sentence.token:
						print(f"{_tok.word}; [{_tok.beginChar}, {_tok.endChar}]; {sent[_tok.beginChar:_tok.endChar]}")
					print(f"token.word: {token.word}")
					print(f"sent[span[0]:span[1]]: {sent[span[0]:span[1]]}")
					total_count_special_token_misplaced_spans += 1
				# raise AssertionError
				mention = {'word': token.word, 'label': token.ner, 'span': span}
				sent_ner_mentions.append(mention)
				sent_ner_spans.append(span)
			if token.word not in sent_pos_tags:
				sent_pos_tags[token.word] = []
			sent_pos_tags[token.word].append(token.pos)
	for word in sent_pos_tags:
		sent_pos_tags[word] = list(set(sent_pos_tags[word]))
	return sent_ner_mentions, sent_ner_spans, sent_pos_tags, total_count_special_token_misplaced_spans


def main_ner(input_fn, output_fn, stats_fn, num_slices, slice_id, port_id):
	total_count_special_token_misplaced_spans = 0
	entries_count = 0

	if slice_id >= 0:
		output_fn = output_fn % slice_id
		stats_fn = stats_fn % slice_id

	input_fp = open(input_fn, 'r', encoding='utf8')
	output_fp = open(output_fn, 'w', encoding='utf8')
	for line in input_fp:
		if entries_count % 100000 == 0:
			print(entries_count)
		entries_count += 1
	print("entries_count: ", entries_count)
	input_fp.close()
	input_fp = open(input_fn, 'r', encoding='utf8')

	if slice_id >= 0:
		slice_canonical_size = entries_count // num_slices
		slice_st = slice_canonical_size * slice_id
		slice_end = slice_canonical_size * (slice_id + 1) if slice_id < num_slices-1 else entries_count
	else:
		slice_st = 0
		slice_end = entries_count

	print(f"Working on slice {slice_id}/{num_slices}; entry id {slice_st} to {slice_end}!")

	StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

	ner_token_bucket = {}

	with CoreNLPClient(
			annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'],
			properties=StanfordCoreNLP_chinese_properties,
			endpoint='http://localhost:%d' % port_id,
			timeout=90000,
			memory='24G',
			be_quiet=True) as client:

		st = time.time()
		doc_id = 0
		for input_line in input_fp:
			if doc_id < slice_st or doc_id >= slice_end:
				doc_id += 1
				continue
			if doc_id % 1000 == 0 and doc_id > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) / 3600
				dur_m = (int(dur) % 3600) / 60
				dur_s = int(dur) % 60
				print(doc_id, '; time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
				print("ner_token_bucket: ", ner_token_bucket)
				print(f"current_count_special_token_misplaced_spans: {total_count_special_token_misplaced_spans}")

			doc = json.loads(input_line)

			doc['corenlp_ner_mentions'] = []  # in each mention: {'word': word, 'label': label, 'span': span}
			doc['corenlp_ner_spans'] = []
			doc['corenlp_pos_tags'] = []
			for sent_id, sent in enumerate(doc['splitted_text']):
				sent_ner_mentions, sent_ner_spans, sent_pos_tags, total_count_special_token_misplaced_spans \
					= ner_pos_per_sentence(client, sent, ner_token_bucket, total_count_special_token_misplaced_spans)
				doc['corenlp_ner_mentions'].append(sent_ner_mentions)
				doc['corenlp_ner_spans'].append(sent_ner_spans)
				doc['corenlp_pos_tags'].append(sent_pos_tags)
			doc_id += 1

			# write the annotated entry back into the output fp
			out_line = json.dumps(doc, ensure_ascii=False)
			output_fp.write(out_line + '\n')

			assert len(doc['corenlp_ner_mentions']) == len(doc['splitted_text'])
			assert len(doc['corenlp_ner_spans']) == len(doc['splitted_text'])

	print("ner_token_bucket: ")
	print(ner_token_bucket)

	input_fp.close()
	output_fp.close()

	with open(stats_fn, 'w', encoding='utf8') as fp:
		json.dump(ner_token_bucket, fp, ensure_ascii=False)

	print(f"total_count_special_token_misplaced_spans: {total_count_special_token_misplaced_spans}")


def merge_nerpos_results(output_fn, stats_fn, merged_output_fn, merged_stats_fn, num_slices):
	print(f"Merging NER and POS tagging results from {num_slices} slices!")

	output_fp = open(merged_output_fn, 'w', encoding='utf8')

	merged_token_bucket = {}

	for slice_id in range(num_slices):
		print(f"Reading in slice {slice_id}!")
		sliced_fp = open(output_fn%slice_id, 'r', encoding='utf8')
		for lidx, line in enumerate(sliced_fp):
			if lidx % 10000 == 0:
				print(lidx)
			output_fp.write(line.strip('\n')+'\n')
		sliced_fp.close()
		with open(stats_fn%slice_id, 'r', encoding='utf8') as sfp:
			current_bucket = json.load(sfp)
			for key in current_bucket:
				if key not in merged_token_bucket:
					merged_token_bucket[key] = 0
				merged_token_bucket[key] += current_bucket[key]

	output_fp.close()

	print("NER token bucket in total: ")
	print(merged_token_bucket)

	with open(merged_stats_fn, 'w', encoding='utf8') as sfp:
		json.dump(merged_token_bucket, sfp, indent=4, ensure_ascii=False)

	print("Done.")


'''Above are routines for NER and POS tagging; below are routines for coreference resolution!'''


def merge_texts(splitted, threshold):
	merged = []
	offsets = []
	cur_m = []
	accum_len = 0
	next_offset = 0
	assert threshold > 512
	for sid in range(len(splitted) + 1):
		if sid < len(splitted):
			sent = splitted[sid]
		if sid == len(splitted) or accum_len + len(sent) > threshold:
			cur_merged = "。".join(cur_m)
			offsets.append([next_offset, sid])
			next_offset = sid
			merged.append(cur_merged)
			cur_m = []
			accum_len = 0
		if sid < len(splitted):
			cur_m.append(sent)
			accum_len += len(sent)

	assert next_offset == len(splitted)
	assert len(offsets) == len(merged)
	return merged, offsets


def fetch_coref(merged_texts, offsets, splitted_texts, client, doc_id, DEBUG, repeated):
	doc_num_coref_chains = 0
	# 				if len(coref_chains) not in coref_chain_bucket:
	# 					coref_chain_bucket[len(coref_chains)] = 0
	# 				coref_chain_bucket[len(coref_chains)] += 1

	splitted_text_corefed = []
	coref_replacements = []
	coref_failed = False
	mismatched_flag = False
	overlap_count = 0

	#if len(merged_texts) > 1:
	#	print('!')

	for text, (cur_spl_start, cur_spl_end) in zip(merged_texts, offsets):

		this_splitted = splitted_texts[cur_spl_start:cur_spl_end]
		try:
			ann = client.annotate(text)
		except Exception as e:
			if repeated:
				print(e)
				print(f"Error occured for doc id {doc_id}!")
				print("Text: ")
				print(text)
			coref_failed = True
			break

		ann_spl_mapping = {}  # ann_sent_id: [spl_sent_id, bias]
		last_splid = 0
		all_mapped = True
		sent_lists = []
		for i in range(len(ann.sentence)):
			sent_list = [token.word for token in ann.sentence[i].token]
			sent_lem = ''.join(sent_list).strip('。')
			sent_mapped = False
			for splid, spl_sent in enumerate(this_splitted):
				if splid < last_splid:
					continue
				match_bias = spl_sent.find(sent_lem)  # the first token of the match
				if match_bias >= 0:
					ann_spl_mapping[i] = [splid, match_bias]
					sent_mapped = True
					break
			if not sent_mapped:
				all_mapped = False
			sent_lists.append(sent_list)

		coref_chains = ann.corefChain
		doc_num_coref_chains += len(coref_chains)

		if not all_mapped:
			splitted_text_corefed.append(None)
			coref_replacements.append(None)
			mismatched_flag = True
		elif len(coref_chains) == 0:
			splitted_text_corefed.append(None)
			coref_replacements.append(None)
		else:  # if all_mapped
			cur_slice_corefed = []
			assert len(sent_lists) == len(ann.sentence)
			replacement_per_sent = [[] for i in range(len(this_splitted))]
			# find out the replacements to make from all the coreference chains.
			for chain in coref_chains:
				spans = []  # [spl_sent_id, [start_tokid, end_tokid]]
				for ment in chain.mention:
					ment_splid, ment_bias = ann_spl_mapping[ment.sentenceIndex]
					ann_sent_list = sent_lists[ment.sentenceIndex]
					ann_sent_list = [x for x in ann_sent_list if x != '。']
					ment_annstid = len(''.join(ann_sent_list[:ment.beginIndex]))  # len of all tokens before starttok
					ment_annedid = len(''.join(ann_sent_list[:ment.endIndex]))  # len of all tokens before endtok
					if ment_annstid >= ment_annedid:
						print(ann_sent_list)
						print(ment)
						print("ment_annstid >= ment_annedid!!!", file=sys.stderr)
					cur_span = [ment_splid, [ment_bias + ment_annstid, ment_bias + ment_annedid]]
					ann_tokens = ''.join(ann_sent_list[ment.beginIndex:ment.endIndex])
					spl_tokens = this_splitted[cur_span[0]][cur_span[1][0]:cur_span[1][1]]
					if ann_tokens != spl_tokens:
						print("Mismatch!")
						print("SPL: ")
						print(this_splitted[cur_span[0]])
						print("ANN: ")
						print(''.join(ann_sent_list))
						print(f"ann tokens: {ann_tokens}")
						print(f"spl tokens: {spl_tokens}")
						return None, None, True, False, overlap_count, 0
					spans.append(cur_span)
				prime_span = spans[chain.representative]
				prime_tokens = this_splitted[prime_span[0]][prime_span[1][0]:prime_span[1][1]]
				for span in spans:
					replacement_per_sent[span[0]].append([span[1], prime_tokens])
			for splid in range(len(this_splitted)):
				overlap_flag = False
				replacement_per_sent[splid] = sorted(replacement_per_sent[splid],
													 key=lambda x: x[0][0])  # sort by start index
				new_sent_rep = []
				for repid in range(len(replacement_per_sent[splid])):
					if repid == 0:
						new_sent_rep.append(replacement_per_sent[splid][repid])
						continue
					if replacement_per_sent[splid][repid][0][0] - replacement_per_sent[splid][repid - 1][0][1] < 0:
						# if an overlap is found and the two replacement are not exactly the same
						if replacement_per_sent[splid][repid][0][0] != replacement_per_sent[splid][repid - 1][0][0] or \
								replacement_per_sent[splid][repid][0][1] != replacement_per_sent[splid][repid - 1][0][1] \
								or replacement_per_sent[splid][repid][1] != replacement_per_sent[splid][repid - 1][1]:
							if DEBUG:
								print(f"splid: {splid}; repid: {repid}")
							overlap_flag = True
						continue
					# forget about the replacement if the pronoun is not shorter than the antecedent,
					# because this probably means this prediction is wrong!
					elif replacement_per_sent[splid][repid][0][1] - replacement_per_sent[splid][repid][0][1] >= len(
							replacement_per_sent[splid][repid][1]):
						continue
					else:
						new_sent_rep.append(replacement_per_sent[splid][repid])
				replacement_per_sent[splid] = new_sent_rep
				corefed_sent = ""
				pointer_idx = 0
				for replacement in replacement_per_sent[splid]:
					corefed_sent += this_splitted[splid][pointer_idx:replacement[0][0]]
					corefed_sent += replacement[1]
					pointer_idx = replacement[0][1]
				corefed_sent += this_splitted[splid][pointer_idx:]
				if overlap_flag:
					overlap_count += 1
				cur_slice_corefed.append(corefed_sent)
			splitted_text_corefed.append(cur_slice_corefed)
			coref_replacements.append(replacement_per_sent)
	return splitted_text_corefed, coref_replacements, coref_failed, mismatched_flag, overlap_count, doc_num_coref_chains


def main_coref(input_fn, output_fn, stats_fn, DEBUG, slice_id, num_slices, port_id):
	entries_count = 0
	mismatched_doc_count = 0
	overlap_count = 0
	coref_chain_bucket = {0: 0}
	number_of_chunks_bucket = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

	input_fp = open(input_fn, 'r', encoding='utf8')
	output_fp = open(output_fn, 'w', encoding='utf8')
	for line in input_fp:
		entries_count += 1
	print("entries_count: ", entries_count)
	input_fp.close()
	input_fp = open(input_fn, 'r', encoding='utf8')
	StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

	slice_size = entries_count // num_slices
	slice_start = slice_id * slice_size
	if slice_id != num_slices - 1:
		slice_end = (slice_id + 1) * slice_size
	else:
		slice_end = entries_count

	with CoreNLPClient(
			annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
			properties=StanfordCoreNLP_chinese_properties,
			endpoint='http://localhost:%d' % port_id,
			threads=5,
			timeout=90000,
			memory='24G',
			be_quiet=True) as client:
		st = time.time()
		doc_id = 0
		print(f"fetching coreference for document number {slice_start} through {slice_end}")
		for input_line in input_fp:
			if doc_id < slice_start:
				doc_id += 1
				continue
			elif doc_id >= slice_end:
				break

			if doc_id % 100 == 0 and doc_id > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) / 3600
				dur_m = (int(dur) % 3600) / 60
				dur_s = int(dur) % 60
				print(doc_id, '; time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
				if doc_id % 1000 == 0:
					print(f"entries count: {entries_count}")
					print(f"mismatched doc count: {mismatched_doc_count}")
					print(f"overlap count: {overlap_count}")
					coref_chain_bucket = {k: v for k, v in sorted(coref_chain_bucket.items(), key=lambda item: item[0])}
					print(f"coref chain bucket: {coref_chain_bucket}")
					print(f"number of chunks bucket: {number_of_chunks_bucket}")
			doc = json.loads(input_line)
			merged_texts, offsets = merge_texts(doc['splitted_text'], 1536)
			if len(merged_texts) not in number_of_chunks_bucket:
				number_of_chunks_bucket[len(merged_texts)] = 0
			number_of_chunks_bucket[len(merged_texts)] += 1
			if DEBUG:
				print(f"number of chunks: {len(merged_texts)}")

			splitted_text_corefed, coref_replacements, coref_failed, mismatched_flag, cur_overlap_count, \
					doc_num_coref_chains = fetch_coref(merged_texts, offsets, doc['splitted_text'], client, doc_id, DEBUG, repeated=False)

			# If coref fails, run another round of coreference resolution with half the maximum length
			if coref_failed:
				print("Coref failure! Halving the length threshold and retrying......")
				merged_texts, offsets = merge_texts(doc['splitted_text'], 768)
				splitted_text_corefed, coref_replacements, coref_failed, mismatched_flag, cur_overlap_count, \
				doc_num_coref_chains = fetch_coref(merged_texts, offsets, doc['splitted_text'], client, doc_id, DEBUG, repeated=True)

			if mismatched_flag:
				mismatched_doc_count += 1
			overlap_count += cur_overlap_count

			doc['splitted_text_corefed'] = []
			doc['coref_replacements'] = []

			if coref_failed:
				doc['splitted_text_corefed'] = None
				doc['coref_replacements'] = None
			else:
				if doc_num_coref_chains not in coref_chain_bucket:
					coref_chain_bucket[doc_num_coref_chains] = 0
				coref_chain_bucket[doc_num_coref_chains] += 1

				for slice_corefed, slice_replacements in zip(splitted_text_corefed, coref_replacements):
					if slice_corefed is None or slice_replacements is None:
						doc['splitted_text_corefed'] = None
						doc['coref_replacements'] = None
						break
					else:
						doc['splitted_text_corefed'] += slice_corefed
						doc['coref_replacements'] += slice_replacements

			assert doc['splitted_text_corefed'] is None or len(doc['splitted_text_corefed']) == len(
				doc['splitted_text'])
			out_line = json.dumps(doc, ensure_ascii=False)
			output_fp.write(out_line + '\n')
			doc_id += 1

	input_fp.close()
	output_fp.close()
	print("Finished!")

	stats_json = {
		'entries_count': entries_count,
		'mismatched_doc_count': mismatched_doc_count,
		'overlap_count': overlap_count,
		'coref_chain_bucket': coref_chain_bucket
	}

	print(stats_json)

	with open(stats_fn, 'w', encoding='utf8') as fp:
		json.dump(stats_json, fp, ensure_ascii=False)


def merge_coref(coref_output, coref_merged_output, nerpos_output, num_slices):
	output_fp = open(coref_merged_output, 'w', encoding='utf8')
	ref_fp = open(nerpos_output, 'r', encoding='utf8')
	ref_num_items = 0
	for line in ref_fp:
		ref_num_items += 1
	print(f"Ref number of items: {ref_num_items}!")
	ref_fp.close()
	ref_fp = open(nerpos_output, 'r', encoding='utf8')
	slices_item_id = 0
	for i in range(num_slices):
		input_fp = open(coref_output%i, 'r', encoding='utf8')
		for line in input_fp:
			entry = json.loads(line)
			ref_line = ref_fp.readline()
			ref_entry = json.loads(ref_line)
			entry['corenlp_ner_mentions'] = ref_entry['corenlp_ner_mentions']
			entry['corenlp_ner_spans'] = ref_entry['corenlp_ner_spans']
			entry['corenlp_pos_tags'] = ref_entry['corenlp_pos_tags']
			assert len(entry['corenlp_ner_mentions']) == len(entry['splitted_text'])
			assert len(entry['corenlp_ner_spans']) == len(entry['splitted_text'])
			assert len(entry['corenlp_pos_tags']) == len(entry['splitted_text'])
			out_line = json.dumps(entry, ensure_ascii=False)
			output_fp.write(out_line+'\n')
			slices_item_id += 1
	print(f"Number of entries in all slices: {slices_item_id}")
	output_fp.close()
	output_fp = open(coref_merged_output, 'r', encoding='utf8')
	line_nums = 0
	for line in output_fp:
		line_nums += 1
	output_fp.close()
	print(f"Total number of entries merged: {line_nums}!")
	return


def main_forcoref_ner(input_fn, output_fn, stats_fn):
	total_count_special_token_misplaced_spans = 0
	entries_count = 0

	input_fp = open(input_fn, 'r', encoding='utf8')
	output_fp = open(output_fn, 'w', encoding='utf8')
	for line in input_fp:
		entries_count += 1
	print("entries_count: ", entries_count)
	input_fp.close()
	input_fp = open(input_fn, 'r', encoding='utf8')

	StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

	ner_token_bucket = {}

	with CoreNLPClient(
			annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'],
			properties=StanfordCoreNLP_chinese_properties,
			timeout=90000,
			memory='24G',
			be_quiet=True) as client:

		st = time.time()
		doc_id = 0
		for input_line in input_fp:
			if doc_id % 1000 == 0 and doc_id > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) / 3600
				dur_m = (int(dur) % 3600) / 60
				dur_s = int(dur) % 60
				print(doc_id, '; time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
				print("ner_token_bucket: ", ner_token_bucket)
				print(f"current_count_special_token_misplaced_spans: {total_count_special_token_misplaced_spans}")

			doc = json.loads(input_line)

			doc['corenlp_ner_mentions_corefed'] = []  # in each mention: {'word': word, 'label': label, 'span': span}
			doc['corenlp_ner_spans_corefed'] = []
			doc['corenlp_pos_tags_corefed'] = []
			if doc['splitted_text_corefed'] is None:
				doc['corenlp_ner_mentions_corefed'] = None
				doc['corenlp_ner_spans_corefed'] = None
				doc['corenlp_pos_tags_corefed'] = None
			else:
				for sent_id, sent in enumerate(doc['splitted_text_corefed']):
					if sent == doc['splitted_text'][sent_id]:
						doc['corenlp_ner_mentions_corefed'].append(doc['corenlp_ner_mentions'][sent_id])
						doc['corenlp_ner_spans_corefed'].append(doc['corenlp_ner_spans'][sent_id])
						doc['corenlp_pos_tags_corefed'].append(doc['corenlp_pos_tags'][sent_id])
					else:
						sent_ner_mentions, sent_ner_spans, sent_pos_tags, total_count_special_token_misplaced_spans \
							= ner_pos_per_sentence(client, sent, ner_token_bucket, total_count_special_token_misplaced_spans)
						doc['corenlp_ner_mentions_corefed'].append(sent_ner_mentions)
						doc['corenlp_ner_spans_corefed'].append(sent_ner_spans)
						doc['corenlp_pos_tags_corefed'].append(sent_pos_tags)
			doc_id += 1

			# write the annotated entry back into the output fp
			out_line = json.dumps(doc, ensure_ascii=False)
			output_fp.write(out_line + '\n')

			assert doc['splitted_text_corefed'] is None or len(doc['corenlp_ner_mentions_corefed']) == len(doc['splitted_text_corefed'])
			assert doc['splitted_text_corefed'] is None or len(doc['corenlp_ner_spans_corefed']) == len(doc['splitted_text_corefed'])

	print("ner_token_bucket: ")
	print(ner_token_bucket)

	input_fp.close()
	output_fp.close()

	with open(stats_fn, 'w', encoding='utf8') as fp:
		json.dump(ner_token_bucket, fp, ensure_ascii=False)

	print(f"total_count_special_token_misplaced_spans: {total_count_special_token_misplaced_spans}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nerpos_input',
						# default='/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/webhose_data_entries_no_corenlp.jsonl',
						# default='../webhose_data_entries_no_corenlp.jsonl',
						default='../clue_data_entries_no_corenlp.jsonl',
						# default='/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/clue_data_entries_no_corenlp.jsonl',
						type=str)
	parser.add_argument('--nerpos_output',
						default='./clue_data_entries_with_corenlpner_and_postag_%d.json',
						# default='/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/webhose_data_entries_with_corenlpner_and_postag.json',
						# default='./webhose_data_entries_with_corenlpner_and_postag.json',
						type=str)
	parser.add_argument('--nerpos_merged_output',
						default='./clue_data_entries_with_corenlpner_and_postag.json',
						type=str)
	parser.add_argument('--nerpos_stats',
						# default='/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/webhose_corenlpner_stats.json',
						# default='./webhose_corenlpner_stats.json',
						default='./clue_corenlpner_stats_%d.json',
						type=str)
	parser.add_argument('--nerpos_merged_stats',
						default='./clue_corenlpner_stats.json',
						type=str)
	parser.add_argument('--nerpos_num_slices', type=int, default=1)
	parser.add_argument('--nerpos_slice_id', type=int, default=-1)

	parser.add_argument('--coref_input',
						default='../webhose_data_entries_no_corenlp.jsonl',
						# default='/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/webhose_data_entries_no_corenlp.jsonl',
						type=str)
	parser.add_argument('--coref_output',
						default='./webhose_data_entries_with_corenlp_coref_%d.json',
						type=str)
	parser.add_argument('--coref_merged_output',
						default='./webhose_data_entries_with_corenlp_coref.json')
	parser.add_argument('--coref_nerpos_output',
						default='./webhose_data_entries_with_nerpos4coref.json')
	parser.add_argument('--coref_stats',
						default='./webhose_corenlpcoref_stats_%d.json')
	parser.add_argument('--coref_nerpos_stats',
						default='./webhose_nerpos4coref_stats.json')
	parser.add_argument('-m', '--mode', default='nerpos')
	parser.add_argument('--slice_id', type=int, default=0)
	parser.add_argument('--num_slices', type=int, default=12)
	parser.add_argument('--port_id', type=int, default=9000)
	args = parser.parse_args()

	if args.mode == 'nerpos':
		main_ner(args.nerpos_input, args.nerpos_output, args.nerpos_stats, args.nerpos_num_slices, args.nerpos_slice_id,
				 args.port_id)
	elif args.mode == 'nerpos_merge':
		merge_nerpos_results(args.nerpos_output, args.nerpos_stats, args.nerpos_merged_output, args.nerpos_merged_stats,
							 args.nerpos_num_slices)
	elif args.mode == 'coref':
		args.coref_output = args.coref_output % args.slice_id
		args.coref_stats = args.coref_stats % args.slice_id
		assert args.slice_id < args.num_slices
		main_coref(args.coref_input, args.coref_output, args.coref_stats, False, args.slice_id, args.num_slices,
				   args.port_id)
	elif args.mode == 'nerpos4coref':
		merge_coref(args.coref_output, args.coref_merged_output, args.nerpos_output, args.num_slices)
		main_forcoref_ner(args.coref_merged_output, args.coref_nerpos_output, args.coref_nerpos_stats)
	else:
		raise AssertionError
