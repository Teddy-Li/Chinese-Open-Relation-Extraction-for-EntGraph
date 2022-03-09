import json
import os
import copy
import re
import time
import argparse
from StanfordCoreNLP import *
import transformers


delimiters = ['\n', '。', '！', '？', '；', '："', '……', ';', '。"']  # ' ' should appear in extra-long sentence detection rather than regular splitting
global_filtered_sentence_count = 0
global_filtered_entry_count = 0
global_total_sentence_count = 0
global_longest_sentence_len = 0
global_outlengthed_count = 0
FILTER_LEN = 4

english_or_digits = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
					 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
					 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
					 'Y', 'Z']


# removes all the '[UNK]', '#' and ' '
class Normalizer():
	def __init__(self):
		self.num_unks = 0
		self.num_sharps = 0
		self.num_double_colons = 0
		self.warned = False
		self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')

	def tok_norm(self, splitted_text):
		new_texts = []
		for line in splitted_text:
			line = self.tokenizer.tokenize(line)
			if '[UNK]' in line:
				self.num_unks += 1
			while '[UNK]' in line:
				line.remove('[UNK]')
			line = ''.join(line)
			if '#' in line:
				self.num_sharps += 1
			line = line.replace('#', '')
			if '::' in line:
				self.num_double_colons += 1
				# print(":: in line!")
			line = line.replace('::', '')
			assert '[UNK]' not in line
			assert '#' not in line
			if len(line) > 0:
				new_texts.append(line)
			else:
				if not self.warned:
					print(f"Warning! Different from last instantiation: lines with 0 length after normalization are discarded!")
					self.warned = True
		return new_texts

	def _print(self):
		print(f"Number of [UNK]s: {self.num_unks}")
		print(f"Number of #: {self.num_sharps}")
		print(f"Number of :: : {self.num_double_colons}")


def split_long_sentences(string, interval):
	global global_filtered_sentence_count
	res_list = []
	offset = 0
	while offset < len(string):
		if offset+interval >= len(string):
			res_list.append(string[offset:])
			offset += interval
			break
		closest = -1
		for i in range(interval, 0, -1):
			if offset+i >= len(string):
				continue
			c = string[offset+i]
			if c in ['，', '；', '、', ' ']:
				if c == '；':
					print(c)
				closest = offset+i
				res_list.append(string[offset:closest])
				offset = closest+1
				break
		if closest == -1 and offset+interval < len(string):
			closest = offset+interval
			res_list.append(string[offset:closest])
			offset = closest

	final_res_list = []
	for item in res_list:
		chinese_character_list = re.findall(r'[\u4e00-\u9fff]+', item)
		chinese_character_list = ''.join(chinese_character_list)
		if len(chinese_character_list) > FILTER_LEN:
			final_res_list.append(item)
		else:
			global_filtered_sentence_count += 1
	return final_res_list


def split_str(splitted, split_mapping=None, DEBUG=False):
	global global_filtered_sentence_count
	global global_longest_sentence_len
	global global_outlengthed_count
	global global_total_sentence_count
	idxs = [i for i in range(len(splitted))] if split_mapping is None else split_mapping
	new_splitted = []
	new_idxs = []

	for d in delimiters:
		for s, idx in zip(splitted, idxs):
			new_s = s.split(d)
			for piece_id, piece in enumerate(new_s):
				if piece_id != len(new_s)-1 and d not in [' ', '\n', '（', '："']:
					new_s[piece_id] = piece+d
				if piece_id > 0 and d in ['（', '："']:
					new_s[piece_id] = d[-1]+piece
			#if d in ['……'] and len(new_s) > 1:
			#	print(new_s)
			new_splitted += new_s
			new_idxs += [idx for k in range(len(new_s))]
		splitted = copy.deepcopy(new_splitted)
		idxs = new_idxs
		assert len(splitted) == len(idxs)
		new_splitted = []
		new_idxs = []

	for s, idx in zip(splitted, idxs):
		chinese_character_list = re.findall(r'[\u4e00-\u9fff]+', s)
		chinese_character_list = ''.join(chinese_character_list)
		if FILTER_LEN < len(chinese_character_list) <= 430 and len(s) < 500:
			new_splitted.append(s)
			new_idxs.append(idx)

		elif len(chinese_character_list) > 430 or len(s) >= 500:
			if DEBUG:
				print("Accounting for overlengthed sentences!")
			further_slitted_s = split_long_sentences(s, 430)
			global_outlengthed_count += 1
			new_splitted += further_slitted_s
			new_idxs += [idx for k in range(len(further_slitted_s))]
		else:
			global_filtered_sentence_count += 1

	splitted = new_splitted
	idxs = new_idxs
	assert len(splitted) == len(idxs)
	new_splitted = []
	new_idxs = []

	for item, idx in zip(splitted, idxs):
		new_splitted.append(item.strip())
		new_idxs.append(idx)

	for s in new_splitted:
		if len(s) > global_longest_sentence_len:
			global_longest_sentence_len = len(s)

	global_total_sentence_count += len(new_splitted)
	assert len(new_splitted) == len(new_idxs)
	return new_splitted, new_idxs


def main_webhose(output_name, excluded_name, use_corenlp):
	global global_filtered_entry_count
	root_dir = '/Users/mask/Files/Potential Corpus/WebHoses_Chinese_News_Articles/630_webhose-2016-10_20170904084325'
	filename_list = os.listdir(root_dir)
	normalizer = Normalizer()

	output_fp = open(output_name, 'w', encoding='utf8')
	excluded_fp = open(excluded_name, 'w', encoding='utf8')

	StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
	with CoreNLPClient(
				annotators=['tokenize', 'ssplit'],
				properties=StanfordCoreNLP_chinese_properties,
				threads=16,
				memory='24G',
				endpoint='http://localhost:9001',
				be_quiet=True) as client:
		st = time.time()
		for idx, filename in enumerate(filename_list):
			if idx % 1000 == 0 and idx > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) / 3600
				dur_m = (int(dur) % 3600) / 60
				dur_s = int(dur) % 60
				print(idx, 'time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
				print("Current global_longest_sentence_len: ", global_longest_sentence_len)
			with open(os.path.join(root_dir, filename), 'r', encoding='utf8') as fp:
				data_entry = json.load(fp)
			text = data_entry['text']

			splitted = []
			if use_corenlp > 0:
				ann = client.annotate(text)
				for sentence in ann.sentence:
					sent_str = ''
					# join([token.word for token in sentence.token])
					for tok_id, token in enumerate(sentence.token):
						if tok_id > 0 and sentence.token[tok_id-1].endChar != token.beginChar:
							num_spaces = token.beginChar - sentence.token[tok_id-1].endChar
							assert num_spaces > 0
							for i in range(num_spaces):
								sent_str += ' '
						sent_str += token.word

					splitted.append(sent_str)
			else:
				splitted = [text]
			splitted_text, _ = split_str(splitted)
			splitted_text = normalizer.tok_norm(splitted_text)  # removes UNKS, #s and spaces

			if len(splitted_text) == 0:
				global_filtered_entry_count += 1
				out_line = json.dumps(data_entry, ensure_ascii=False)
				excluded_fp.write(out_line+'\n')
				continue
			data_entry['splitted_text'] = splitted_text
			out_line = json.dumps(data_entry, ensure_ascii=False)
			output_fp.write(out_line+'\n')

	print("Total number of sentences filtered out via length criterion: ", global_filtered_sentence_count)
	print("Total number of entries filtered out due to no lengthly enough sentences: ", global_filtered_entry_count)
	print("Total number of entries exceeding length limit and have to be splitted in finer granularity: ", global_outlengthed_count)
	print("global_longest_sentence_len: ", global_longest_sentence_len)
	print("global_total_sentence_count: ", global_total_sentence_count)

	output_fp.close()
	excluded_fp.close()

	print("Saved!")


def main_clue(input_name, output_name, excluded_name, use_corenlp):
	global global_filtered_entry_count
	normalizer = Normalizer()

	input_fp = open(input_name, 'r', encoding='utf8')
	output_fp = open(output_name, 'w', encoding='utf8')
	excluded_fp = open(excluded_name, 'w', encoding='utf8')

	StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
	with CoreNLPClient(
			annotators=['tokenize', 'ssplit'],
			properties=StanfordCoreNLP_chinese_properties,
			threads=16,
			memory='24G',
			endpoint='http://localhost:9001',
			be_quiet=True) as client:
		st = time.time()
		for idx, data_line in enumerate(input_fp):
			if idx % 3000 == 0 and idx > 0:
				ct = time.time()
				dur = ct - st
				dur_h = int(dur) / 3600
				dur_m = (int(dur) % 3600) / 60
				dur_s = int(dur) % 60
				print(idx, 'time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
				if idx % 30000 == 0:
					print("Current global_longest_sentence_len: ", global_longest_sentence_len)
					print("Total number of sentences filtered out via length criterion: ", global_filtered_sentence_count)
					print("Total number of entries filtered out due to no lengthly enough sentences: ",
						  global_filtered_entry_count)
					print("Total number of entries exceeding length limit and have to be splitted in finer granularity: ",
						  global_outlengthed_count)
					print("global_total_sentence_count: ", global_total_sentence_count)

			data_entry = json.loads(data_line)
			text = data_entry['content']

			splitted = []
			if use_corenlp > 0:
				ann = client.annotate(text)
				for sentence in ann.sentence:
					sent_str = ''
					# join([token.word for token in sentence.token])
					for tok_id, token in enumerate(sentence.token):
						if tok_id > 0 and sentence.token[tok_id - 1].endChar != token.beginChar:
							num_spaces = token.beginChar - sentence.token[tok_id - 1].endChar
							assert num_spaces > 0
							for i in range(num_spaces):
								sent_str += ' '
						sent_str += token.word

					splitted.append(sent_str)
			else:
				splitted = [text]
			splitted_text, _ = split_str(splitted)
			splitted_text = normalizer.tok_norm(splitted_text)  # removes UNKS, #s and spaces

			if len(splitted_text) == 0:
				global_filtered_entry_count += 1
				out_line = json.dumps(data_entry, ensure_ascii=False)
				excluded_fp.write(out_line + '\n')
				continue
			data_entry['splitted_text'] = splitted_text
			out_line = json.dumps(data_entry, ensure_ascii=False)
			output_fp.write(out_line + '\n')

	print("Total number of sentences filtered out via length criterion: ", global_filtered_sentence_count)
	print("Total number of entries filtered out due to no lengthly enough sentences: ", global_filtered_entry_count)
	print("Total number of entries exceeding length limit and have to be splitted in finer granularity: ",
		  global_outlengthed_count)
	print("global_longest_sentence_len: ", global_longest_sentence_len)
	print("global_total_sentence_count: ", global_total_sentence_count)

	input_fp.close()
	output_fp.close()
	excluded_fp.close()

	print("Saved!")


def main_newsspike(input_name, output_name, excluded_name):
	global global_filtered_entry_count

	rechunked_articles_count = 0

	input_fp = open(input_name, 'r', encoding='utf8')
	output_fp = open(output_name, 'w', encoding='utf8')
	excluded_fp = open(excluded_name, 'w', encoding='utf8')

	normalizer = Normalizer()

	st = time.time()
	for idx, data_line in enumerate(input_fp):
		if idx % 1000 == 0 and idx > 0:
			ct = time.time()
			dur = ct - st
			dur_h = int(dur) / 3600
			dur_m = (int(dur) % 3600) / 60
			dur_s = int(dur) % 60
			print(idx, 'time lapsed: %d hours %d minutes %d seconds' % (dur_h, dur_m, dur_s))
			normalizer._print()
			print(f"Current global_longest_sentence_len: {global_longest_sentence_len}; current re-chunked articles count: {rechunked_articles_count}")
		data_entry = json.loads(data_line)
		splitted = data_entry['splitted_text']
		splitted_text, split_mapping = split_str(splitted, data_entry['split_mapping'])
		if len(splitted_text) > len(splitted):
			# assert len(splitted_text) > len(splitted)
			rechunked_articles_count += 1
		splitted_text = normalizer.tok_norm(splitted_text)  # removes UNKS, #s and spaces

		if len(splitted_text) == 0:
			global_filtered_entry_count += 1
			out_line = json.dumps(data_entry, ensure_ascii=False)
			excluded_fp.write(out_line+'\n')
			continue
		assert len(splitted_text) == len(split_mapping)
		data_entry['splitted_text'] = splitted_text
		data_entry['split_mapping'] = split_mapping
		out_line = json.dumps(data_entry, ensure_ascii=False)
		output_fp.write(out_line+'\n')

	print("Total number of sentences filtered out via length criterion: ", global_filtered_sentence_count)
	print("Total number of entries filtered out due to no lengthly enough sentences: ", global_filtered_entry_count)
	print("Total number of entries exceeding length limit and have to be splitted in finer granularity: ", global_outlengthed_count)
	print("global_longest_sentence_len: ", global_longest_sentence_len)
	print("global_total_sentence_count: ", global_total_sentence_count)

	input_fp.close()
	output_fp.close()
	excluded_fp.close()

	print("Saved!")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--output', type=str, default='./webhose_data_entries_no_corenlp.jsonl')
	parser.add_argument('--corenlp', type=int, default=0, help='whether or not to use corenlp for sentence splitting.')
	# parser.add_argument('--excluded', type=str, default='./webhose_data_entries_no_corenlp_excluded.jsonl')
	parser.add_argument('--mode', type=str, default='webhose', help='[webhose/clue/newsspike]')

	args = parser.parse_args()

	if args.mode == 'webhose':
		output_name = './webhose_data_entries_no_corenlp.jsonl'
		excluded_name = './webhose_data_entries_no_corenlp_excluded.jsonl'
		main_webhose(output_name, excluded_name, args.corenlp)
	elif args.mode == 'clue':
		input_name = '/Users/mask/Files/Potential Corpus/nlp_chinese_corpus/new2016zh/news2016zh_train.json'
		output_name = './clue_data_entries_no_corenlp.jsonl'
		excluded_name = './clue_data_entries_no_corenlp_excluded.jsonl'
		main_clue(input_name, output_name, excluded_name, args.corenlp)
	elif args.mode == 'newsspike':
		input_name = '/Users/mask/Files/relational-implication-dataset_levy_holts/newsspike_data_entries.jsonl'
		output_name = './newsspike_data_entries_no_corenlp.jsonl'
		excluded_name = './newsspike_data_entries_no_corenlp_excluded.jsonl'
		main_newsspike(input_name, output_name, excluded_name)