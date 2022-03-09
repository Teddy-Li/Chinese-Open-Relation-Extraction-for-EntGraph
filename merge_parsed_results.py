import json
import argparse
import random
import time


# merge entries in dict_2 into dict_1
def merge_dict(dict_1, dict_2):
	for key in dict_2:
		if key not in dict_1:
			dict_1[key] = dict_2[key]
		else:
			dict_1[key] += dict_2[key]
	return


def sort_dict(dct):
	dct_tup = [(key, dct[key]) for key in dct]
	dct_tup.sort(key=lambda tup: tup[1], reverse=True)
	new_dct = {}
	for (key, val) in dct_tup:
		new_dct[key] = val
	return new_dct


def visualize_dict(dct):
	for key in dct:
		print(f"{key}: ")
		print(dct[key])
	return


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default='webhose_data_entries_with_corenlp_ner_and_parse_%d.json', type=str)
	parser.add_argument('-o', '--output', default='webhose_data_entries_with_corenlp_ner_and_parse.json', type=str)
	parser.add_argument('-t', '--toy', default='webhose_data_entries_with_corenlp_ner_and_parse_toy.json', type=str)
	parser.add_argument('-s', '--stats', default='rel_counts_%d.json', type=str)
	parser.add_argument('--total_stats', default='rel_counts_total.json', type=str)
	parser.add_argument('--total_slices', default=8, type=int)

	args = parser.parse_args()

	total_fine_rel_count = 0
	total_coarse_rel_count = 0
	amend_fine_rel_count = 0
	amend_coarse_rel_count = 0
	crossed_rel_count = 0
	amend_crossed_rel_count = 0

	out_fp = open(args.output, 'w', encoding='utf8')
	random.seed(time.time())
	toy_lines = []

	for slice_id in range(args.total_slices):
		print(f"Reading from parsed entry file: {args.input%slice_id}")
		with open(args.input%slice_id, 'r', encoding='utf8') as input_fp:
			for line in input_fp:
				item = json.loads(line)
				assert len(item['splitted_text']) == len(item['fine_rels'])
				for sent_f_rels in item['fine_rels']:
					total_fine_rel_count += len(sent_f_rels)
				for sent_f_rels in item['amend_fine_rels']:
					total_fine_rel_count += len(sent_f_rels)
					amend_fine_rel_count += len(sent_f_rels)
				for sent_c_rels in item['coarse_rels']:
					total_coarse_rel_count += len(sent_c_rels)
				for sent_c_rels in item['amend_coarse_rels']:
					total_coarse_rel_count += len(sent_c_rels)
					amend_coarse_rel_count += len(sent_c_rels)
				for sent_rels in item['crossed_rels']:
					crossed_rel_count += len(sent_rels)
				for sent_rels in item['amend_crossed_rels']:
					amend_crossed_rel_count += len(sent_rels)
				out_fp.write(line.strip('\n')+'\n')
				randnum = random.random()
				if abs(randnum) < 0.0003:
					toy_lines.append(line)

	out_fp.close()

	total_stats = {'fine_stop_word_count_bucket': {},
				  'coarse_stop_word_count_bucket': {},
				  'fine_digit_excluded_count': 0,
				  'coarse_digit_excluded_count': 0,
				  'amend_fine_count': amend_fine_rel_count,
				  'amend_coarse_count': amend_coarse_rel_count,
				  'crossed_count': crossed_rel_count,
				  'amend_crossed_count': amend_crossed_rel_count,
				  'vcmp_bucket': {},
				   'total_fine_rel_count': total_fine_rel_count,
				   'total_coarse_rel_count': total_coarse_rel_count}

	for slice_id in range(args.total_slices):
		print(f"Reading slice stats from stats file: {args.stats%slice_id}")
		with open(args.stats%slice_id, 'r', encoding='utf8') as fp:
			slice_stats = json.load(fp)
			for key in slice_stats:
				assert key in total_stats
			merge_dict(total_stats['fine_stop_word_count_bucket'], slice_stats['fine_stop_word_count_bucket'])
			merge_dict(total_stats['coarse_stop_word_count_bucket'], slice_stats['coarse_stop_word_count_bucket'])
			total_stats['fine_digit_excluded_count'] += slice_stats['fine_digit_excluded_count']
			total_stats['coarse_digit_excluded_count'] += slice_stats['coarse_digit_excluded_count']
			# total_stats['amend_fine_count'] += slice_stats['amend_fine_count']
			# total_stats['amend_coarse_count'] += slice_stats['amend_coarse_count']
			merge_dict(total_stats['vcmp_bucket'], slice_stats['vcmp_bucket'])

	toy_entries = [json.loads(line) for line in toy_lines]
	for item in toy_entries:
		print("Fine: ")
		for sent_id, sent_f_rels in enumerate(item['amend_fine_rels']):
			if len(sent_f_rels) == 0:
				continue
			print(item['splitted_text'][sent_id])
			print(sent_f_rels)
			print("")

		print("Coarse: ")
		for sent_id, sent_c_rels in enumerate(item['amend_coarse_rels']):
			if len(sent_c_rels) == 0:
				continue
			print(item['splitted_text'][sent_id])
			print(sent_c_rels)
			print("")
		print("")
		print("")

	print(f"Dumping to a toy entries file: {args.toy}")
	with open(args.toy, 'w', encoding='utf8') as fp:
		for line in toy_lines:
			fp.write(line+'\n')

	total_stats['fine_stop_word_count_bucket'] = sort_dict(total_stats['fine_stop_word_count_bucket'])
	total_stats['coarse_stop_word_count_bucket'] = sort_dict(total_stats['coarse_stop_word_count_bucket'])
	total_stats['vcmp_bucket'] = sort_dict(total_stats['vcmp_bucket'])

	visualize_dict(total_stats)

	print(f"Dumping to a merged stats file: {args.total_stats}")
	with open(args.total_stats, 'w', encoding='utf8') as fp:
		json.dump(total_stats, fp, indent=4, ensure_ascii=False)

	print("Done.")


if __name__ == '__main__':
	main()
