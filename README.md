# Open Relation Extraction Algorithm for Chinese Entailment Graphs

## Overview
This repository is part of the [Chinese Entailment Graph project]().
Within this repository is the Chinese open relation extraction (ORE) method used for building Chinese entailment graphs. 
This algorithm is based on the dependency labels outputted by [DDParser](https://github.com/baidu/DDParser), 
and provides a comprehensive account for the Chinese linguistic constructions relevant to open relations. 
For a detailed description of our method, please refer to our paper.

## Structure of this Repository
- **preprocess_corpus.py**: code for preprocessing news corpora for ORE;
- **main_ner.py**: code for doing NER the POS tagging with CoreNLP;
- **dudepparse.py**: code for retrieving DDParser dependency paths and extracting open relations based on the results;
- **merge_parsed_results.py**: the extracted open relations are stored in slices, this piece of code glues them together;
- **extract.py**: extended from [this file](https://github.com/baidu/DDParser/blob/master/tools/struct_info/extract.py), 
with outputs reformatted for our ORE method;
- **StanfordCoreNLP.py**: borrowed from [this repository](https://github.com/elisa-aleman/StanfordCoreNLP_Chinese), 
wraps the Stanford CoreNLP package in a python API.

## Requirements
- stanza
- langdetect
- transformers
- ddparser

## How to Run
1. Download Webhose Chinese News Corpus from [here](https://webz.io/free-datasets/chinese-news-articles/), 
place the unzipped folder under the root folder;

2. Preprocess the corpus using `python preprocess_corpus.py --mode webhose`, the set of valid news articles 
will be stored in `./webhose_data_entries_no_corenlp.jsonl`;

3. Run NER and POS tagging on the preprocessing results `./webhose_data_entries_no_corenlp.jsonl` (in 4 slices) with 
`python main_ner.py --mode nerpos --nerpos_num_slices 4 --nerpos_slice_id [0-3]`.
Then glue the 4 slices together using `python main_ner.py --mode nerpos_merge --nerpos_num_slices 4`. Change the input
and output file paths to the ones on your machine. The result file should by default be 
`./clue_data_entries_with_corenlpner_and_postag.json`.

4. Run the open relation extraction algorithm with `python dudepparse.py -s [0-7] --data_entry_filename ./clue_data_entries_with_corenlpner_and_postag.json`. For doing open relation extraction, 
the corpus is by default split into 8 slices; the `-s` flag specifies the slice_id on which to run the algorithm. 
If you have modified this ORE method, but want to re-use the DDParser results elicited from previous rounds, simply set
`--ready_ddp 1` and the dependency paths will be re-used. (But make sure your input hasn't changed from the
last intermediate results! Otherwise the program will crash!)

5. Finally, run `python merge_parsed_results.py` to glue the slices back into a single file, by default 
`webhose_data_entries_with_corenlp_ner_and_parse.json`. 🎉

In our codes we have also provided assorted options for processing other datasets than Webhose corpus. Please read 
through these options or modify on our codes if you are looking to extract open relations for your own corpus!

## Next Steps

For building Chinese Entailment Graphs, this resulting file with the corpus and its extracted open relations
is fed into our [NE_Pipeline](), where the argument type information is added to each relation triple; the typed relation 
triples are then used to construct Chinese entailment graphs with [CEntGraph](), then evaluated in [chinese_entgraph_eval]().

## Cite Us

Coming soon.