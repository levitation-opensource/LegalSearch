
#
# Author: Roland Pihlakas, 2017
#
# roland@simplify.ee
#

print('LegalSearchText 31.01.2017')


import hashlib
import logging
import numpy
import random
import os
import string
import sys
import warnings


# http://stackoverflow.com/questions/1943747/python-logging-before-you-run-logging-basicconfig
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim     # import gensim only after disabling gensim warnings


# cores = multiprocessing.cpu_count()
# assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"   # TODO: multicore support



# process arguments

argv = sys.argv
arg_index = 1   # NB! for Python the first real argument has index 1



# display help?

if (
    len(argv) == 1
    or ((len(argv) > arg_index) and (argv[arg_index].lower() == "help"))
):
    print('Usage:')
    print('LegalSearchText.exe help')
    print('LegalSearchText.exe learn [level: text|line (text)] [language: eng|est|none (eng)] [num_dims: (152/252)]')
    print('LegalSearchText.exe [level: text|line (text)] [language: eng|est|engest|esteng|none (eng)] [scope: ee|eu|eeeu|folder:... (ee)] [num_results: (20)] [num_dims: (152/252)] [notlike #1 #3 #8 ...] query words ... -negative -words ...')    
    print('    Notes:')
    print('    * language parameter engest means that the query is in English and results should be shown from corresponding Estonian corpus')
    print('    * language parameter esteng means that the query is in Estonian and results should be shown from corresponding English corpus')
    print('    * scope parameter eeeu means that both Estonian and EU laws are used for search')
    print('')
    print('Copyrights:')
    print('GenSim, GNU LGPLv2.1 licence. https://radimrehurek.com/gensim/')
    print('Miniconda, BSD 3-clause licence. https://conda.io/miniconda')

    sys.exit()



# init search index?

init = (len(argv) > arg_index) and (argv[arg_index].lower() == 'learn')

if (init):
    arg_index = arg_index + 1



# use line search or document search?

use_line_search = False           # default to text level  

if (len(argv) > arg_index): 
    if (argv[arg_index].lower() == "text"):
        use_line_search = False
        arg_index = arg_index + 1
    elif (argv[arg_index].lower() == "line"):
        use_line_search = True
        arg_index = arg_index + 1



# select search language

search_language = ""
result_language = ""
do_not_detect_tags = False

if (len(argv) > arg_index): 
    if (argv[arg_index].lower() == "eng" or argv[arg_index].lower() == "engeng"):
        search_language = "eng" 
        result_language = "eng"
    elif (argv[arg_index].lower() == "est" or argv[arg_index].lower() == "estest"):
        search_language = "est" 
        result_language = "est"
    elif (argv[arg_index].lower() == "engest"):
        search_language = "eng" 
        result_language = "est"
    elif (argv[arg_index].lower() == "esteng"):
        search_language = "est" 
        result_language = "eng"
    elif (argv[arg_index].lower() == "none"):
        search_language = "none" 
        result_language = "none"
        do_not_detect_tags = True

if (search_language != ""):
    arg_index = arg_index + 1



# choose corpus directories

search_corpus_name = ""
search_corpus_dirs = []
using_custom_corpus = False

if (len(argv) > arg_index): 
    if (argv[arg_index].lower() == "eeeu" or argv[arg_index].lower() == "euee"):
        search_corpus_name = "eeeu"
        search_corpus_dirs = ['et-en/', 'en-et_t/', 'en-et_u/']    # Estonian and EU laws
    elif (argv[arg_index].lower() == "eu"):
        search_corpus_name = "eu"
        search_corpus_dirs = ['en-et_t/', 'en-et_u/']              # only EU laws
    elif (argv[arg_index].lower() == "ee"):
        search_corpus_name = "ee"
        search_corpus_dirs = ['et-en/']                            # only Estonian laws
    elif (argv[arg_index].lower()[:7] == "folder:"):
        using_custom_corpus = True
        custom_folder = os.path.normpath(argv[arg_index][7:])
        
        custom_folder_id = hashlib.md5(custom_folder.encode('utf-8')).hexdigest()

        search_corpus_name = "custom_" + custom_folder_id
        search_corpus_dirs = [custom_folder + '/']                 # custom folder

if (len(search_corpus_dirs) == 0):
    search_corpus_name = "ee"
    search_corpus_dirs = ['et-en/']    # default to only Estonian laws
else:
    arg_index = arg_index + 1



# choose default language

if (search_language == ""):
    if (using_custom_corpus):
        search_language = "none"    # default to non-parallel corpus when using custom corpus
        result_language = "none"
        do_not_detect_tags = True
    else:
        search_language = "eng"     # default to English language
        result_language = "eng"



# decide full corpus directories list

if do_not_detect_tags and not using_custom_corpus:
    print("If 'none' mode is selected for language then custom corpus must be used")
    sys.exit()

all_search_corpus_dirs = search_corpus_dirs if using_custom_corpus else ['et-en/', 'en-et_t/', 'en-et_u/']



# select corpus scope

use_common_corpus = True    # index always the whole corpus?

index_corpus_name = "full" if use_common_corpus else search_corpus_name  
index_corpus_dirs = ['et-en/', 'en-et_t/', 'en-et_u/']



# select number of results

num_results = 20

if ((len(argv) > arg_index) and (argv[arg_index].isnumeric())):
    num_results = int(argv[arg_index])
    arg_index = arg_index + 1



# select number of dimensions

num_dims = 250 if use_line_search else 150  
num_dims = int((num_dims + 3) / 4) * 4  # round up to next multiple of 4 for improved performance and suppress any warning messages since defaults behaviour is used anyway

if ((len(argv) > arg_index) and (argv[arg_index].isnumeric())):
    num_dims = int(argv[arg_index])
    arg_index = arg_index + 1

num_dims1 = num_dims
num_dims = int((num_dims + 3) / 4) * 4  # round up to next multiple of 4 for improved performance



# get previous results to exclude and avoid similarity to this time

notlike = []
notlike_tags = []
if (len(argv) > arg_index and argv[arg_index].lower() == 'notlike'):
    arg_index = arg_index + 1
    while (len(argv) >= arg_index and argv[arg_index].isnumeric()):
        notlike.append(int(argv[arg_index]))
        arg_index = arg_index + 1



# define word filters
        
# TODO!!!: use NLTK
forbidden_words = ['&sect;', 'a', 'an', 'and', 'at', 'be', 'because', 'can', 'either', 'did', 'does', \
                   'in', 'is', 'of', 'neither', 'nor', 'not', 'on', 'or', 'the', 'was', 'xor']

# https://en.wikipedia.org/wiki/Stemming
forbidden_sufixes = ['.', 'ed', 'ing', 'ly', "n't", 's', "s'", "'s"]    # translate(translator) does not remove dots after numerics

def filter_words(line):
    for word in line:
        if word not in forbidden_words:

            for sufix in forbidden_sufixes:
                sufix_length = len(sufix)
                if (word[-sufix_length:] == sufix):
                    word = word[:-sufix_length]

            word = word.strip()

            if (word != '' and not word.isnumeric()):   # TODO!: implement min word length?
                yield word

#/ def filter_words(line):



# TODO:
# bigram_transformer = gensim.models.Phrases(sentences)
# model = gensim.models.Word2Vec(bigram_transformer[sentences], min_count=1, size=100, workers=4)

# TODO!!! use stemmer from https://github.com/arthur-flam/search-tfidf-word2vec-poc
# TODO!!! use Phraser?
# TODO!!! converts &html; strings to normal characters
# TODO!!! remove punctuation

translator = str.maketrans('', '', string.punctuation)  # remove all punctuation



# get search query words, positive and negative

positive_words = []
negative_words = []

while (len(argv) > arg_index):

    arg = argv[arg_index].strip().lower()

    if (arg.translate(translator) == ''):   # includes arg = '-'
        pass
    elif (arg[:1] == '-'):
        negative_words.append(arg[1:].translate(translator).strip())
    else:
        positive_words.append(arg.translate(translator).strip())

    arg_index = arg_index + 1


negative_words = [x for x in filter_words(negative_words)]
positive_words = [x for x in filter_words(positive_words)]


print('positive query words: ' + str(positive_words))
print('negative query words: ' + str(negative_words))
print('negative query result indexes from previous query: ' + str(notlike))



# init paths

curdir = os.getcwd()
basedatadir = curdir

all_datadirs = [x for x in all_search_corpus_dirs if os.path.exists(os.path.join(basedatadir, x))]
datadirs = [x for x in search_corpus_dirs if os.path.exists(os.path.join(basedatadir, x))]

if (init and len(all_datadirs) != len(all_search_corpus_dirs)):
    print('Some texts folders not found')
    sys.exit()

if (len(datadirs) != len(search_corpus_dirs)):
    print('Some texts folders not found')
    sys.exit()



# select model file based on search language and corpus

model_file = os.path.join(curdir, 'legalsearch_' + ('line' if use_line_search else 'text') + '_' + search_language + '_' + index_corpus_name + '_' + str(num_dims) + 'd.dat')



# choose line tags for file processing

eng_tag_start = '<inglise>'
eng_tag_end = '</inglise>'

est_tag_start = '<eesti>'
est_tag_end = '</eesti>'

if (search_language == "eng"):
    search_tag_start = eng_tag_start    
    search_tag_end = eng_tag_end
elif (search_language == "est"):
    search_tag_start = est_tag_start    
    search_tag_end = est_tag_end

if (result_language == "eng"):
    result_tag_start = eng_tag_start    
    result_tag_end = eng_tag_end
elif (result_language == "est"):
    result_tag_start = est_tag_start    
    result_tag_end = est_tag_end


last_results_file_name = os.path.join(curdir, 'legalsearchtext_last_results_' + search_language + '_' + search_corpus_name + '.dat')



# init random number seed for deterministic results     # TODO

seed = 0


    
# https://rare-technologies.com/word2vec-tutorial/
class ReadSentences(object):

    def __init__(self, local_datadirs):

        random.seed(seed) # NB! need deterministic shuffle
        self.files = []

        for datadir in local_datadirs:
            fulldirpath = os.path.join(basedatadir, datadir)
            if (os.path.exists(fulldirpath)):
                for file in os.listdir(fulldirpath):

                    fullfilename = os.path.join(fulldirpath, file)

                    filesize = os.path.getsize(fullfilename)
                    if filesize == 0:   # ignore zero-size files
                        continue


                    corpus = datadir

                    # first entry in tuple is a tag but if will be used in displaying the results so it needs to be properly formatted too
                    self.files.append([corpus, os.path.join(datadir, file), fullfilename])

        #/ for datadir in local_datadirs:

        # shuffling is needed for improved learning
        # TODO: but order files must be same for all loops?
        # random.shuffle(self.files)   
        self.max_num_files = -1 #10

    #/ def __init__(self, datadirs):
 

    def __iter__(self):

        random.seed(seed) # NB! need deterministic shuffle

        num_files = 0

        for fname_kvp in self.files:

            num_files = num_files + 1
            if (self.max_num_files != -1 and num_files > self.max_num_files):
                break

            (corpus, tag_fname, fullfilename) = fname_kvp

            num_lines = 0
            lang_line_no = 0

            lines = [x for x in enumerate(open(fullfilename, 'r'))]
            if (len(lines) == 0):   # empty lines?
                continue

            lines2 = []


            line2 = []  # used for text search


            for line_kvp in lines:

                (line_no, line) = line_kvp
                line = line.strip() 

                if (len(line) == 0):         # empty line
                    continue

                    
                if (do_not_detect_tags
                    or (
                        line[:len(search_tag_start)].lower() == search_tag_start 
                        and line[-len(search_tag_end):].lower() == search_tag_end)):  

                    num_lines = num_lines + 1
                    lang_line_no = lang_line_no + 1

                    # TODO: do not lowercase abbreviations
                    line = line if do_not_detect_tags else line[len(search_tag_start):-len(search_tag_end)]
                    line = line.lower().translate(translator).split()    # split() also strips()
                    line = [x for x in filter_words(line)]

                    if (use_line_search):    # line search

                        lines2.append([line_no, lang_line_no, line])

                    else:   # document search

                        for word in filter_words(line):
                            line2.append(word)

                #/ if (line[:len(tag_start)].lower() == tag_start and line[-len(tag_end):].lower() == tag_end):


            #/ for line_kvp in lines:

            if not use_line_search:
                lines2 = [[1, 0, line2]]



            random.shuffle(lines2)  # shuffling is needed for improved learning

            num_lines = 0

            for line_tuple in lines2:

                (line_no, lang_line_no, line) = line_tuple

                num_lines = num_lines + 1

                tag = corpus + ':' + tag_fname + ':' + str(line_no) + ':' + str(lang_line_no)
                tag2 = tag_fname

                # https://medium.com/@rajkumar021989/one-additional-point-in-labeledlinesentence-function-ccff6fcab3c9#.b9c9jrfib
                if use_line_search:
                    yield gensim.models.doc2vec.TaggedDocument(line, [tag, tag2])
                else:
                    yield gensim.models.doc2vec.TaggedDocument(line, [tag])

                # TODO
                # >>> bigram_phraser = gensim.models.Phrases([line], min_count=2, threshold=2, delimiter=b' ')
                # >>> bigram_phraser = gensim.models.phrases.Phraser([line])
                # >>> yield bigram_phraser[line]

            #/ for line_tuple in lines2:

    #/ def __iter__(self):

#/ class ReadSentences(object):


# debug helper
def print_vocab(model):
    count = 0
    for x in model.wv.index2word:
        print(x)
        print(model.wv.vocab[x])
        count = count + 1
        if (count == 100):  # lets not go crazy
            break

#/ def print_vocab(model):



if (init or not os.path.exists(model_file)):
    

    print('Studying texts and generating associations with ' + str(num_dims) + ' semantic dimensions' + ('. (Number of dimensions adjusted for performance).' if num_dims1 != num_dims else ''))


    # hs = if 1 (default), hierarchical sampling will be used for model training (else set to 0).
    # negative = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
    # sample = threshold for configuring which higher-frequency words are randomly downsampled;
    # default is 0 (off), useful value is 1e-5.

    # iter=1 : prevent internal iterations
    # Let's say you want to invoke the multiple passes yourself, by calling `train()` for each pass. (The main reason I can think of to do this would be to do interim reporting/model-evaluation after each pass.) In such a case you'd probably want to (1) set `iter` to 1 so the internal repeats are prevented; and (2) manage `alpha` yourself, so that it still decays gradually over all passes from its max to its min. (You certainly *don't* want a saw-tooth pattern, where each call to `train()` sends it from 0.025 to 0.001, which you'd get if you left `alpha` and `min_alpha` at their defaults.) 
    # The most simple way to manage alpha yourself is to set `alpha` and `min_alpha` to the same initial fixed value, so a full pass uses that value, then decrement them both before each next pass, in fixed-size steps down to the desired final-pass value. So, that approach has been shown in a number of published examples, including the doc2vec-IMDB.ipynb notebook bundled with gensim. (More sophisticated smoother-decay approaches are if course also possible.)
    # https://groups.google.com/forum/#!msg/gensim/7eiwqfhAbhs/NwoTI-OFHwAJ
    # TODO!!! use normal iteration in non-debug mode
    #    model = gensim.models.Doc2Vec(iter=1, min_count=10, size=num_dims, workers=16, sample=1e-5, max_vocab_size=10000, hs=1)  # an empty model, no training yet
    #    model = gensim.models.Doc2Vec(seed=0, dm=0, min_count=10, size=num_dims, workers=16, sample=1e-5, max_vocab_size=10000, hs=1)  # an empty model, no training yet
    #    model = gensim.models.Doc2Vec(seed=0, dm=0, min_count=10, size=num_dims, workers=1, sample=1e-5, max_vocab_size=10000, hs=1)  # an empty model, no training yet
    model = gensim.models.Doc2Vec(seed=0, min_count=10, size=num_dims, workers=1, sample=1e-5, max_vocab_size=10000, hs=1)  # an empty model, no training yet

    sentences = ReadSentences(all_datadirs if use_common_corpus else datadirs) # a memory-friendly iterator
    # >>> bigram_transformer = gensim.models.phrases.Phraser(sentences)



    # >>> model.build_vocab(bigram_transformer[sentences])
    # >>> model.build_vocab(sentences)

    print('Scanning files for vocabulary')
    print('... this may take a few minutes, go eat some apples')
    model.scan_vocab(sentences)



    print('Weighing words')

    # calculate what a specific value of `min_count' will do to the vocabulary-size and memory requirements, and what different values of `sample` will do to the total corpus size (and thus rough training time)
    # >>> print(model.scale_vocab(dry_run=True))    # TODO?

    model.scale_vocab()

    # Apply vocabulary settings for min_count (discarding less-frequent words) and sample (controlling the downsampling of more-frequent words).
    # >>> model.scale_vocab(sample=1000)    # TODO



    print('Finalising vocabulary')
    model.finalize_vocab()



    # >>> print_vocab(model) # TODO: remove common words


    # The model is better trained if in each training epoch, the sequence of sentences fed to the model is randomized
    # This is important: missing out on this steps gives you really shitty results
    # http://linanqiu.github.io/2015/10/07/word2vec-sentiment/

    print('Training associations')
    print('... this will certainly take a few more minutes, do you have more apples?')
    # for epoch in range(10):       # TODO display progress TODO: tune the parameter
    #   model.train(sentences)
    #   print('Training epoch ' + str(epoch))
    #   seed = epoch + 1
    #
    # https://groups.google.com/forum/#!topic/gensim/sbJBb7sEBVE
    # https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1#.5w3ablgqb
    #   model.alpha -= 0.002  # decrease the learning rate
    #   model.min_alpha = model.alpha  # fix the learning rate, no decay

    model.train(sentences)



    # If you're finished training a model (=no more updates, only querying), you can do
    # to trim unneeded model memory = use (much) less RAM.
    #
    # model.init_sims(replace=True)
    #
    # comment-out:
    # causes "AttributeError: 'Doc2Vec' object has no attribute 'syn1'"
    # I could handle the error i two ways:
    # setting the parameter hs=0 by initializing the model or
    # not calling model.init_sims()
    # ...
    # I just figured out. it also works with init_sims(replace=False).
    # init_sims(replace=True) seems do delete the attribute syn1 from the model which is used by infer_vector
    # model.infer_vectors trains the new documents with the neural weights of the actual model (https://github.com/piskvorky/gensim/blob/develop/gensim/models/doc2vec.py#L684).
    # As model.init_sims(replace=True) is deleting them for memory save reasons, the method model.infer_vectors can not work. It's the same reason why model.train is not working after model.init_sims(replace=True).
    # Thanks for your report. Yes, inference works almost exactly like training, so a model with training-state discarded won't be able to reasonably infer either. The comment for init_sims(replace=True) could be a bit clearer.
    # TODO!!! choose
    # https://github.com/RaRe-Technologies/gensim/issues/483



    print('Pruning the associations')
    model.init_sims(replace=False)


    print('Total number of texts: ' + str(len(model.docvecs)))


    print('Saving associations. (This will take some more time).')
    model.save(model_file)


    print('Done. You can issue search queries now')
    #sys.exit()



    # reload the model for improved results immediately after indexing. Else the first time search gives bad results for some reason. TODO: why?
    numpy.random.seed(0)
    random.seed(0)

    if (len(positive_words) > 0 or len(negative_words) > 0):
        model = gensim.models.Word2Vec.load(model_file)

    # If you need such determinism, you should be able to force it by explicitly resetting the model.random property to a freshly- and deterministically seeded RandomState instance
    # https://github.com/RaRe-Technologies/gensim/issues/447
    model.random.seed(0)



else:  # if (init or not os.path.exists(model_file)):

    
    # TODO!!! why are result scores so much higher when seeds are zeroed here?


    numpy.random.seed(0)
    random.seed(0)


    print('Loading associations with ' + str(num_dims) + ' semantic dimensions.' + (' (Number of dimensions adjusted for performance).' if num_dims1 != num_dims else '') + ' (First time loading may take some time).')
    model = gensim.models.Word2Vec.load(model_file)


    # If you need such determinism, you should be able to force it by explicitly resetting the model.random property to a freshly- and deterministically seeded RandomState instance
    # https://github.com/RaRe-Technologies/gensim/issues/447
    model.random.seed(0)


    print('Total number of texts: ' + str(len(model.docvecs)))


#/ if (init or not os.path.exists(model_file)):



if (len(positive_words) > 0 or len(negative_words) > 0):
    print('Inferring search query semantics')



# positive and negative words

inferred_positive_vector = numpy.asarray([])
inferred_negative_vector = numpy.asarray([])
has_negative_query = False

if (len(positive_words) > 0):
    inferred_positive_vector = model.infer_vector(positive_words)

if (len(negative_words) > 0):
    inferred_negative_vector = model.infer_vector(negative_words)
    has_negative_query = True



# previous results to exclude and avoid similars

if (len(notlike) > 0):


    if (1 == 0):    # TODO: excluding previos results without saving them?

        if (inferred_positive_vector.size > 0 and inferred_negative_vector.size > 0):
            inferred_vector1 = inferred_positive_vector - inferred_negative_vector
        elif (inferred_positive_vector.size > 0):
            inferred_vector1 = inferred_positive_vector
        elif (inferred_negative_vector.size > 0):
            inferred_vector1 = -inferred_negative_vector
        else:
            print('Please enter a query')
            sys.exit()


        print('Searching for excluded matches')
        similars = model.docvecs.most_similar([inferred_vector1], topn=num_results)
        # NB! .docvecs: https://groups.google.com/forum/#!topic/gensim/sbJBb7sEBVE
        # model.most_similar: Find similar words
        # model.doc2vec.most_similar: Find similar sentences or documents

    else:

        #with open(last_results_file_name, 'r') as infile:
        if (os.path.exists(last_results_file_name)):
            lastresults = [line.strip() for line in open(last_results_file_name, 'r')]
        else:
            print('Notlike command: previous search result not available')
            sys.exit()


    print('Inferring semantics of excluded matches')
    for notlike_index in notlike:
        
        tag = lastresults[notlike_index - 1]    # NB! -1 since the gui indexing is 1-based

        notlike_tags.append(tag)

        if not tag in model.docvecs:
            print('Excluded match not found: ' + str(notlike_index))
            continue

        negative_similar_vector = model.docvecs[tag]
        # NB! .docvecs: https://groups.google.com/forum/#!topic/gensim/sbJBb7sEBVE


        if (inferred_negative_vector.size > 0):
            inferred_negative_vector = inferred_negative_vector + negative_similar_vector
        else:
            inferred_negative_vector = negative_similar_vector



    inferred_negative_vector = inferred_negative_vector / (len(notlike) + (1 if has_negative_query else 0))

#/ if (len(notlike) > 0):



if (inferred_positive_vector.size > 0 and inferred_negative_vector.size > 0):
    inferred_vector = inferred_positive_vector - inferred_negative_vector
elif (inferred_positive_vector.size > 0):
    inferred_vector = inferred_positive_vector
elif (inferred_negative_vector.size > 0):
    inferred_vector = -inferred_negative_vector
else:
    print('Please enter a search query')
    sys.exit()



displayed_tags = []

previously_found_tags = []

number_of_previously_generated_similars = 0
result_index = 0



print('Searching for matches')


prev_result_count = 0

while True:     # in case only EE laws are looked at, the training corpus still contains both EE and EU laws, so we need to filter out any EU laws
                # TODO: do the filtering using some additional pseudo-keywords?

    # number_of_previously_generated_similars will be skipped
    topn = num_results + len(notlike) + number_of_previously_generated_similars

    # similars = model.docvecs.most_similar([inferred_vector], topn=(num_results + len(notlike)))

    # need separate cases since most_similar() method does not accept positive/negative arguments with empty lists
    if (inferred_positive_vector.size > 0 and inferred_negative_vector.size > 0):
        similars = model.docvecs.most_similar(positive=[inferred_positive_vector], negative=[inferred_negative_vector], topn=topn)
    elif (inferred_positive_vector.size > 0):
        similars = model.docvecs.most_similar(positive=[inferred_positive_vector], topn=topn)
    elif (inferred_negative_vector.size > 0):
        similars = model.docvecs.most_similar(negative=[inferred_negative_vector], topn=topn)

    # TODO: for larger corpuses use Annoy indexer (most_similar() argument indexer=AnnoyIndexer):
    # Why use Annoy?
    # Annoy = Approximate Nearest Neighbors Oh Yeah
    # The current implementation for finding k nearest neighbors in a vector space in gensim has linear complexity via brute force in the number of indexed documents, although with extremely low constant factors. The retrieved results are exact, which is an overkill in many applications: approximate results retrieved in sub-linear time may be enough. Annoy can find approximate nearest neighbors much faster.
    # https://markroxor.github.io/gensim/static/notebooks/annoytutorial.html


    # TODO: try to find cosmul method for doc2vec too
    # similars = model.docvecs.most_similar_cosmul(positive=[inferred_positive_vector], negative=[inferred_negative_vector], topn=(num_results + len(notlike)))
    # From the Levy and Goldberg paper, if you are trying to find analogies (or combining/comparing more than 2 word vectors), the first method (3CosAdd or eq.3 of paper) is more susceptible of getting dominated by 1 comparison, as compared to second method (3CosMul or eq.4 of paper).
    # http://stackoverflow.com/questions/31524898/gensim-word2vec-semantic-similarity


    # similars = similars[:number_of_previously_generated_similars]  # skip that amount of results
    # comment-out: this is a problem in case the results are dancing! Instead we are using tags list

    number_of_previously_generated_similars = topn



    if (len(similars) == prev_result_count):    # no more results can be generated, prevent infinite loop
        break

    prev_result_count = len(similars)


    
    for similar in similars:

        (tag, score) = similar


        
        # need to detect previously processed search results in case we are extending the search results list when looking for restricted corpus results and the database is indexed on full corpus
        # TODO: gather all results and sort them again and only the display the results. Else it may happen that invoking the search in multiple loops causes the scores to "dance".
        if (tag in previously_found_tags):
            continue

        previously_found_tags.append(tag)



        if (result_index >= num_results):
            break

        if (tag in notlike_tags):
            continue


        (corpus, fname, line_no, lang_line_no) = tag.split(':')
        line_no = int(line_no)
        lang_line_no = int(lang_line_no)


        if (corpus not in search_corpus_dirs):
            continue

    
        filesize = os.path.getsize(os.path.join(basedatadir, fname))
        if filesize == 0:   # ignore zero-size files. Search may return them when using old index database
            continue

        lines = [x for x in open(os.path.join(basedatadir, fname))] 
        if (len(lines) == 0):    # empty lines?
            continue


        displayed_tags.append(tag)


        result_index = result_index + 1


        # read document title lines

        title_lines = ''
        num_lines = 0

        est_tags_encountered = False
        eng_tags_encountered = False
        est_eng_order_detected = False
        est_eng_line_offset = 0

        for line in lines:

            if (len(line) == 0):         # empty line
                continue


            line = line.strip()   


            # detect est-eng line ordering. Detect that for each source file separately

            if (
                not do_not_detect_tags
                and not est_eng_order_detected
                and not est_tags_encountered
                and line[:len(est_tag_start)].lower() == est_tag_start 
                and line[-len(est_tag_end):].lower() == est_tag_end
            ):
                est_tags_encountered = True 
                        
                if (eng_tags_encountered):
                    est_eng_order_detected = True
                    est_eng_line_offset = -1    # eng tags are before est tags

            if (
                not do_not_detect_tags
                and not est_eng_order_detected
                and not eng_tags_encountered
                and line[:len(eng_tag_start)].lower() == eng_tag_start 
                and line[-len(eng_tag_end):].lower() == eng_tag_end
            ):
                eng_tags_encountered = True 
                        
                if (est_tags_encountered):
                    est_eng_order_detected = True
                    est_eng_line_offset = 1    # est tags are before eng tags


            # gather title text

            if (do_not_detect_tags
                or (
                    line[:len(result_tag_start)].lower() == result_tag_start 
                    and line[-len(result_tag_end):].lower() == result_tag_end)):  

                num_lines = num_lines + 1

                line = line if do_not_detect_tags else line[len(result_tag_start):-len(result_tag_end)].strip()

                title_lines = ('' if title_lines == '' else title_lines + ' - ') + line
            
                if (num_lines == 3):    # enough title lines
                    break
    
            #/ if (line[:len(result_tag_start)] == result_tag_start and line2[-len(result_tag_end):] == result_tag_end):  

        #/ for line in lines:



        # print the result

        if (use_line_search):

            # est_eng_line_offset : 1 if est tags are before eng tags
            if (search_language == result_language):
                search_result_line_offset = 0   # NB!
            elif (search_language == "est"):
                search_result_line_offset = est_eng_line_offset     # result offset: 1 if est tags are before eng tags 
            else:   # if (search_language == "eng")
                search_result_line_offset = -est_eng_line_offset    # result offset: -1 if est tags are before eng tags 

            line_no = line_no + search_result_line_offset        # NB! search_result_line_offset
    

            line = lines[line_no].strip()
            line = line if do_not_detect_tags else line[len(result_tag_start):-len(result_tag_end)].strip()

            print("[" + str(result_index) + "] " + "{0:.3f}".format(score) + ' "' + title_lines + '" : /' + line + "/ : line #" + str(lang_line_no + 1) + " @ " + fname)
            # print("[" + str(result_index) + "] (" + str(score) + ") : " + fname + " (" + title_lines + ")" + " : line " + str(lang_line_no + 1) + ' : ' + line)
            # print("[" + str(result_index) + "] " + "{0:.3f}".format(score) + ' /' + line + '/ "' + title_lines + '" : line #' + str(lang_line_no + 1) + " @ " + fname)

        else:   # if (use_line_search):

            print("[" + str(result_index) + "] " + "{0:.3f}".format(score) + ' "' + title_lines + '" @ ' + fname)


    #/ for similar in similars:


    if (result_index >= num_results):
        break
    else:
        continue    # while True:



# save query results so that they can be used for next query with "notlike" argument

with open(last_results_file_name, 'w') as filehandle:
    filehandle.write('\n'.join(displayed_tags))



sys.exit()
