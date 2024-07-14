from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def get_model(args):
    if args.model == 'roberta-base':
        return AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('roberta-base',model_max_length=args.max_length)
    elif args.model == 'distilroberta-base':
        return AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('distilroberta-base',model_max_length=args.max_length)
    elif args.model == 'bert-base':
        return AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
    elif args.model == 'distilbert-base':
        return AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('distilbert-base-cased',model_max_length=args.max_length)
    elif args.model == 'microsoft/deberta-base':
        return AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=args.number_of_class, problem_type="multi_label_classification"), \
               AutoTokenizer.from_pretrained('microsoft/deberta-base',model_max_length=args.max_length)
    elif args.model == 'google/electra-base-discriminator':
        return AutoModelForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=args.number_of_class), \
               AutoTokenizer.from_pretrained('google/electra-base-discriminator',model_max_length=args.max_length)
    elif args.model == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2',model_max_length=args.max_length)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=args.number_of_class)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer
    elif args.model == 't5-base':
        from transformers import AutoConfig
        from .t5_custom import T5ForSequenceClassification
        config = AutoConfig()
        tokenizer = AutoTokenizer.from_pretrained('t5-base',model_max_length=args.max_length)
        # model = AutoModelForSequenceClassification.from_pretrained('t5-base', num_labels=args.number_of_class)
        model = T5ForSequenceClassification(num_labels=args.number_of_class)
        return model, tokenizer
    elif args.model == 'facebook/bart-base':
        from transformers import BartTokenizer, BartForSequenceClassification
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', model_max_length=args.max_length)
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=args.number_of_class)
        return model, tokenizer
    elif args.model == 'char_cnn':
        from .char_cnn import CharacterLevelCNN, CharCNNTokenizer
        return CharacterLevelCNN(args), CharCNNTokenizer(args)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
    args.vocab_size = tokenizer.vocab_size 
    if args.model == 'word_cnn':
        from .word_cnn import WordLevelCNN
        model = WordLevelCNN(args)
    elif args.model == 'bilstm':
        from .lstm import LSTM
        model = LSTM(args)
    elif args.model == 'lstm':
        from .lstm import LSTM
        model = LSTM(args)
    elif args.model == 'rnn':
        from .rnn import RNN
        model = RNN(args)
    elif args.model == 'birnn':
        from .rnn import RNN
        model = RNN(args)
    else:
        raise Exception('Wrong model')
    return model, tokenizer