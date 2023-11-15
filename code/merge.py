import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from tokenization_chatglm import ChatGLMTokenizer
import argparse
parser = argparse.ArgumentParser()



def merge_tokenizer(chatglm_tokenizer_dir,tibet_sp_model,output_sp_dir,save_model_name):

    # load
    chatglm_tokenizer_model = ChatGLMTokenizer(vocab_file = chatglm_tokenizer_dir)
    tibet_sp_model = spm.SentencePieceProcessor(tibet_sp_model)

    tibet = sp_pb2_model.ModelProto()
    tibet.ParseFromString(chatglm_tokenizer_model.tokenizer.sp_model.serialized_model_proto())
    tibet_sm = sp_pb2_model.ModelProto()
    tibet_sm.ParseFromString(tibet_sp_model.serialized_model_proto())


    ## Add Chinese tokens to ChatGLM tokenizer
    tibet_tokens_set=set(p.piece for p in tibet.pieces)
    print(len(tibet_tokens_set))
    print(f"Before:{len(tibet_tokens_set)}")
    for p in tibet_sm.pieces:
        piece = p.piece
        if piece not in tibet_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            tibet.pieces.append(new_p)
    print(f"New model pieces: {len(tibet.pieces)}")

    ## Save
    output_sp_dir = output_sp_dir
    os.makedirs(output_sp_dir,exist_ok=True)
    with open(os.path.join(output_sp_dir,save_model_name),'wb') as f:
        f.write(tibet.SerializeToString())

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--chatglm_tokenizer_modelfile',default="chatglm6b/tokenizer.model",type=str)
    arg_parser.add_argument('--tibet_sp_model',default="model/code_bpe.model",type=str)
    arg_parser.add_argument('--output_sp_dir',default="model/merged_tokenizer_sp",type=str)
    arg_parser.add_argument('--save_model_name',default="tibet_tokenizer.model",type=str)
    return arg_parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    #合并tokenizer model 并且保存
    chatglm_tokenizer_modelfile = args.chatglm_tokenizer_modelfile
    tibet_sp_model = args.tibet_sp_model
    output_sp_dir = args.output_sp_dir
    save_model_name = args.save_model_name
    merge_tokenizer(chatglm_tokenizer_modelfile,tibet_sp_model,output_sp_dir,save_model_name)
    
    #---------测试---------
    raw_input_tokenizer_file = chatglm_tokenizer_modelfile
    new_input_tokenizer_file = os.path.join(output_sp_dir,save_model_name)
    #加上tibet的tokenizer
    raw_chatglm_tokenizer = ChatGLMTokenizer(vocab_file = raw_input_tokenizer_file)
    new_chatglm_tokenizer = ChatGLMTokenizer(vocab_file = new_input_tokenizer_file)

    text='ལ གཙོ གནད ཐོག སྔོན འགོག བལྟ སྐྱོང བྱ རྒྱུ དང བརྟག དཔྱད ཞིབ བཤེར བྱེད ཚད དེ བརྒྱ ཆ ཡན ཟིན པ བྱ རྒྱུ དེ དང ཆབས'
    print(f"Tokenized by raw chatglm tokenizer:{raw_chatglm_tokenizer.tokenize(text)}")
    print(f"Tokenized by new chatglm tokenizer:{new_chatglm_tokenizer.tokenize(text)}")