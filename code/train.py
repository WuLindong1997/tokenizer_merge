import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import os
import argparse

def train_sentence_piece_token(train_data_dir,save_model_path):
    
    data_paths = os.listdir(train_data_dir)
    data_paths = [os.path.join(train_data_dir, path) for path in data_paths]

    # 训练并保存 model
    spm.SentencePieceTrainer.train(input=",".join(data_paths), 
    model_prefix=save_model_path,
        vocab_size=10000, 
        max_sentence_length=20000,
        num_threads=64,
        train_extremely_large_corpus=True,
        model_type='bpe', 
        byte_fallback=True)
    
# 测试训练的model是否可用
def read(model_path):
    # load
    chinese_sp_model = spm.SentencePieceProcessor(model_path)
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
    chinese_spm_tokens_set = set(p.piece for p in chinese_spm.pieces)

    #test
    print("len(chinese_spm_tokens_set):", len(chinese_spm_tokens_set))
    text='མཚེའུ དེ ཉིད སྲུང སྐར དེའི ས ངོས ཀྱི འཁྱགས རོམ བང རིམ ལས སྤྱི ལེ༣གྱི ས འོག ཏུ ཡོ'
    print("Test text:\n",text)
    print(f"encode:{chinese_sp_model.encode(text)}")

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_data_dir',default="data",type=str)
    arg_parser.add_argument('--save_model_path_name',default="model/code_bpe111",type=str)
    return arg_parser.parse_args()
if __name__ == "__main__":

    args = get_args()
    train_sentence_piece_token(args.train_data_dir,args.save_model_path_name)
    model_path = f'{args.save_model_path_name}.model'
    read(model_path)
