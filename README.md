# ChatGLM 2 词表扩充代码

- 该代码分为两部分
    - 第一部分：训练要扩充的词表sp模型
    - 第二部分：将训练好的词表模型进行融合原始的chatglm2的tokenizer的模型


### 1.训练Spmodel

~~~shell
python code/train.py \
    --train_data_dir data \                         #要训练的数据集文件夹
    --save_model_path_name model/code_bpe111        #要存储的文件名字code_bpe111,注意：只写文件名就好，不用加后缀，内部会自动加后缀".model"
~~~


### 2.合并tokenizer

~~~shell
python code/merge.py \
    --chatglm_tokenizer_modelfile chatglm6b/tokenizer.model \   #chatglm 的tokenizer_model 的文件地址
    --tibet_sp_model model/code_bpe.model \                     #第一步训练好的模型地址
    --output_sp_dir model/merged_tokenizer_sp \                 #输出的文件地址
    --save_model_name tibet_tokenizer.model                     #输出的模型名字（只写一个名字就可以了）
~~~


### 3.最后把训练好的模型复制到chatglm文件中



### 可以采用脚本执行
    bash train.sh
    bash merge.sh
