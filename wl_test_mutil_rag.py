import json
import os
import sys
import re
import time
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument
#from llama_index.llms.openai import OpenAI

from llama_index.core import PromptTemplate
from llama_index.core.response.notebook_utils import (
    display_query_and_multimodal_response,
)

from llama_index.core.base.embeddings.base import (
    BaseEmbedding
)
from llama_index.core.schema import ImageNode, TextNode

from wl_custom_embeding import wlEmbedding, wlMultiModalEmbedding
from wl_MultiModal_VectorStoreIndex import wlMultiModalVectorStoreIndex

import logging

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

'''
a = os.path.abspath(__file__)
print(a)
b = os.path.dirname(a)
print(b)

sys.path.append(b)
print(sys.path)
'''

# context images  加载上下文图像
image_path = "./asl_data/images"
image_documents = SimpleDirectoryReader(image_path).load_data()
for image in image_documents:
    print("=================image.image_path:",image.image_path)
    print("=================image:",image)

# context text  上下文描述
asl_text_descriptions = None
with open("./asl_data/asl_text_descriptions.json") as json_file:
    asl_text_descriptions = json.load(json_file)
text_format_str = "To sign {letter} in ASL: {desc}."
text_documents = [
    Document(text=text_format_str.format(letter=k, desc=v))
    for k, v in asl_text_descriptions.items()
]

# 打印文本内容
for doc in text_documents:
    print("-------------->doc.text:",doc.text)

node_parser = SentenceSplitter.from_defaults()
image_nodes = node_parser.get_nodes_from_documents(image_documents)
text_nodes = node_parser.get_nodes_from_documents(text_documents)

# 遍历 image_nodes 并添加描述
for imageNode in image_nodes:
    print("image_nodes is imageNode:",isinstance(imageNode, ImageNode)) 
    print("=================imageNode.image_path:",imageNode.image_path)

    # 提取文件名，不包括扩展名
    file_name = re.search(r"([^/]+)(?=\.\w+$)", imageNode.image_path).group(0)

    # 查找对应的描述
    if file_name in asl_text_descriptions:
        description = asl_text_descriptions[file_name]
        # 使用模板格式化描述内容
        formatted_description = text_format_str.format(letter=file_name, desc=description)

        # 将描述添加到 imageNode 中，比如 text 字段
        imageNode.text = formatted_description  # imageNode 有 text 字段来存储文本内容

    print("=================imageNode:",imageNode)

# 打印文本内容
for textNode in text_nodes:
    print("text_nodes is  TextNode:",isinstance(textNode, TextNode)) 
    print("-------------->textNode:",textNode)

# 文本Embedding
myEmbed_model = wlEmbedding()
print(type(myEmbed_model)) 
print(myEmbed_model.__class__.__module__)  # 打印模块路径
print(BaseEmbedding.__module__)  # 打印基类模块路径
# 多模态 Embedding
myMutilModalEmbed_model = wlMultiModalEmbedding()

# 设置多模态 的Embedding ,创建多模态的 index, image_embed_model: EmbedType = "clip:ViT-B/32",
#-------------------------
#   这是第一种方法使用 text_nodes +  image_nodes 创建索引
#-------------------------
asl_index = wlMultiModalVectorStoreIndex(text_nodes + image_nodes,embed_model = myMutilModalEmbed_model, image_embed_model = myMutilModalEmbed_model)
#asl_index = MultiModalVectorStoreIndex(image_nodes,embed_model = myMutilModalEmbed_model, image_embed_model = myMutilModalEmbed_model)


# define our QA prompt template
qa_tmpl_str = (
    "Images of hand gestures for ASL are provided.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "If the images provided cannot help in answering the query\n"
    "then respond that you are unable to answer the query. Otherwise,\n"
    "using only the context provided, and not prior knowledge,\n"
    "provide an answer to the query."
    "Query: {query_str}\n"
    "Answer: "
)

qa_tmpl_str_cn = (
    "图片提供了 ASL 的手势图像。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "如果根据提供的图像无法回答查询的问题\n"
    "则回答您无法回答查询。否则,\n"
    "仅使用提供的上下文，而不是先前的知识，\n"
    "回答查询的结果。"
    "查询: {query_str}\n"
    "答案: "
)

qa_tmpl = PromptTemplate(qa_tmpl_str)
print("qa_tmpl---:",qa_tmpl)

local_chatLLm_cfg = {
    "model":"/work/wl/wlwork/my_models/Qwen2-VL-72B-Instruct-GPTQ-Int4",  # 使用您的模型名称
    "api_base":"http://172.21.30.230:8980/v1/",  # 您的 vllm 服务器地址
    "api_key":"EMPTY",  # 如果需要的话
}

# define our lmms ,采用本地 qwen2-vl大模型
openai_mm_llm = OpenAIMultiModal(
    model="/work/wl/wlwork/my_models/Qwen2-VL-72B-Instruct-GPTQ-Int4",
    api_key = "EMPTY",
    api_base = "http://172.21.30.230:8980/v1/",
)

# define our RAG query engines
rag_engines = {
    "mm_qwen2vl": asl_index.as_query_engine(
        llm = openai_mm_llm,
        text_qa_template = qa_tmpl,
    ), 
}

# 



QUERY_STR_TEMPLATE = "How can I sign a \"{symbol}\"?."
QUERY_STR_TEMPLATE_CN = "我该如何用ASL手势表示字母 \"{symbol}\"?.请描述。另外请告诉我参考第几张图片，中文回答"

letter = "A"
query = QUERY_STR_TEMPLATE_CN.format(symbol=letter)
# 记录访问大模型的初始时间
wlstartTime = time.time()
#rag_engines["mm_qwen2vl"].retriever.image_similarity_top_k = 1
#rag_engines["mm_qwen2vl"].retriever.similarity_top_k = 1

response = rag_engines["mm_qwen2vl"].query(query)

wlEndTime = time.time()
queryTimeCost = wlEndTime - wlstartTime

print("============response.metadata:",response.metadata)
print("花费了实际是(s):",queryTimeCost,flush=True)

print("============response.metadata:",response.metadata)

display_query_and_multimodal_response(query, response)

