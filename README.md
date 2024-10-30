本工程是llama index支持多模态RAG的工程。

1：llama index的多模态 rag，文本和图像的Embedding都是独立。目前支持多模态的Embedding 还有 图片描述文字+图片 这种方式进行Embedding的。经过测试，这种相似性查找准确率比较高。所以进行了修改。

2：新增了个自定义的 wlMultiModalVectorStoreIndex 继承自 llama index的MultiModalVectorStoreIndex。 wlMultiModalEmbedding 继承自 MultiModalEmbedding。

3：此外使用本地llm大模型还需要修改下llama index的 llama_index/multi_modal_llms/openai/utils.py 文件内容。
