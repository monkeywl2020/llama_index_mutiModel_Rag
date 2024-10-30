from typing import Dict, Sequence, List
from llama_index.core.schema import ImageNode
from llama_index.core.schema import BaseNode, ImageNode, TextNode
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.indices.utils import (
    async_embed_image_nodes,
    async_embed_nodes,
    embed_image_nodes,
    embed_nodes,
)
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from wl_custom_embeding import wlMultiModalEmbedding

class wlMultiModalVectorStoreIndex(MultiModalVectorStoreIndex):
    """一个改进的多模态向量存储索引类，增强了图像编码功能，允许在生成图像嵌入时使用图像的文本描述。"""

    @property
    def image_embed_model(self) -> wlMultiModalEmbedding:
        return self._image_embed_model
    
    def embed_image_nodes(
        self,
        nodes: Sequence[ImageNode],
        embed_model: wlMultiModalEmbedding,
        show_progress: bool = False,
    ) -> Dict[str, List[float]]:
        """Get image embeddings of the given nodes, run image embedding model if necessary.

        Args:
            nodes (Sequence[ImageNode]): The nodes to embed.
            embed_model (MultiModalEmbedding): The embedding model to use.
            show_progress (bool): Whether to show progress bar.

        Returns:
            Dict[str, List[float]]: A map from node id to embedding.
        """
        id_to_embed_map: Dict[str, List[float]] = {}

        images_to_embed = []       # 要Embedding的图片
        images_desc_to_embed = []  # 要Embedding的图片的描述
        ids_to_embed = []
        for node in nodes:
            if node.embedding is None:
                ids_to_embed.append(node.node_id)
                images_to_embed.append(node.resolve_image())
                images_desc_to_embed.append(node.get_text())# 获取 ImageNode 中的图片描述内容 ，这部分内容和 上面是 一 一对应的
            else:
                id_to_embed_map[node.node_id] = node.embedding

        new_embeddings = embed_model.get_image_with_desc_embedding_batch(
            images_to_embed, images_desc_to_embed, show_progress=show_progress
        )

        for new_id, img_embedding in zip(ids_to_embed, new_embeddings):
            id_to_embed_map[new_id] = img_embedding

        return id_to_embed_map

    #=======================================
    #   重写覆盖 _get_node_with_embedding 函数
    #   改进的这个类里面 image的Embedding，在做Embedding图片的时候可以将 图片的描述 + 图片 一起做Embedding
    #=======================================
    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        is_image: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_text_embed_map = None

        if is_image:
            assert all(isinstance(node, ImageNode) for node in nodes)
            id_to_embed_map = self.embed_image_nodes(
                nodes,  # type: ignore
                embed_model=self._image_embed_model,
                show_progress=show_progress,
            )

            # text field is populate, so embed them
            if self._is_image_to_text:
                id_to_text_embed_map = embed_nodes(
                    nodes,
                    embed_model=self._embed_model,
                    show_progress=show_progress,
                )
                # TODO: refactor this change of image embed model to same as text
                self._image_embed_model = self._embed_model  # type: ignore

        else:
            id_to_embed_map = embed_nodes(
                nodes,
                embed_model=self._embed_model,
                show_progress=show_progress,
            )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.model_copy()
            result.embedding = embedding
            if is_image and id_to_text_embed_map: # 如果是图片，并且设置了将图片Embedding成text的Embedding
                assert isinstance(result, ImageNode)
                text_embedding = id_to_text_embed_map[node.node_id]
                result.text_embedding = text_embedding
                result.embedding = (
                    text_embedding  # TODO: re-factor to make use of both embeddings
                )
            results.append(result)
        return results
