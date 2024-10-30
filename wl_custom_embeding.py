import torch
import asyncio
from typing import Any, List, Optional,Coroutine,Tuple
#from io import BytesIO
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding
)

from FlagEmbedding.visual.modeling import Visualized_BGE
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import ImageType
from llama_index.core.utils import get_tqdm_iterable

# 定义自定义的 Embedding 类，纯文本
class wlEmbedding(BaseEmbedding):
    # 声明一个可选的 model 字段
    model: Optional[Any] = None

    def __init__(self, 
                 model_name: str = '/work/wl/wlwork/my_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',
                 model_weight: str = "/work/wl/wlwork/my_models/bge_visualized/Visualized_m3.pth",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = Visualized_BGE(model_name_bge = model_name, model_weight = model_weight)

    def _get_query_embedding(self, query: str) -> Embedding:
        embedding = self.model.encode(text = query,image = None)
        return embedding.squeeze(0).tolist() 

    async def _aget_query_embedding(self, query: str) -> Embedding:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._get_query_embedding, query)
        return embedding
    
    # 对文本和image进行Embedding，返回Embedding内容
    def _get_text_embedding(self, text: str) -> Embedding:
        embedding = self.model.encode(text = text, image = None)
        return embedding.squeeze(0).tolist() 


class wlMultiModalEmbedding(MultiModalEmbedding):
    # 声明一个可选的 model 字段
    model: Optional[Any] = None
    def __init__(self, 
                 model_name: str = '/work/wl/wlwork/my_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181',
                 model_weight: str = "/work/wl/wlwork/my_models/bge_visualized/Visualized_m3.pth",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = Visualized_BGE(model_name_bge = model_name, model_weight = model_weight)

    # 获取输入字符串的 Embedding内容
    def _get_text_embedding(self, text: str) -> Embedding:
        embedding = self.model.encode(text = text, image = None)
        return embedding.squeeze(0).tolist() 

    # 获取查询内容的Embedding ,入参是 文字查询内容
    def _get_query_embedding(self, query: str) -> Embedding:
        embedding = self.model.encode(text = query,image = None)
        return embedding.squeeze(0).tolist() 

    #异步： 获取查询内容的Embedding ,入参是 文字查询内容
    async def _aget_query_embedding(self, query: str) -> Embedding:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._get_query_embedding, query)
        return embedding

    #获取查询内容的Embedding ,入参是 文字查询内容 和 用来做查询的图片
    def _get_query_embedding_with_img(self, query: str, query_img_file_path: ImageType) -> Embedding:
        embedding = self.model.encode(text = query,image = query_img_file_path)
        return embedding.squeeze(0).tolist() 

    #异步： 获取查询内容的Embedding ,入参是 文字查询内容 和 用来做查询的图片
    async def _aget_query_embedding_with_img(self, query: str, query_img_file_path: ImageType) -> Embedding:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._get_query_embedding_with_img, query, query_img_file_path)
        return embedding

    #获取 image的 Embedding值 
    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        #Embedding的入参是 文件描述 和 文件路径
        embedding = self.model.encode(text = None, image = img_file_path)
        return embedding.squeeze(0).tolist() 
    
    #异步： 获取 image的 Embedding值 ，入参是 文件描述 和 文件路径
    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._get_image_embedding, img_file_path)
        return embedding
    
    
    #获取 image的 Embedding值 
    def _get_image_with_desc_embedding(self, img_file_path: ImageType, img_desc: str) -> Embedding:
        #Embedding的入参是 文件描述 和 文件路径
        embedding = self.model.encode(text = img_desc, image = img_file_path)
        return embedding.squeeze(0).tolist() 
    
    #异步： 获取 image的 Embedding值 ，入参是 文件描述 和 文件路径
    async def _aget_image_with_desc_embedding(self, img_file_path: ImageType, img_desc: str) -> Embedding:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._get_image_with_desc_embedding, img_file_path, img_desc)
        return embedding

    def _get_image_with_desc_embeddings(self, img_file_paths_with_desc: List[Tuple[ImageType, str]]) -> List[Embedding]:
        """
        Embed the input sequence of image synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # 将路径和描述一起传递给 _get_image_embedding
        return [
            self._get_image_with_desc_embedding(img_file_path, img_desc) 
            for img_file_path, img_desc in img_file_paths_with_desc
        ]

    async def _aget_image_with_desc_embeddings(
        self, img_file_paths_with_desc: List[Tuple[ImageType, str]]
    ) -> List[Embedding]:
        """
        Embed the input sequence of image asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[
                self._aget_image_with_desc_embedding(img_file_path,img_desc)
                for img_file_path, img_desc in img_file_paths_with_desc
            ]
        )


    # 图片做批量 Embedding
    def get_image_with_desc_embedding_batch(
        self, img_file_paths: List[ImageType], img_file_descs: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of image embeddings, with batching."""
        # 检查图片路径和描述列表长度一致
        assert len(img_file_paths) == len(img_file_descs), "图片路径和描述数量不匹配"

        cur_batch: List[Tuple[ImageType, str]] = []
        result_embeddings: List[Embedding] = []

        # 使用 zip 将图片路径和描述组合成 (路径, 描述) 的元组对
        img_path_desc_pairs = list(zip(img_file_paths, img_file_descs))

        queue_with_progress = enumerate(
            get_tqdm_iterable(
                img_path_desc_pairs, show_progress, "Generating image embeddings"
            )
        )

        for idx, (img_file_path, img_desc) in queue_with_progress:
            cur_batch.append((img_file_path, img_desc)) # 将路径和描述作为一个元组存入批次列表
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_image_with_desc_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                cur_batch = []

        return result_embeddings

    # 异步：图片做批量 Embedding
    async def aget_image_with_desc_embedding_batch(
        self, img_file_paths: List[ImageType],  img_file_descs: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of image embeddings, with batching."""
        cur_batch: List[Tuple[ImageType, str]] = []

        callback_payloads: List[Tuple[str, List[ImageType]]] = []
        result_embeddings: List[Embedding] = []

        # 使用 zip 将图片路径和描述组合成 (路径, 描述) 的元组对
        img_path_desc_pairs = list(zip(img_file_paths, img_file_descs))

        embeddings_coroutines: List[Coroutine] = []
        for idx, (img_file_path, img_desc) in enumerate(img_path_desc_pairs):
            cur_batch.append((img_file_path, img_desc))
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_image_with_desc_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                nested_embeddings = await tqdm_asyncio.gather(
                    *embeddings_coroutines,
                    total=len(embeddings_coroutines),
                    desc="Generating embeddings",
                )
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, image_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: image_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings