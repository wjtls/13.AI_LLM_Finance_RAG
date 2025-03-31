from typing import List

from llama_index.core import PromptTemplate
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
import asyncio

import a_Agent_sys_prompt

from tqdm.asyncio import tqdm
import nest_asyncio
nest_asyncio.apply()


# 퓨전리트리버 클래스 정의 (git 코드 참고)
class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        sys_query: str,
        similarity_top_k,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        super().__init__()

        # 쿼리 디컴포지션 (복잡성 높은 질문일때)
        self.breaker_prompt = sys_query
        self.prompt = PromptTemplate(self.breaker_prompt)
        #self.sub_prompt= PromptTemplate(self.breaker_prompt2)

    def generate_queries(self, llm, query: str, num_queries):
        response = llm.predict(self.prompt, num_queries=num_queries, query=query)
        response = str(response).strip().replace('\n\n','\n') # 줄바꿈때문에 공백을 질문으로 넣으면 리트리버 엔진 오류발생함  -> 줄바꿈으로 인한 공백제거 \n한번만 하게끔
        queries = response.split("\n")
        queries_str = "\n".join(queries)
        print(f"Fusion_break_model -> Generated queries:\n{queries_str}")
        return queries

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        num_q = 2
        queries = self.generate_queries(self._llm, query_bundle.query_str, num_queries=num_q)

        results = asyncio.run(self.run_queries(queries, self._retrievers))
        final_results = self.fuse_results(results, similarity_top_k=self._similarity_top_k)

        return final_results


    async def run_queries(self,query_group,retriever): # 다수 리트리버 결과 합치는 용도
        task=[]
        for query in query_group:
            for step, retriever_model in enumerate(retriever):
                task.append(retriever_model.aretrieve(str(query)))

        task_result = await tqdm.gather(*task)

        result_dict = {}
        for step,(query,query_result) in enumerate(zip(query_group,task_result)):
            result_dict[(query,step)]=query_result

        return result_dict

    def fuse_results(self, results_dict, similarity_top_k):
        """Fuse results."""
        k = 60.0
        fused_scores = {}
        text_to_node = {}

        for nodes_with_scores in results_dict.values():
            for rank, node_with_score in enumerate(
                    sorted(
                        nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
                    )
            ):
                text = node_with_score.node.get_content()
                text_to_node[text] = node_with_score
                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + k)

        # fusion 스코어 기반 소팅
        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )
        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score

        return reranked_nodes[:similarity_top_k]
