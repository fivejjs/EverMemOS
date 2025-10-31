from typing import List
import requests

class RerankerProvider:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def rerank(self, queries: List[str], docs: List[str], instruction: str = None) -> str:
        if 'Qwen3' in self.model_name:
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            if instruction is None:
                instruction = "Given a user's question and a text passage, determine if the passage contains specific information that directly answers the question. A relevant passage should provide a clear and precise answer, not just be on the same topic."
            query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
            document_template = "<Document>: {doc}{suffix}"
        else:
            raise ValueError(f"Model {self.model_name} is not supported, only Qwen3-Reranker series models is supported")


        queries = [
            query_template.format(prefix=prefix, instruction=instruction, query=query)
                for query in queries
        ]
        documents = [
            document_template.format(doc=doc, suffix=suffix) for doc in docs
        ]

        response = requests.post(self.base_url,
                                json={
                                    "model": self.model_name,
                                    "text_1": queries,
                                    "text_2": documents,
                                    "truncate_prompt_tokens": -1,
                                }).json()
        scores = [item['score'] for item in response['data']]
        return scores

if __name__ == "__main__":
    queries = [
        "What book did Melanie read from Caroline's suggestion?",
        "What book did Melanie read from Caroline's suggestion?",
        "What book did Melanie read from Caroline's suggestion?",
        "What book did Melanie read from Caroline's suggestion?",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Tom的家乡是瑞士",
        "On October 13, 2023 at 10:34 AM UTC, Melanie initiated a conversation with Caroline, expressing gratitude for a previous tip and sharing that she had recently experienced a setback. Last month (September 2023), Melanie had sustained an injury that forced her to take a break from pottery, an activity she uses for self-expression and emotional peace. She described the break as emotionally challenging but noted she was coping well. To stay engaged creatively, Melanie had been reading a book recommended by Caroline earlier and had begun painting as a substitute outlet. Caroline responded with empathy, expressing concern for Melanie’s well-being and offering support. Melanie then shared a photograph of a painting she completed last week (October 6, 2023), depicting a sunset with a pink sky, explaining that the colors evoked a sense of calm and served as a personal emotional anchor. Caroline praised the artwork, highlighting its serene vibe and vibrant colors, and asked if Melanie had any other pieces to share. Melanie responded by sharing a second photograph of an abstract painting on a wall with a blue background, describing her intention to convey tranquility through flowing blue streaks while incorporating vibrant colors to maintain visual energy. Caroline admired the piece, noting how the blue enhanced the peaceful atmosphere, and shared that she had recently been experimenting with abstract art as a form of unstructured self-expression. She also mentioned attending a poetry reading the previous Friday (October 6, 2023), which she found powerful and emotionally resonant. The conversation concluded with both women affirming the therapeutic value of artistic expression and reinforcing their mutual appreciation for creative outlets during personal challenges.",
        "On July 12, 2023 at 4:36 PM UTC, Caroline reflected on her personal struggles with mental health and expressed gratitude for the support she received, which had a profound impact on her life. This experience motivated her to explore careers in counseling and mental health, with the goal of helping others on similar journeys. Melanie responded with encouragement, sharing a photograph of a book cover with a gold coin on it, symbolizing the pursuit of dreams. The image reminded Melanie of a book she read last year (July 2022), which inspired her to stay committed to her aspirations. Caroline expressed deep appreciation for the photo, noting that books have played a significant role in guiding, motivating, and helping her discover her identity. She shared a photograph of a dog sitting in a boat on the water, which accompanied her recommendation of the book 'Becoming Nicole' by Amy Ellis Nutt. This true story about a transgender girl and her family resonated deeply with Caroline, providing her with a sense of connection, hope, and validation for her own path. Melanie asked what Caroline had taken away from the book, and Caroline explained that it taught her the importance of self-acceptance, the value of finding support systems, and the enduring presence of hope and love even during difficult times. She also highlighted the joy that pets bring, referencing the dog in her photo. Melanie shared a photograph of two little girls sitting on steps with a dog, affirming the emotional benefits of pets and expressing agreement with Caroline’s insights. Melanie then shared another photo of a cat lying on the floor with its head resting on the ground, introducing her pets Luna and Oliver. Caroline responded with delight, calling them adorable and asking for their names. Melanie revealed that the pets were named Luna and Oliver, describing them as sweet and playful, and shared a photo of a person wearing pink sneakers on a white rug, explaining that the shoes were new and intended for running. Melanie mentioned that she had been running more frequently since their last conversation (June 2023), noting that running helped her destress and clear her mind. Caroline complimented the purple color of the sneakers and asked if they were for walking or running. Melanie confirmed they were for running, reinforcing her commitment to the activity as a form of mental wellness. Throughout the conversation, both women expressed positive emotions, including gratitude, inspiration, joy, and encouragement, with Caroline’s journey toward a mental health career and Melanie’s ongoing self-care practices serving as central themes.",
    
        
    ]
    reranker = RerankerProvider(base_url="http://0.0.0.0:12000/score", model_name="Qwen3-Reranker-4B")
    print(reranker.rerank(queries, documents))