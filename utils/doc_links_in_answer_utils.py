from langchain.docstore.document import Document


class DocLinksAnswerSupport:

    _source_link_var_name = 'source_link'

    def prompt_support(self, prompt):
        return f"""{prompt}
        You have a list of documents that you will take into account in the answer. Each document has a {self._source_link_var_name}. If you have chosen any document or documents for the answer, specify the link taken from {self._source_link_var_name} in your answer for each document you use in the answer. The link must exactly match the value of {self._source_link_var_name}. Very important not to use the word {self._source_link_var_name} as anchor text of link in your answer, only value from it. For link anchor text create some words in the same language as your answer and related with document that you link."""

    def enrich_docs_with_source_links(self, docs: list[Document]):
        for i, doc in enumerate(docs):
            text = doc.page_content
            enriched_text = f"""
            ---
            {text}
            {self._source_link_var_name}: {doc.metadata.get('url')}
            ---
            """
            doc.page_content = enriched_text
        return docs


doc_links_answer_support = DocLinksAnswerSupport()
