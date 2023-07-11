from langchain.docstore.document import Document


class DocLinksAnswerSupport:

    _source_link_var_name = 'source_doc_link'

    def prompt_support(self, prompt):
        return f"""{prompt}
        You have a list of documents that you will take into account in the answer. Each document has a {self._source_link_var_name}. If you have chosen any document for the answer, specify the link taken from {self._source_link_var_name} in your answer. The link must exactly match the {self._source_link_var_name}. Don't change it."""

    def enrich_docs_with_source_links(self, docs: list[Document]):
        for i, doc in enumerate(docs):
            text = doc.page_content
            enriched_text = f"""
            Document {i}
            {text} 
            Document {i} {self._source_link_var_name}: {doc.metadata.get('url')}
            """
            doc.page_content = enriched_text
        return docs


doc_links_answer_support = DocLinksAnswerSupport()
