def prompt_with_system_info(prompt, langchain_chat_history_prompt_enable=False):
    chat_history_prompt_part = ''
    if langchain_chat_history_prompt_enable:
        chat_history_prompt_part = """
        =========
        Chat history:
        {chat_history}
        """

    return f"""{prompt}
                {chat_history_prompt_part}
                =========      
                Question: {{question}}
                =========
                Documents: 
                {{context}}
                =========
                Answer in Markdown format:
            """


