def prompt_with_system_info(prompt):
    return f"""{prompt}
        
Question: {{question}}
=========
{{context}}
=========
Answer in Markdown format:
        """


