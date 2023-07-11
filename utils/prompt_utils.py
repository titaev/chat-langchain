def prompt_with_system_info(prompt):
    return f"""{prompt}
        
Question: {{question}}
=========
Documents: 
{{context}}
=========
Answer in Markdown format:
        """


