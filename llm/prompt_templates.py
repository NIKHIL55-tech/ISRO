prompt_templates = {
            'data_access': """
            Answer the following question about MOSDAC data access using the context below.
            
            Context: {context}
            
            Question: {query}
            
            Provide a step-by-step answer about how to access this data.
            """,
            
            'information': """
            Answer the following question about MOSDAC data using the context below.
            
            Context: {context}
            
            Question: {query}
            
            Provide a clear and concise explanation.
            """,
            
            'technical': """
            Answer the following technical question about MOSDAC data using the context below.
            
            Context: {context}
            
            Question: {query}
            
            Provide technical details and specifications in the answer.
            """
        }
