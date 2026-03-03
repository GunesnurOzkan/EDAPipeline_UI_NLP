import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLPInsightGenerator:
    def __init__(self, model_name="t5-small"):
        """
        Initializes the NLP Insight Generator with a specified T5 model.
        t5-small is used by default for faster local inference.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_insight(self, context_data):
        """
        Generates a natural language insight based on the provided EDA context data.
        
        Args:
            context_data (str): A string summarizing the EDA metrics (e.g., correlations, missing values).
            
        Returns:
            str: Generated insight text.
        """
        prompt = f"Summarize the following data analysis insights in simple terms: {context_data}"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating insight: {str(e)}"

# Example Usage
if __name__ == "__main__":
    generator = NLPInsightGenerator()
    sample_context = "The 'Age' column has 20% missing values. The correlation between 'Height' and 'Weight' is 0.85 (very high)."
    print("Insight:", generator.generate_insight(sample_context))
