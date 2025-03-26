**描述**  
1.請直接使用v3版的測試問題集，他和deepseek本身的測試格是一樣  
2.competition math 是text2text的數學簡答集，預期LLM會輸出運算過程，並得出最終答案  
3.\\boxed{final anwer}  
4.format_prompt.py是將few_shot串到問題前: question = format_test_prompt_with_few_shot(question)
