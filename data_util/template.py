template_0shot =  """You are a helpful data analyst. I'll give you a tabular dataset's task description, features, and label classes from which you will make a classification prediction for a new instance. No analyzing, directly give the prediction answer, which is only words in your response, there can only be one category of prediction.

Task description: ${task_description}
Features: ${features}
Target label classes: ${label_str}

Now use the provided metadata to infer about the label of this new instance:
${note}"""

# GTL's numerical features need <NUM_BEGIN> -> {self.tokenizer.num_begin_token} and {self.tokenizer.num_end_token}
template_0shot_gtl = """${role_prompt}
${task_prompt}
${answer_prompt}
Features:${note}
Answer:"""