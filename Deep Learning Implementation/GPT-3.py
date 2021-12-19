import openai
import os
import jsonlines

openai.organization = "[Insert your own code]"
openai.api_key = "[Insert your own code]"

filename = '/content/drive/MyDrive/para/7.txt'

dict1 = {}
  
with open(filename) as fh:
  
    for line in fh:
        line = 'text' + '$%' + line
        command, description = line.strip().split('$%', 1)
        dict1[command] = description.strip()
  
with jsonlines.open('7.jsonl', mode='w') as writer:
    writer.write(dict1)
    writer.close()

openai.File.create(
  file=open("/content/7.jsonl"),
  purpose='answers')

openai.Answer.create(
  search_model="davinci",
  model="davinci",
  question="When have human rights been discussed in the context of COVID-19?",
  documents=["7.jsonl"],
  #file = "file-nYWFf5V4zKtZMv82WyakRZme",
  examples_context="In 2017, U.S. life expectancy was 78.6 years.",
  examples=[["What is human life expectancy in the United States?","78 years."]],
  max_tokens=50,
  temperature = 0.3,
  stop=["\n", "<|endoftext|>"],
)
