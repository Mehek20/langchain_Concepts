from abc import ABC, abstractmethod

## Runnable class

class Runnable(ABC):
  @abstractmethod
  def invoke(input_data):
    pass

## LLM Class
import random
class nakliLLM(Runnable):

  def __init__(self):
    print('LLM created')

  def invoke(self,prompt):
    response_list=['Delhi is the capital of India',
                   'IPL is a cricket legue',
                   'AI stands for Artificial Intelligence'
                   ]
    return {'responce': random.choice(response_list)}



  def predict(self,prompt):
    response_list=['Delhi is the capital of India',
                   'IPL is a cricket legue',
                   'AI stands for Artificial Intelligence'
                   ]
    return {'responce': random.choice(response_list)}

## Template Class

class nakliPromptTemplate(Runnable):

  def __init__(self,template,input_variables):
    self.template = template
    self.input_variables = input_variables

  def invoke(self,input_dict):
    return self.format(**input_dict)

  def format(self,input_dict):
    return self.template.format(**input_dict)

class NakliStrOutputParser(Runnable):

  def __init__(self):
    pass

  def invoke(self,input_data):
    return input_data['responce']

class RunnableConnector(Runnable):

  def __init__(self,runnable_list):
    self.runnable_list = runnable_list

  def invoke(self,input_data):
    for runnable in self.runnable_list:
      input_data = runnable.invoke(input_data)

    return input_data

template = nakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length','topic']
)

llm = nakliLLM()

parser = NakliStrOutputParser()

chain = RunnableConnector([template,llm,parser])

chain.invoke({'length':'short','topic':'AI'})