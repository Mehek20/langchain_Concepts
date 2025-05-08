from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name="claude-3")

prompt1 = PromptTemplate(
    template = 'generate short and simple note from the following text: \n {text}',
    input_variables = ['text']
)
prompt2 = PromptTemplate(
    template= 'generate 5 short question answer from the following text: \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document. \n {notes} \n {quiz}',
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({"text":text})
print(result)

# The above code is a parallel chain that generates a short note and quiz from a given text using two different models (ChatOpenAI and ChatAnthropic) and then merges the results into a single document using another model (ChatOpenAI).
# The final result is printed out.
# The graph will show the flow of data from the prompts to the models and then to the output parser.
# The graph will also show the parallel execution of the two models for generating notes and quiz.
# The final merge step will be shown as a separate node in the graph.
# The graph will help in understanding the flow of data and the parallel execution of the models in the chain.

chain.get_graph().print_ascii() ## This will print the graph of the chain in ASCII format.