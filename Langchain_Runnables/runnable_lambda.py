## Runnable lambda
## This runnable lambda is used to make any python function a runnable in Langchain.


from langchain.schema.runnable import RunnableLambda

## Defining a simple function that counts the number of words in a given text.
def word_count(text):
    return len(text.split())

runnable_wc = RunnableLambda(word_count) ## converting the function to a runnable

print(runnable_wc.invoke("This is a simple test sentence."))  ## invoking the runnable with a sample text