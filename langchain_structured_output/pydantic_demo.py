from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = 'meow' # setting default value
    age: Optional[int] = None # setting optional value
    email: EmailStr  # checking valid email
    cgpa: float = Field(gt=0,lt=10,default=5,description='decimal value representing the cgpa of student') # checking the value is between 0 and 10 through Field

new_student = {'name':'John','age':30,'email':'jhon123@gmail.com','cgpa':8.5} # creating a dictionary with the values of the student

student = Student(**new_student)

student_dict = student.model_dump() # convert the pydantic model to dict

print(student_dict['age'])
#print(student)

student_json = student.model_dump_json() # convert the pydantic model to json