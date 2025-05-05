from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = 'meow' # setting default value
    age: Optional[int] = None # setting optional value
    email: EmailStr  # checking valid email
    cgpa: float = Field(...,gt=0,lt=10,default=5,description='decimal value representing the cgpa of student') # checking the value is between 0 and 10 through Field

new_student = {'name':'John','age':30}

student = Student(**new_student)

print(student)