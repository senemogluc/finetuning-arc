import re
import os

COMP_PATH = "outputs/starling-outputs-eval-0.5/"
OUTPUTS_PATH = "answers/eval-answers/"

matrix_pattern = re.compile(r"((\s*\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*)+)")
total = 0
correct = 0
for file in os.listdir(COMP_PATH):

    with open(COMP_PATH + file, mode='r') as f:
        llm_output = f.read()
    with open(OUTPUTS_PATH + file.replace('comp', 'out'), mode='r') as f:
        expected_matrix = f.read()
    total += 1
    print("Testing File: " + file)
    print("Expected Matrix:")
    print(expected_matrix)
    matrices = matrix_pattern.findall(llm_output)
    if matrices:
        matrix = matrices[-1]
        matrix = "  " + matrix[0].strip()
        print("Matrix: ")
        print(matrix)
        if matrix == expected_matrix:
            correct += 1
            print("Correct!")
        
print("Correct: " , correct, "/", total)
