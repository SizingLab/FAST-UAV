import matlab.engine
import numpy as np

inputs=1
eng = matlab.engine.start_matlab()
print("Made it")
outputs = eng.doc_multicopter(1 ,8 ,6 ,0.24 ,2 ,0.03 ,0.015 ,inputs ,2)
print(outputs)
eng.quit()