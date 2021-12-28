# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:57:28 2021

@author: kusma
"""

from pyrouge import Rouge155
import os
from time import process_time

r=Rouge155(r'C:/Users/kusma/PycharmProjects/pythonProject/HW3/pyrouge-master/tools/ROUGE-1.5.5')
r.model_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/Human_Summaries/eval'
r.system_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/System_Summaries/Centroid' 
r.model_filename_pattern='D#ID#.M.100.T.[A-Z]'
r.system_filename_pattern='d(\d+)t.Centroid'
time1 = process_time()
centroid_value=r.convert_and_evaluate()
print(centroid_value)
output_dict=r.output_to_dict(centroid_value)


r.model_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/Human_Summaries/eval'
r.system_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/System_Summaries/DPP'
r.model_filename_pattern='D#ID#.M.100.T.[A-Z]'
r.system_filename_pattern='d(\d+)t.DPP'
DPP_value=r.convert_and_evaluate()
print(DPP_value)
output_dict_1=r.output_to_dict(DPP_value)


r.model_dir= 'C:/Users/kusma/PycharmProjects/pythonProject/HW3/Human_Summaries/eval'
r.system_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/System_Summaries/ICSISumm'
r.model_filename_pattern='D#ID#.M.100.T.[A-Z]'
r.system_filename_pattern='d(\d+)t.ICSISumm'
ICSISumm_value=r.convert_and_evaluate()
print(ICSISumm_value)
output_dict_2=r.output_to_dict(ICSISumm_value)


r.model_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/Human_Summaries/eval'
r.system_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/System_Summaries/LexRank'
r.model_filename_pattern='D#ID#.M.100.T.[A-Z]'
r.system_filename_pattern='d(\d+)t.LexRank'
LexRank_value=r.convert_and_evaluate()
print(LexRank_value)
output_dict_3=r.output_to_dict(LexRank_value)



r.model_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/Human_Summaries/eval'
r.system_dir='C:/Users/kusma/PycharmProjects/pythonProject/HW3/System_Summaries/Submodular'
r.model_filename_pattern='D#ID#.M.100.T.[A-Z]'
r.system_filename_pattern='d(\d+)t.Submodular'
Submodular_value=r.convert_and_evaluate()
print(Submodular_value)
time2 = process_time()
print("Time taken:", time2-time1)
output_dict_4=r.output_to_dict(Submodular_value)
