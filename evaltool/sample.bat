@set Input_qaPariFile=.\sample.QApair.txt
@set Input_resultFile=.\sample.score.txt
@set Output_metricResultFile=.\sample.result.txt

call .\test.exe %Input_qaPariFile% %Input_resultFile% %Output_metricResultFile%

pause